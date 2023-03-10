import random

from shutil import copy

from functools import partial

import argparse
import inspect
import logging
import lpips
import math
import os
from copy import deepcopy
from pathlib import Path
from typing import Optional, Union, Tuple, Any, Dict, Iterable

import accelerate
import datasets
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from datasets import load_dataset
from huggingface_hub import HfFolder, Repository, create_repo, whoami
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm

import diffusers
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel, DDIMScheduler, DDIMPipeline
from diffusers.optimization import get_scheduler
from diffusers.schedulers.scheduling_ddim import DDIMSchedulerOutput
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, is_tensorboard_available, is_wandb_available


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.14.0.dev0")

logger = get_logger(__name__, log_level="INFO")


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    if not isinstance(arr, torch.Tensor):
        arr = torch.from_numpy(arr)
    res = arr[timesteps].float().to(timesteps.device)
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


class ShadowEMAModel(torch.nn.Module):
    def __init__(self,
                model,
                decay: float = 0.9999,
                min_decay: float = 0.0,
                update_after_step: int = 0,
                use_ema_warmup: bool = False,
                inv_gamma: Union[float, int] = 1.0,
                power: Union[float, int] = 2 / 3,
                model_cls: Optional[Any] = None,
                model_config: Dict[str, Any] = None,
                **kwargs,
            ):
        super(ShadowEMAModel, self).__init__()
        self.shadow_model = deepcopy(model)
        self.decay = decay
        self.min_decay = min_decay
        self.update_after_step = update_after_step
        self.use_ema_warmup = use_ema_warmup
        self.inv_gamma = inv_gamma
        self.power = power
        self.optimization_step = 0
        self.cur_decay_value = None  # set in `step()`

        self.model_cls = model_cls
        self.model_config = model_config

    def get_decay(self, optimization_step: int) -> float:
        """
        Compute the decay factor for the exponential moving average.
        """
        step = max(0, optimization_step - self.update_after_step - 1)

        if step <= 0:
            return 0.0

        if self.use_ema_warmup:
            cur_decay_value = 1 - (1 + step / self.inv_gamma) ** -self.power
        else:
            cur_decay_value = (1 + step) / (10 + step)

        cur_decay_value = min(cur_decay_value, self.decay)
        # make sure decay is not smaller than min_decay
        cur_decay_value = max(cur_decay_value, self.min_decay)
        return cur_decay_value

    @torch.no_grad()
    def step(self, model: torch.nn.Module):
        named_params = dict(model.named_parameters())
        shadow_params = dict(self.shadow_model.named_parameters())

        self.optimization_step += 1

        # Compute the decay factor for the exponential moving average.
        decay = self.get_decay(self.optimization_step)
        self.cur_decay_value = decay
        one_minus_decay = 1 - decay

        for k in named_params.keys():
            if named_params[k].requires_grad:
                shadow_params[k].sub_(one_minus_decay * (shadow_params[k] - named_params[k].data))
            else:
                shadow_params[k].copy_(named_params[k].data)

        torch.cuda.empty_cache()

    def copy_to(self, model) -> None:
        """
        Copy current averaged parameters into given collection of parameters.

        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                updated with the stored moving averages. If `None`, the parameters with which this
                `ExponentialMovingAverage` was initialized will be used.
        """
        parameters = list(model.parameters)
        for s_param, param in zip(self.shadow_model.parameters(), parameters):
            param.data.copy_(s_param.to(param.device).data)

    def to(self, device=None, dtype=None) -> None:
        r"""Move internal buffers of the ExponentialMovingAverage to `device`.

        Args:
            device: like `device` argument to `torch.Tensor.to`
        """
        # .to() on the tensors handles None correctly
        self.shadow_model.to(device=device, dtype=dtype)
        return self

    def state_dict(self) -> dict:
        r"""
        Returns the state of the ExponentialMovingAverage as a dict. This method is used by accelerate during
        checkpointing to save the ema state dict.
        """
        # Following PyTorch conventions, references to tensors are returned:
        # "returns a reference to the state and not its copy!" -
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict
        ret = {
            "decay": self.decay,
            "min_decay": self.min_decay,
            "optimization_step": self.optimization_step,
            "update_after_step": self.update_after_step,
            "use_ema_warmup": self.use_ema_warmup,
            "inv_gamma": self.inv_gamma,
            "power": self.power,
            "shadow_model": self.shadow_model.state_dict(),
        }
        # ret = {k: torch.tensor(v) if isinstance(v, (bool, int, float)) else v for k, v in ret.items()}
        return ret

    def load_state_dict(self, state_dict: dict) -> None:
        r"""
        Args:
        Loads the ExponentialMovingAverage state. This method is used by accelerate during checkpointing to save the
        ema state dict.
            state_dict (dict): EMA state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        # deepcopy, to be consistent with module API
        state_dict = deepcopy(state_dict)

        self.decay = state_dict.get("decay", self.decay)
        if self.decay < 0.0 or self.decay > 1.0:
            raise ValueError("Decay must be between 0 and 1")

        self.min_decay = state_dict.get("min_decay", self.min_decay)
        if not isinstance(self.min_decay, float):
            raise ValueError("Invalid min_decay")

        self.optimization_step = state_dict.get("optimization_step", self.optimization_step)
        if not isinstance(self.optimization_step, int):
            raise ValueError("Invalid optimization_step")

        self.update_after_step = state_dict.get("update_after_step", self.update_after_step)
        if not isinstance(self.update_after_step, int):
            raise ValueError("Invalid update_after_step")

        self.use_ema_warmup = state_dict.get("use_ema_warmup", self.use_ema_warmup)
        if not isinstance(self.use_ema_warmup, bool):
            raise ValueError("Invalid use_ema_warmup")

        self.inv_gamma = state_dict.get("inv_gamma", self.inv_gamma)
        if not isinstance(self.inv_gamma, (float, int)):
            raise ValueError("Invalid inv_gamma")

        self.power = state_dict.get("power", self.power)
        if not isinstance(self.power, (float, int)):
            raise ValueError("Invalid power")

        shadow_model = state_dict.get("shadow_model", None)
        self.shadow_model.load_state_dict(shadow_model)

    @classmethod
    def from_pretrained(cls, path, model_cls) -> "ShadowEMAModel":
        _, ema_kwargs = model_cls.load_config(path, return_unused_kwargs=True)
        ema_config = ema_kwargs["ema_config"]
        model = model_cls.from_pretrained(path)

        ema_model = cls(model, model_cls=model_cls, model_config=model.config)

        ema_model.load_state_dict(ema_config)
        return ema_model

    def save_pretrained(self, path):
        if self.model_cls is None:
            raise ValueError("`save_pretrained` can only be used if `model_cls` was defined at __init__.")

        if self.model_config is None:
            raise ValueError("`save_pretrained` can only be used if `model_config` was defined at __init__.")

        state_dict = self.state_dict()
        state_dict.pop("shadow_model", None)

        self.shadow_model.register_to_config(ema_config=state_dict)
        self.shadow_model.save_pretrained(path)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--no_student_ema",
        default=False,
        action="store_true",
        help=(
            "Do not use ema in training"
        ),
    )

    parser.add_argument(
        "--use_skip_model",
        action="store_true",
        help=(
            "Whether to use skip formulation"
        ),
    )

    parser.add_argument(
        "--no_skip_model",
        action="store_false",
        dest="use_skip_model",
        help=(
            "Whether to use skip formulation"
        ),
    )
    parser.set_defaults(use_skip_model=True)

    parser.add_argument(
        "--use_lpips",
        action="store_true",
        help=(
            "Whether to use LPIPS loss instead of L2."
        ),
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that HF Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--model_config_name_or_path",
        type=str,
        default=None,
        help="The config of the UNet model to train, leave as None to use standard DDPM configuration.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="consistency-64",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=64,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        default=False,
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=16, help="The number of images to generate for evaluation."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "The number of subprocesses to use for data loading. 0 means that the data will be loaded in the main"
            " process."
        ),
    )
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--num_epochs_init", type=int, default=1)
    parser.add_argument("--save_images_epochs", type=int, default=1, help="How often to save images during training.")
    parser.add_argument(
        "--save_model_epochs", type=int, default=10, help="How often to save the model during training."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.95, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-6, help="Weight decay magnitude for the Adam optimizer."
    )
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer.")

    parser.add_argument("--ema_inv_gamma", type=float, default=1.0, help="The inverse gamma value for the EMA decay.")
    parser.add_argument("--ema_power", type=float, default=3 / 4, help="The power value for the EMA decay.")
    parser.add_argument("--ema_max_decay", type=float, default=0.99, help="The maximum decay magnitude for EMA.")

    parser.add_argument(
        "--logger",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb"],
        help=(
            "Whether to use [tensorboard](https://www.tensorflow.org/tensorboard) or [wandb](https://www.wandb.ai)"
            " for experiment tracking and logging of model metrics and model checkpoints"
        ),
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument("--num_noise_levels", type=int, default=1024)
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=(
            "Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`."
            " See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"
            " for more docs"
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )

    args = parser.parse_args()
    args.use_ema = True
    args.checkpointing_steps = int(1e12)

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("You must specify either a dataset name from the hub or a train data directory.")

    return args


def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


def compute_cskip_and_cout(t, T=1000, M=80, sigmadata=0.5):
    t = t * M / T
    cskip = (sigmadata ** 2) / (t ** 2 + sigmadata ** 2)
    cout = 2 * sigmadata * t / torch.sqrt(sigmadata ** 2 + t ** 2)
    return cskip, cout


def add_noise(images, noise, noise_levels, max_noise_levels):
    while noise_levels.dim() < images.dim():
        noise_levels = noise_levels.unsqueeze(-1)

    denoise_progress_t = noise_levels / max_noise_levels
    noise = torch.tanh(noise)       # tanh of gaussian
    x_t = images * (1 - denoise_progress_t) + noise * denoise_progress_t
    return x_t


def main(args):
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(total_limit=args.checkpoints_total_limit)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.logger,
        logging_dir=logging_dir,
        project_config=accelerator_project_config,
    )

    if args.logger == "tensorboard":
        if not is_tensorboard_available():
            raise ImportError("Make sure to install tensorboard if you want to use it for logging during training.")

    elif args.logger == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            modelcount = 0
            for model in models:
                if isinstance(model, ShadowEMAModel):
                    model.save_pretrained(os.path.join(output_dir, f"unet_ema"))
                    modelcount += 1
                else:
                    model.save_pretrained(os.path.join(output_dir, f"unet"))
                    modelcount += 1
                weights.pop()

        def load_model_hook(models, input_dir):
            for model in models:
                if isinstance(model, ShadowEMAModel):
                    load_model = ShadowEMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DModel)
                else:
                    load_model = UNet2DModel.from_pretrained(os.path.join(input_dir, "unet"))
                    model.register_to_config(**load_model.config)
                model.load_state_dict(load_model.state_dict())
                model.to(accelerator.device)
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Initialize the model
    if args.model_config_name_or_path is None:
        model = UNet2DModel(
            sample_size=args.resolution,
            in_channels=3,
            out_channels=3,
            layers_per_block=2,
            block_out_channels=(128, 128, 256, 256),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )
    else:
        config = UNet2DModel.load_config(args.model_config_name_or_path)
        model = UNet2DModel.from_config(config)
    # """

    # Create EMA for the model.
    ema_model = None
    if args.use_ema:
        ema_model = ShadowEMAModel(
            model,
            decay=args.ema_max_decay,
            use_ema_warmup=True,
            inv_gamma=args.ema_inv_gamma,
            power=args.ema_power,
            model_cls=UNet2DModel,
            model_config=model.config,
        )

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
            split="train",
        )
    else:
        dataset = load_dataset("imagefolder", data_dir=args.train_data_dir, cache_dir=args.cache_dir, split="train")
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder

    # Preprocessing the datasets and DataLoaders creation.
    augmentations = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def transform_images(examples):
        images = [augmentations(image.convert("RGB")) for image in examples["image"]]
        return {"input": images}

    logger.info(f"Dataset size: {len(dataset)}")

    dataset.set_transform(transform_images)
    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers
    )

    # Initialize the learning rate scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=(len(train_dataloader) * args.num_epochs),
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler, ema_model = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler, ema_model
    )


    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        run = os.path.split(__file__)[-1].split(".")[0]
        accelerator.init_trackers(run)

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.num_epochs * num_update_steps_per_epoch

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num Epochs = {args.num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)


    # define training phases
    train_steps_per_stage =  int(round(max_train_steps * 0.7)) // (args.num_noise_levels * 2)

    generate_every = args.save_images_epochs * num_update_steps_per_epoch

    print(f"Training has {train_steps_per_stage} steps per noise level stage increase.")

    # get validation images
    validation_images = []
    for step, batch in enumerate(train_dataloader):
        validation_images.append(batch["input"])
        if len(validation_images) * validation_images[-1].size(0) >= args.eval_batch_size:
            break
    validation_images = torch.cat(validation_images, 0)[:args.eval_batch_size]

    updates_left = args.num_epochs_init * num_update_steps_per_epoch

    # define loss
    loss_fn = lpips.LPIPS(net='vgg').to(accelerator.device) if args.use_lpips else partial(F.mse_loss, reduction="none")
    DEBUG = False
    # DEBUG = True
    if DEBUG:
        # current_max_noise_level = 100
        train_steps_per_stage = 1
        generate_every = 1
        updates_left = train_steps_per_stage

    def noise_level_for_model(_noise_level):
        ret = (_noise_level * 1000 / args.num_noise_levels).long()
        return ret

    noise_level_regions = {0: list(range(0, args.num_noise_levels))}
    end_noise_levels = list(noise_level_regions.keys())

    # Train!
    for epoch in range(first_epoch, args.num_epochs):
        model.train()
        progress_bar = tqdm(total=num_update_steps_per_epoch, disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            clean_images = batch["input"]
            # Sample noise that we'll add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bsz = clean_images.shape[0]
            # Sample a random timestep for each image
            # Limit the range of timesteps according to training phase
            # sample ending noise levels uniformly from available end noise levels
            noise_levels_tm1 = random.choices(end_noise_levels, k=bsz)
            # for every end noise level, sample uniformly from the corresponding start noise levels
            noise_levels_t = []
            for noise_level_tm1 in noise_levels_tm1:
                noise_levels_t.append(random.choice(noise_level_regions[noise_level_tm1]))
            # send to right devices
            noise_levels_t = torch.tensor(noise_levels_t).long().to(clean_images.device)
            noise_levels_tm1 = torch.tensor(noise_levels_tm1).long().to(clean_images.device) - 1

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)

            with torch.no_grad():
                x_t = add_noise(clean_images, noise, noise_levels_t + 1, args.num_noise_levels)
                x_tm1 = add_noise(clean_images, noise, noise_levels_tm1 + 1, args.num_noise_levels)
                # Use (ema) model to compute reference x_0 from x_tmk
                referencemodel = ema_model.shadow_model if not args.no_student_ema else model
                x_0_pred = referencemodel(x_tm1, noise_level_for_model(noise_levels_tm1.clamp_min(0))).sample.detach()
                if args.use_skip_model:
                    cskip_tm1, cout_tm1 = compute_cskip_and_cout(noise_levels_tm1 + 1, T=args.num_noise_levels)
                    x_0_pred = cskip_tm1[:, None, None, None] * x_tm1 + cout_tm1[:, None, None, None] * x_0_pred
                else:
                    x_0_pred = torch.where(noise_levels_tm1[:, None, None, None] >= 0, x_0_pred, x_tm1)

            with accelerator.accumulate(model):
                # Predict the noise residual
                model_output = model(x_t, noise_level_for_model(noise_levels_t)).sample
                if args.use_skip_model:
                    cskip_t, cout_t = compute_cskip_and_cout(noise_levels_t + 1, T=args.num_noise_levels)
                    model_output = cskip_t[:, None, None, None] * x_t + cout_t[:, None, None, None] * model_output

                loss = loss_fn(model_output, x_0_pred)     # match ema prediction from x_tmk to model prediction
                loss = loss.mean()

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                updates_left -= 1

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_model.step(model)
                progress_bar.update(1)
                global_step += 1

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            if args.use_ema:
                logs["ema_decay"] = ema_model.cur_decay_value
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step % generate_every == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    unet = accelerator.unwrap_model(model) if not args.use_ema else accelerator.unwrap_model(
                        ema_model.shadow_model)

                    # region sampling
                    unet_training = unet.training
                    unet.eval()
                    with torch.no_grad():
                        # generate using one-step generation from a batch of clean images
                        generator = torch.Generator(device=validation_images.device).manual_seed(0)
                        noise = torch.randn(validation_images.shape, generator=generator, device=validation_images.device)
                        noise_levels_t = torch.tensor([args.num_noise_levels-1], device=validation_images.device)
                        noisy_test_images = add_noise(validation_images, noise, noise_levels_t+1, args.num_noise_levels)

                        model_output = unet(noisy_test_images, noise_level_for_model(noise_levels_t)).sample.detach()
                        if args.use_skip_model:
                            cskip_t, cout_t = compute_cskip_and_cout(noise_levels_t + 1, T=args.num_noise_levels)
                            modelpred = cskip_t[:, None, None, None] * noisy_test_images + cout_t[:, None, None, None] * model_output

                        images = modelpred.detach().cpu().numpy()
                    unet.train(unet_training)

                    images = images * 0.5 + 0.5
                    noisy_test_images = noisy_test_images * 0.5 + 0.5

                    # denormalize the images and save to tensorboard
                    images_processed = (images * 255).round().astype("uint8")
                    noisy_test_images = (noisy_test_images.detach().cpu().numpy() * 255).round().astype("uint8")

                    if args.logger == "tensorboard":
                        accelerator.get_tracker("tensorboard").add_images(
                            "test_denoised", images_processed, global_step
                        )
                        accelerator.get_tracker("tensorboard").add_images(
                            "test_noisy", noisy_test_images, global_step
                        )
                    elif args.logger == "wandb":
                        accelerator.get_tracker("wandb").log(
                            {"test_denoised": [wandb.Image(img) for img in images_processed], "global_step": global_step},
                            step=global_step,
                        )
                        accelerator.get_tracker("wandb").log(
                            {"test_noisy": [wandb.Image(img) for img in noisy_test_images], "global_step": global_step},
                            step=global_step,
                        )

            if updates_left <= 0:
                updates_left = train_steps_per_stage
                # split up noise regions
                end_noise_levels = sorted(list(noise_level_regions.keys()))
                _end_noise_levels = end_noise_levels + [args.num_noise_levels]
                broken = False
                for i in range(len(_end_noise_levels)-1):
                    end_noise_level = _end_noise_levels[i]
                    if len(noise_level_regions[end_noise_level]) > 1:
                        new_end_noise_level = _end_noise_levels[i] + (_end_noise_levels[i+1] -_end_noise_levels[i]) // 2
                        region = noise_level_regions[end_noise_levels[i]]
                        noise_level_regions[end_noise_level] = region[:len(region)//2]
                        noise_level_regions[new_end_noise_level] = region[len(region)//2:]
                        broken = True
                        break
                if broken:
                    print(f"advanced noise regions: now we have {len(noise_level_regions)}")
                end_noise_levels = sorted(list(noise_level_regions.keys()))

        progress_bar.close()

        accelerator.wait_for_everyone()

        # save model
        if accelerator.is_main_process:
            if epoch % args.save_model_epochs == 0 or epoch == args.num_epochs - 1:
                unet = accelerator.unwrap_model(model)
                unet.save_pretrained(os.path.join(args.output_dir, "unet"))
                # copy(os.path.join(args.output_dir, "unet"), os.path.join(args.output_dir, f"unet_step_{global_step}"))
                if args.use_ema:
                    ema_unet = accelerator.unwrap_model(ema_model.shadow_model)
                    ema_unet.save_pretrained(os.path.join(args.output_dir, "unet_ema"))
                    # copy(os.path.join(args.output_dir, "unet_ema"), os.path.join(args.output_dir, f"unet_ema_step_{global_step}"))

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)


# flowers dataset: "huggan/flowers-102-categories"
# flowers model: "anton-l/ddpm-ema-flowers-64"
