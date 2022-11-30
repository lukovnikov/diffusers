import argparse
import inspect
import math
import os
from copy import deepcopy
from pathlib import Path
from typing import Optional
import shutil
import numpy as np

import torch
import torch.nn.functional as F

from accelerate import Accelerator
from accelerate.logging import get_logger
from datasets import load_dataset
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel, DiffusionPipeline
from diffusers.optimization import get_scheduler
from diffusers.schedulers.scheduling_ddim import DDIMExtendedScheduler, DistilledDDIMScheduler
from diffusers.training_utils import EMAModel
from huggingface_hub import HfFolder, Repository, whoami
from torchvision.transforms import (
    CenterCrop,
    Compose,
    InterpolationMode,
    Normalize,
    RandomHorizontalFlip,
    Resize,
    ToTensor,
)
from tqdm.auto import tqdm

from examples.unconditional_image_generation.sample_unconditional import _ddim_scheduler_from_ddpm_scheduler

logger = get_logger(__name__)


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


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
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
        "--load_dir",
        type=str,
        help="The directory where the teacher model is stored.",
    )
    parser.add_argument(
        "--output_subdir",
        type=str,
        default="distilled",
        help="The directory where the student models and logs are stored.",
    )
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )

    parser.add_argument(
        "--distill_schedule",
        type=str,
        default="512 256 128 64 32 16 8 4 2 1",
        help="Number of steps to use in every distillation phase"
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
    parser.add_argument("--save_images_epochs", type=int, default=10, help="How often to save images during training.")
    parser.add_argument(
        "--save_model_epochs", type=int, default=10, help="How often to save the model during training."
    )
    parser.add_argument(
        "--store_model_epochs",
        type=int,
        default=-1,
        help=(
            "How often to store the model during training. Different from --save_model_epoch, all these saved models"
            " are retained after training."
        ),
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
        "--dropout",
        type=float,
        default=0.0,
        help="Dropout to be used during training (default=0).",
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
    parser.add_argument(
        "--use_ema",
        action="store_true",
        default=True,
        help="Whether to use Exponential Moving Average for the final model weights.",
    )
    parser.add_argument("--ema_inv_gamma", type=float, default=1.0, help="The inverse gamma value for the EMA decay.")
    parser.add_argument("--ema_power", type=float, default=3 / 4, help="The power value for the EMA decay.")
    parser.add_argument("--ema_max_decay", type=float, default=0.9999, help="The maximum decay magnitude for EMA.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--hub_private_repo", action="store_true", help="Whether or not to create a private repository."
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

    parser.add_argument(
        "--resolution",
        type=int,
        default=None,           # by default, the resolution is determined by the used dataset
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution. The default value of 'None' lets the script choose the resolution based on the dataset"
            " automatically."
        ),
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("You must specify either a dataset name from the hub or a train data directory.")

    if args.resolution is None:  # resolution default
        if args.dataset_name == "cifar10":
            args.resolution = 32
        else:
            args.resolution = 64

    return args


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_grad_norm(model):
    sqsum = 0.0
    for p in model.parameters():
        if p.grad is not None:
            sqsum += (p.grad ** 2).sum().item()
    return np.sqrt(sqsum)


def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


def main(args):
    outputdir = os.path.join(args.load_dir, args.output_subdir)
    logging_dir = os.path.join(outputdir, args.logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        logging_dir=logging_dir,
    )

    pipeline = DiffusionPipeline.from_pretrained(args.load_dir)
    teachermodel = pipeline.unet
    originalscheduler = pipeline.scheduler
    ddimsched = _ddim_scheduler_from_ddpm_scheduler(pipeline.scheduler, _class=DistilledDDIMScheduler)
    pipeline.scheduler = ddimsched

    print(f"Number of parameters: {count_parameters(teachermodel)//1e6:.2f}M")

    # accepts_predict_epsilon = "predict_epsilon" in set(inspect.signature(DDPMScheduler.__init__).parameters.keys())
    #
    # if accepts_predict_epsilon:
    #     noise_scheduler = DDPMScheduler(
    #         num_train_timesteps=args.ddpm_num_steps,
    #         beta_schedule=args.ddpm_beta_schedule,
    #         predict_epsilon=args.predict_epsilon,
    #     )
    # else:
    #     noise_scheduler = DDPMScheduler(num_train_timesteps=args.ddpm_num_steps, beta_schedule=args.ddpm_beta_schedule)

    studentmodel = deepcopy(teachermodel)

    optimizer = torch.optim.AdamW(
        studentmodel.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    augmentations = Compose(
        [
            Resize(args.resolution, interpolation=InterpolationMode.BILINEAR),
            CenterCrop(args.resolution),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize([0.5], [0.5]),
        ]
    )

    if args.dataset_name is not None:
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
            split="train",
        )
    else:
        dataset = load_dataset("imagefolder", data_dir=args.train_data_dir, cache_dir=args.cache_dir, split="train")

    if args.dataset_name == "cifar10":
        imgkey = "img"
    else:
        imgkey = "image"

    def transforms(examples):
        images = [augmentations(image.convert("RGB")) for image in examples[imgkey]]
        return {"input": images}

    logger.info(f"Dataset size: {len(dataset)}")

    dataset.set_transform(transforms)
    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * args.num_epochs) // args.gradient_accumulation_steps,
    )

    studentmodel, teachermodel, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        studentmodel, teachermodel, optimizer, train_dataloader, lr_scheduler
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)

    ema_model = EMAModel(studentmodel, inv_gamma=args.ema_inv_gamma, power=args.ema_power, max_value=args.ema_max_decay)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(outputdir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            repo = Repository(outputdir, clone_from=repo_name)

            with open(os.path.join(outputdir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif outputdir is not None:
            os.makedirs(outputdir, exist_ok=True)

    if accelerator.is_main_process:
        run = os.path.split(__file__)[-1].split(".")[0]
        accelerator.init_trackers(run)

    # parse distillation schedule
    distillsched = [int(x) for x in args.distill_schedule.split(" ")]
    distillsched = list(zip(distillsched[:-1], distillsched[1:]))
    distillphase = 0
    prevtimesteps = None

    global_step = 0
    for epoch in range(args.num_epochs):
        studentmodel.train()
        teachermodel.eval()
        progress_bar = tqdm(total=num_update_steps_per_epoch, disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in enumerate(train_dataloader):
            x0 = batch["input"]
            # Sample noise that we'll add to the images
            noise = torch.randn(x0.shape).to(x0.device)
            bsz = x0.shape[0]

            prevnumsteps, numsteps = distillsched[distillphase]
            assert prevnumsteps / numsteps == prevnumsteps // numsteps, "Supporting only whole jump sizes"
            jumpsize = int(prevnumsteps / numsteps)

            # run original pipeline for few steps on noisy images
            ddimsched.set_timesteps(prevnumsteps, prevtimesteps=prevtimesteps)
            oldtimesteps = ddimsched.timesteps.to(noise.device)
            timestepselect = torch.randint(0, oldtimesteps.size(0) - jumpsize, (bsz,))
            init_t = oldtimesteps[timestepselect]

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            alphas_cumprod = ddimsched.alphas_cumprod.to(x0.device)
            alpha_bar_t = _extract_into_tensor(alphas_cumprod, init_t, x0.shape)
            x_t = alpha_bar_t.sqrt() * x0 + (1 - alpha_bar_t).sqrt() * noise
            # _x_t = originalscheduler.add_noise(x0, noise, init_t)

            x_tmk = x_t

            with torch.no_grad():
                # run original sampler for a few steps
                for substep_i in range(jumpsize):
                    start_t, end_t = oldtimesteps[timestepselect + substep_i], oldtimesteps[timestepselect + substep_i + 1]
                    # 1. predict noise model_output
                    model_output = teachermodel(x_tmk, start_t).sample

                    # 2. predict previous mean of image x_t-1 and add variance depending on eta
                    # eta corresponds to η in paper and should be between [0, 1]
                    # do x_t -> x_t-1
                    schedoutput = ddimsched.step(model_output, (start_t, end_t), x_tmk, 0.)
                    x_tmk = schedoutput.prev_sample

            # compute target eps or x0:
            #   last 'image' from previous steps gives the target x_tm1
            if originalscheduler.config.predict_epsilon:
                # sqrt_recip_alphas_cumprod_t = _extract_into_tensor(self.sqrt_recip_alphas_cumprod, jump_start_t, x_t.shape)
                # sqrt_recipm1_alphas_cumprod_t = _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, jump_start_t, x_t.shape)
                alpha_bar_t =  _extract_into_tensor(alphas_cumprod, init_t, x0.shape)
                alpha_bar_tmX = _extract_into_tensor(alphas_cumprod, end_t, x0.shape)
                # eps_star = (x_tmX - alpha_bar_tmX.sqrt() * sqrt_recip_alphas_cumprod_t * x_t) / \
                #            ((1-alpha_bar_tmX).sqrt() -  alpha_bar_tmX.sqrt() * sqrt_recipm1_alphas_cumprod_t)
                eps_star = (x_tmk - (alpha_bar_tmX / alpha_bar_t).sqrt() * x_t) / \
                           ((1 - alpha_bar_tmX).sqrt() - (alpha_bar_tmX * (1 - alpha_bar_t) / alpha_bar_t).sqrt())
                target = eps_star
                # _x0_star = (x_t - (1-alpha_bar_t).sqrt()*eps_star)/alpha_bar_t.sqrt()
                # _test = ((alpha_bar_tmX.sqrt() * _x0_star + (1 - alpha_bar_tmX).sqrt() * eps_star) - x_tmk).abs().max()
            else:
                alpha_bar_t =  _extract_into_tensor(alphas_cumprod, init_t, x0.shape)
                alpha_bar_tmX = _extract_into_tensor(alphas_cumprod, end_t, x0.shape)
                x0_star = (x_tmk - ((1 - alpha_bar_tmX) / (1 - alpha_bar_t)).sqrt() * x_t) /\
                          (alpha_bar_tmX.sqrt() - (alpha_bar_t * (1 - alpha_bar_tmX) / (1 - alpha_bar_t)).sqrt())
                target = x0_star

            # TODO
            # run student model
            with accelerator.accumulate(studentmodel):
                # Predict the noise residual or original image
                model_output = studentmodel(x_t, init_t).sample

                if args.predict_epsilon:
                    loss = F.mse_loss(model_output, noise)  # this could have different weights!
                else:
                    alpha_t = _extract_into_tensor(
                        originalscheduler.alphas_cumprod, init_t, (x0.shape[0], 1, 1, 1)
                    )
                    snr_weights = alpha_t / (1 - alpha_t)
                    # TODO: clipped SNR weighting
                    loss = snr_weights * F.mse_loss(
                        model_output, x0, reduction="none"
                    )  # use SNR weighting from distillation paper
                    loss = loss.mean()

                accelerator.backward(loss)

                assert accelerator.sync_gradients
                gradnorm = compute_grad_norm(studentmodel)
                accelerator.clip_grad_norm_(studentmodel.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                if args.use_ema:
                    ema_model.step(studentmodel)
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0],
                    "gradnorm": gradnorm, "step": global_step}
            if args.use_ema:
                logs["ema_decay"] = ema_model.decay
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
        progress_bar.close()

        accelerator.wait_for_everyone()

        # Generate sample images for visual inspection
        if accelerator.is_main_process:
            if epoch % args.save_images_epochs == 0 or epoch == args.num_epochs - 1:
                pipeline = DDPMPipeline(
                    unet=accelerator.unwrap_model(ema_model.averaged_model if args.use_ema else model),
                    scheduler=noise_scheduler,
                )

                generator = torch.manual_seed(0)
                # run pipeline in inference (sample random noise and denoise)
                images = pipeline(
                    generator=generator,
                    batch_size=args.eval_batch_size,
                    output_type="numpy",
                ).images

                # denormalize the images and save to tensorboard
                images_processed = (images * 255).round().astype("uint8")
                accelerator.trackers[0].writer.add_images(
                    "test_samples", images_processed.transpose(0, 3, 1, 2), epoch
                )

            if epoch % args.save_model_epochs == 0 or epoch == args.num_epochs - 1 or args.store_model_epochs > 0 and epoch % args.store_model_epochs == 0:
                # save the model
                pipeline.save_pretrained(args.output_dir)
                if args.push_to_hub:
                    repo.push_to_hub(commit_message=f"Epoch {epoch}", blocking=False)

            if args.store_model_epochs > 0 and epoch % args.store_model_epochs == 0:
                # copy saved model
                shutil.copytree(args.output_dir, args.output_dir + f"_{epoch}ep")
        accelerator.wait_for_everyone()

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
