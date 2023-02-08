import argparse
import hashlib
import itertools
import json
import math
import os
import pathlib
from functools import partial
from pathlib import Path
from typing import Optional
import subprocess

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel, StableDiffusionPipeline, \
    DDIMScheduler, DPMSolverMultistepScheduler
from diffusers.models.attention import BasicTransformerBlock, CrossAttention
from diffusers.models.customnn import StructuredCrossAttention, \
    StructuredCLIPTextTransformer
from diffusers.optimization import get_scheduler
from huggingface_hub import HfFolder, Repository, whoami
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig


logger = get_logger(__name__)


def execute(cmd):
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)


def count_parameters(model):
    if isinstance(model, torch.nn.Module):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:   # assuming iterable of parameters
        return sum(p.numel() for p in model)


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        required=True,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--instance_prompts",
        type=str,
        default=None,
        required=True,
        help="The prompt with identifier specifying the instance",
    )
    parser.add_argument(
        "--instance_type",
        type=str,
        default="thing",
        required=False,
        help="The type of instance (thing or person)",
    )
    parser.add_argument(
        "--class_prompts",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="The weight of prior preservation loss.")
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help=(
            "Minimal class images for prior preservation loss. If there are not enough images already present in"
            " class_data_dir, additional images will be sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="text-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop", action="store_true", help="Whether to center crop images before resizing to resolution"
    )
    parser.add_argument("--train_text_encoder", action="store_true", help="Whether to train the text encoder")
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=4, help="Batch size (per device) for sampling images."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument("--generate_every", type=int, default=500, help="Generate images using script every X updates steps.")
    parser.add_argument("--generate_script", type=str, default="generate_during_train_script.py", help="Script to use to generate during training.")
    parser.add_argument("--generate_concept", type=str, default=None, help="The name of the concept to be pasted into prompts from generation script.")
    parser.add_argument("--generate_concept_class", type=str, default=None, help="The name of the concept class to be pasted into prompts from generation script.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
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
        "--lr_warmup_steps", type=int, default=10, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
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
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    parser.add_argument(
        "--train_kv_emb_only", action="store_true", help="When enabled, only KV params of cross-attentions are trained, as well as the parameters of the special token(s). (~Custom Diffusion)"
    )
    parser.add_argument(
        "--train_emb_only", action="store_true", help="When enabled, only parameters of the special token(s) embeddings are trained. (~Textual Inversion)"
    )
    parser.add_argument(
        "--train_textenc_only", action="store_true", help="When enabled, only parameters of the text encoder are trained."
    )
    parser.add_argument(
        "--train_emb_mem", action="store_true", help="When enabled, only parameters of the text encoder are trained."
    )

    parser.add_argument(
        "--initialize_extra_tokens",
        type=str,
        default=None,
        help="Specification of which tokens are source extra tokens and how to initialize them. "
             "Format example: '<sophia_winsell>=woman;<alvin_dog>=dog"
    )
    parser.add_argument(
        "--num_vectors_per_extra_token",
        type=int,
        default=1,
        help="Number of vectors to use for every new source extra token."
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    assert sum([int(args.train_kv_emb_only), int(args.train_emb_only), int(args.train_textenc_only), int(args.train_emb_mem)]) <= 1

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.with_prior_preservation:
        if args.class_data_dir is None:
            raise ValueError("You must specify a data directory for class images.")
        if args.class_prompts is None:
            raise ValueError("You must specify prompt for class images.")
    else:
        if args.class_data_dir is not None:
            logger.info("You need not use --class_data_dir without --with_prior_preservation.")
        if args.class_prompts is not None:
            logger.info("You need not use --class_prompt without --with_prior_preservation.")

    return args


class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_root,
        instance_prompts,
        tokenizer,
        class_data_root=None,
        class_prompts=None,
        size=512,
        center_crop=False,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")

        self.instance_images_path = list(Path(instance_data_root).iterdir())
        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompts = instance_prompts if len(instance_prompts) > 1 else instance_prompts * self.num_instance_images
        self._length = self.num_instance_images

        if class_data_root is not None:     # TODO: number of class prompts etc
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
            self.class_prompts = class_prompts if len(class_prompts) > 1 else class_prompts * self.num_class_images
        else:
            self.class_data_root = None

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = Image.open(self.instance_images_path[index % self.num_instance_images])
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)

        instance_prompt = self.instance_prompts[index % self.num_instance_images]
        tokenized = self.tokenizer(
            instance_prompt,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        )

        example["instance_prompt_ids"] = tokenized.input_ids
        example["instance_attention_mask"] = tokenized.attention_mask

        if self.class_data_root:
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)

            class_prompt = self.class_prompts[index % self.num_class_images]
            tokenized = self.tokenizer(
                class_prompt,
                padding="do_not_pad",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
            )
            example["class_prompt_ids"] = tokenized.input_ids
            example["class_attention_mask"] = tokenized.attention_mask

        return example


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"



def collate_fn(tokenizer, examples):
    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]
    attention_masks = [example["instance_attention_mask"] for example in examples]

    # Concat class and instance examples for prior preservation.
    # We do this to avoid doing two forward passes.
    if args.with_prior_preservation:
        input_ids += [example["class_prompt_ids"] for example in examples]
        pixel_values += [example["class_images"] for example in examples]
        attention_masks += [example["class_attention_mask"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    maxlen = max([len(x) for x in input_ids])
    maxlen = min(maxlen, tokenizer.model_max_length)
    # maxlen = tokenizer.model_max_length

    padded = tokenizer.pad(
        {"input_ids": input_ids, "attention_mask": attention_masks},
        padding="max_length",
        max_length=maxlen,
        return_tensors="pt",
    )

    batch = {
        "input_ids": padded.input_ids,
        "attention_mask": padded.attention_mask,
        "pixel_values": pixel_values,
    }
    return batch


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        logging_dir=logging_dir,
    )

    # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
    # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
    # TODO (patil-suraj): Remove this check when gradient accumulation with two models is enabled in accelerate.
    if args.train_text_encoder and args.gradient_accumulation_steps > 1 and accelerator.num_processes > 1:
        raise ValueError(
            "Gradient accumulation is not supported when training the text encoder in distributed training. "
            "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
        )

    if args.seed is not None:
        set_seed(args.seed)

    if args.with_prior_preservation:
        class_images_dir = Path(args.class_data_dir)
        if not class_images_dir.exists():
            class_images_dir.mkdir(parents=True)
        cur_class_images = len(list(class_images_dir.iterdir()))

        if cur_class_images < args.num_class_images:
            torch_dtype = torch.float16 if accelerator.device.type == "cuda" else torch.float32
            pipeline = DiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                torch_dtype=torch_dtype,
                safety_checker=None,
                revision=args.revision,
            )
            use_dpm = True
            if use_dpm:
                numsteps = 64
                print(f"Using DPM with {numsteps} steps")
                oldsched = pipeline.scheduler
                pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
                pipeline.scheduler.set_timesteps(numsteps)
                assert torch.allclose(oldsched.alphas_cumprod, pipeline.scheduler.alphas_cumprod)
            pipeline.set_progress_bar_config(disable=True)

            num_new_images = args.num_class_images - cur_class_images
            logger.info(f"Number of class images to sample: {num_new_images}.")
            class_prompts = args.class_prompts.split(";")
            i = 0
            for class_prompt in class_prompts:
                sample_dataset = PromptDataset(class_prompt, num_new_images // len(class_prompts))
                sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=args.sample_batch_size)

                sample_dataloader = accelerator.prepare(sample_dataloader)
                pipeline.to(accelerator.device)

                for example in tqdm(
                    sample_dataloader, desc="Generating class images", disable=not accelerator.is_local_main_process
                ):
                    images = pipeline(example["prompt"]).images

                    for _, image in enumerate(images):
                        hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                        image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                        image.save(image_filename)
                        i += 1

            del pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            logger.info("Using already sampled images.")
    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name,
            revision=args.revision,
            use_fast=False,
        )
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )

    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path)

    # Load models and create wrapper for stable diffusion
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.revision,
    ).to(torch.float32)


    #### CUSTOM 3000 extra token, done for all methods just for coding convenience)
    print(f"Adding 3000 extra tokens to tokenizer")
    tokenizer.add_tokens([f"<extra-token-{i}>" for i in range(3000)])
    tokenizervocab = tokenizer.get_vocab()

    print(f"Extending token embedder with 3000 extra token vectors, randomly initialized")
    embed = text_encoder.text_model.embeddings.token_embedding
    extra_embed = torch.nn.Embedding(3000, embed.embedding_dim)
    embed.weight.data = torch.cat([embed.weight.data, extra_embed.weight.data.to(embed.weight.dtype)])
    embed.num_embeddings += 3000

    print(f"Initializing extra tokens from spec")
    extra_token_map = {}            # will be used during training to replace source special token with regular special tokens
    extra_token_spec = {}
    if args.initialize_extra_tokens is not None:  # "<sophia_winsell>=woman;<alvin_dog>=dog"
        extra_token_inits = args.initialize_extra_tokens.split(";")
        for extra_token_init in extra_token_inits:
            sourcetoken, initword = extra_token_init.split("=")
            extra_token_spec[sourcetoken] = initword
            i = len(extra_token_map)
            numvecs = args.num_vectors_per_extra_token
            extra_token_map[sourcetoken] = [f"<extra-token-{i*numvecs+j}>" for j in range(numvecs)]
            initword_id = tokenizer(initword)["input_ids"][1]
            initword_vector = embed.weight.data[initword_id]
            for j, extra_token in enumerate(extra_token_map[sourcetoken]):
                randfactor = (.5 if j > 0 else 0.) if args.train_emb_mem else 0.5
                _initword_vector = initword_vector + torch.randn_like(initword_vector) * initword_vector.std() * randfactor
                # _initword_vector = torch.randn_like(initword_vector) * initword_vector.std()
                embed.weight.data[tokenizervocab[extra_token], :] = _initword_vector
    if len(extra_token_spec) == 1:
        generate_concept, generate_concept_class = list(extra_token_spec.items())[0]
        print(f"Automatically inferred generate concept and class: {generate_concept}=>{generate_concept_class}")

    extra_token_map_ids = {k: [tokenizervocab[vi] for vi in v] for k, v in extra_token_map.items()}     # only used in model saving --> save memory ids too

    if args.train_emb_mem:
        # adapt unet to use StructuredCrossAttention
        print("replacing CrossAttention in BasicTransformerBlock with StructuredCrossAttention")
        _c = 0
        for m in unet.modules():
            if isinstance(m, BasicTransformerBlock):
                m.attn2.__class__ = StructuredCrossAttention
                if m.only_cross_attention:
                    m.attn1.__class__ = StructuredCrossAttention
                _c += 1
        print(f"Replaced {_c} modules")
        print("Replacing CLIP Encoder with Structured CLIP Encoder")
        text_encoder.text_model.__class__ = StructuredCLIPTextTransformer

        new_extra_token_map = {}
        mem_token_map = {}
        for source_extra_token, extra_tokens in extra_token_map.items():
            new_extra_token_map[source_extra_token] = [extra_tokens[0]]
            mem_token_map[extra_tokens[0]] = extra_tokens[1:]

        extra_token_map = new_extra_token_map       # only used to replace text before running through model

        # compute mem spec
        supercellsize = len(list(mem_token_map.values())[0])
        numsupercells = len(extra_token_map)
        _c = 0
        tokenid_to_mem_map = torch.zeros(embed.num_embeddings, dtype=torch.long)
        mem_to_tokenid_map = torch.zeros(numsupercells, supercellsize, dtype=torch.long)
        for source_extra_token, extra_tokens in extra_token_map.items():
            tokenid_to_mem_map[tokenizervocab[extra_tokens[0]]] = _c + 1
            mem_to_tokenid_map[_c, :] = torch.tensor([tokenizervocab[mem_token] for mem_token in mem_token_map[extra_tokens[0]]])
            _c += 1
        text_encoder.text_model.init_mem(tokenid_to_mem_map=tokenid_to_mem_map, mem_to_tokenid_map=mem_to_tokenid_map)

    ####### END CUSTOM

    vae.requires_grad_(False)
    if not args.train_text_encoder:
        text_encoder.requires_grad_(False)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder.gradient_checkpointing_enable()

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW


    params_to_optimize = list(
        itertools.chain(unet.parameters(), text_encoder.parameters()) if args.train_text_encoder else unet.parameters()
    )
    allparamstooptimize = list(params_to_optimize)

    ####### CUSTOM #######
    toreplace = set()      # parameter names which should be replaced in the base model when reloading finetuned model
    if args.train_emb_only or args.train_emb_mem or args.train_kv_emb_only:
        params_to_optimize = []

    if args.train_emb_only or args.train_emb_mem or args.train_kv_emb_only:
        print("optimizing token embeddings")
        params_to_optimize.append(embed.weight)
        if args.initialize_extra_tokens:
            print("Optimizing extra tokens only (last 3000 ids)")
            embedding_gradient_mask = torch.zeros_like(
                embed.weight[:, 0:1])
            for tokenid in [tokenizervocab[f"<extra-token-{i}>"] for i in range(3000)]:
                embedding_gradient_mask[tokenid] = 1
            embed.register_buffer("gradmask", embedding_gradient_mask)

    if args.train_kv_emb_only:
        print("optimizing kv params")
        # train token embeddings of text encoder --> we need to zero out all words except the finetuned one(s), if it's specified
        for name, submodule in unet.named_modules():
            if isinstance(submodule, BasicTransformerBlock):
                # include cross attention
                params_to_optimize += list(submodule.attn2.to_k.parameters())
                params_to_optimize += list(submodule.attn2.to_v.parameters())
                toreplace.add(name + ".attn2.to_k")
                toreplace.add(name + ".attn2.to_v")
                # if transformer is purely cross attention, train other attn module too
                if submodule.only_cross_attention:
                    params_to_optimize += list(submodule.attn1.to_k.parameters())
                    params_to_optimize += list(submodule.attn1.to_v.parameters())
                    toreplace.add(name + ".attn1.to_k")
                    toreplace.add(name + ".attn1.to_v")

    ####### END CUSTOM

    for param in allparamstooptimize:
        param.requires_grad = False
    for param in params_to_optimize:
        param.requires_grad = True

    print(f"#Params to optimize: {count_parameters(list(params_to_optimize))//1e6:.1f}M/{count_parameters(list(allparamstooptimize))//1e6:.1f}M")

    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")


    instance_prompts = args.instance_prompts
    for k, v in extra_token_map.items():
        instance_prompts = instance_prompts.replace(k, " ".join(v))
    instance_prompts = instance_prompts.split(";")
    class_prompts = args.class_prompts.split(";")

    train_dataset = DreamBoothDataset(
        instance_data_root=args.instance_data_dir,
        instance_prompts=instance_prompts,
        class_data_root=args.class_data_dir if args.with_prior_preservation else None,
        class_prompts=class_prompts,
        tokenizer=tokenizer,
        size=args.resolution,
        center_crop=args.center_crop,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=partial(collate_fn, tokenizer), num_workers=1
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    if args.train_text_encoder:
        unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler
        )
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu.
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    vae.to(accelerator.device, dtype=weight_dtype)
    if not args.train_text_encoder:
        text_encoder.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("dreambooth", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    global_step = 0

    for epoch in range(args.num_train_epochs):
        unet.train()
        if args.train_text_encoder:
            text_encoder.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * 0.18215

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"], attention_mask=batch["attention_mask"])[0]

                # Predict the noise residual
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, encoder_mask=batch["attention_mask"]).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                if args.with_prior_preservation:
                    # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                    model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                    target, target_prior = torch.chunk(target, 2, dim=0)

                    # Compute instance loss
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none").mean([1, 2, 3]).mean()

                    # Compute prior loss
                    prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")

                    # Add the prior loss to the instance loss.
                    loss = loss + args.prior_loss_weight * prior_loss
                else:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:

                    text_model = text_encoder.module.text_model if isinstance(text_encoder,
                                  torch.nn.parallel.DistributedDataParallel) else text_encoder.text_model
                    if (args.train_kv_emb_only or args.train_emb_only or args.train_emb_mem) and hasattr(text_model.embeddings.token_embedding, "gradmask"):
                        # apply gradmask on embedding
                        text_model.embeddings.token_embedding.weight.grad \
                            *= text_model.embeddings.token_embedding.gradmask

                    params_to_clip = (
                        itertools.chain(unet.parameters(), text_encoder.parameters())
                        if args.train_text_encoder
                        else unet.parameters()
                    )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step % args.save_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        save_model(unet, text_encoder, save_path, args, accelerator, extra_token_map_ids)

                if args.generate_every > 0 and global_step % args.generate_every == 0:
                    if accelerator.is_main_process:
                        save_model(unet, text_encoder, args.output_dir, args, accelerator, extra_token_map_ids)
                        print("generating")
                        # run generator script command
                        import subprocess
                        subp = subprocess.Popen(["python", args.generate_script,
                                          "--outputdir", args.output_dir,
                                          "--step", str(global_step),
                                          "--concept", generate_concept,
                                          "--conceptclass", generate_concept_class,
                                          "--instancetype", args.instance_type])
                        subp.communicate()  # wait until generation script finishes

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

        accelerator.wait_for_everyone()

    # Create the pipeline using using the trained modules and save it.
    if accelerator.is_main_process:
        save_model(unet, text_encoder, args.output_dir, args, accelerator, extra_token_map_ids)
        if args.push_to_hub:
            repo.push_to_hub(commit_message="End of training", blocking=False, auto_lfs_prune=True)

    accelerator.end_training()


def save_model(unet, text_encoder, path, args, accelerator, extra_token_map):
    # ensure path exists
    Path(path).mkdir(parents=True, exist_ok=True)
    saved = False
    if args.train_emb_only or args.train_emb_mem or args.train_kv_emb_only:
        savedict = {}
        savedict["source"] = args.pretrained_model_name_or_path
        print("Extracting and saving embeddings")
        text_encoder = accelerator.unwrap_model(text_encoder)
        embed = text_encoder.text_model.embeddings.token_embedding
        vectors = {}
        for source_extra_token, extra_token_list in extra_token_map.items():
            vectors[source_extra_token] = [embed.weight.data[extra_token_id, :].cpu().detach() for extra_token_id in extra_token_list]
        savedict["custom_embeddings"] = vectors

        if args.train_emb_mem:
            savedict["use_mem"] = True

        if args.train_emb_only or args.train_emb_mem:
            torch.save(savedict, os.path.join(path, "custom.pth"))
            saved = True

    if args.train_kv_emb_only:
        print(f"Saving key-value projections from cross-attention in Unet.")
        toreplace = set()  # parameter names which should be replaced in the base model when reloading finetuned model
        unet = accelerator.unwrap_model(unet)
        for name, submodule in unet.named_modules():
            if isinstance(submodule, BasicTransformerBlock):
                # include cross attention
                toreplace.add(name + ".attn2.to_k")
                toreplace.add(name + ".attn2.to_v")
                # if transformer is purely cross attention, train other attn module too
                if submodule.only_cross_attention:
                    toreplace.add(name + ".attn1.to_k")
                    toreplace.add(name + ".attn1.to_v")
        pruned_dict = {}
        for k, v in unet.state_dict().items():
            for repl_prefix in toreplace:
                if k.startswith(repl_prefix):
                    pruned_dict[k] = v
                    break
        print(f"Saving {len(pruned_dict)} elements from unet state_dict")
        savedict["unet_state_dict"] = pruned_dict

        torch.save(savedict, os.path.join(path, "custom.pth"))
        saved = True

    # if args.train_emb_mem:
    #     savedict["unet-mem"] = unet.mem
    #     savedict["unet-tokenid_to_mem_map"] = unet.tokenid_to_mem_map
    #     savedict["convert"] = "structured"
    #     torch.save(savedict, os.path.join(path, "custom.pth"))
    #     saved = True

    if saved is False:
        print("Saving the entire model")
        pipeline = DiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            unet=accelerator.unwrap_model(unet),
            text_encoder=accelerator.unwrap_model(text_encoder),
            revision=args.revision,
        )
        pipeline.save_pretrained(path)


def load_model(path, dtype=torch.float16):
    if pathlib.Path(os.path.join(path, "custom.pth")).is_file():
        d = torch.load(os.path.join(path, "custom.pth"))
        pipe = StableDiffusionPipeline.from_pretrained(d["source"], torch_dtype=dtype)
        for k in d["replace"]:
            m = pipe
            splits = k.split(".")
            while len(splits) > 0:
                head, *splits = splits
                m = getattr(m, head)
            if isinstance(m, torch.nn.Parameter):
                m.data = d["replace"][k].data.to(dtype)
            else:
                raise NotImplementedError(f"Types other than torch.nn.Parameter not supported for replacing yet")
    else:
        pipe = StableDiffusionPipeline.from_pretrained(path, torch_dtype=torch.float16)
    return pipe


if __name__ == "__main__":
    args = parse_args()
    main(args)
