# Copyright 2023 MultiDiffusion Authors and The HuggingFace Team. All rights reserved."
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import numpy as np
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

from ...models import AutoencoderKL, UNet2DConditionModel
from ...schedulers import DDIMScheduler, PNDMScheduler
from ...utils import is_accelerate_available, logging, randn_tensor, replace_example_docstring
from ..pipeline_utils import DiffusionPipeline
from . import StableDiffusionPipelineOutput
from .safety_checker import StableDiffusionSafetyChecker


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import StableDiffusionPanoramaPipeline, DDIMScheduler

        >>> model_ckpt = "stabilityai/stable-diffusion-2-base"
        >>> scheduler = DDIMScheduler.from_pretrained(model_ckpt, subfolder="scheduler")
        >>> pipe = StableDiffusionPanoramaPipeline.from_pretrained(
        ...     model_ckpt, scheduler=scheduler, torch_dtype=torch.float16
        ... )

        >>> pipe = pipe.to("cuda")

        >>> prompt = "a photo of the dolomites"
        >>> image = pipe(prompt).images[0]
        ```
"""



class LayerMode(Enum):
    NORMAL = "normal"


def unpack_layers(layers, trilevel=False):
    # Unpacks the layers PSD file, creates masks, preprocesses prompts etc.
    unpacked_layers = []
    global_descriptions = {}
    for layer in layers:
        if layer.name == "Background":      # Ignore layer named "Background"
            continue
        if layer.name.startswith("[GLOBAL]"):
            assert len(global_descriptions) == 0, "Only one global description"
            splits = [x.strip() for x in layer.name[len("[GLOBAL]"):].split("|")]
            pos, neg = splits if len(splits) == 2 else (splits[0], "")
            global_descriptions["pos"] = pos
            global_descriptions["neg"] = neg
            continue
        splits = [x.strip() for x in layer.name.split("|")]
        pos, neg = splits if len(splits) == 2 else (splits[0], "")
        layermatrix = torch.tensor(np.asarray(layer.topil().getchannel("A"))).float() / 255
        # build tri-level matrices
        if trilevel:
            fullmatrix = (layermatrix > 254/255)
            emptymatrix = (layermatrix < 2/256)
            middlematrix = ~(fullmatrix | emptymatrix)
            assert torch.all(torch.ones_like(middlematrix) == (fullmatrix | emptymatrix | middlematrix))
            assert torch.all(torch.zeros_like(middlematrix) == (fullmatrix & emptymatrix))
            assert torch.all(torch.zeros_like(middlematrix) == (fullmatrix & middlematrix))
            assert torch.all(torch.zeros_like(middlematrix) == (emptymatrix & middlematrix))
            layermatrix = 0. * emptymatrix.float() + 1. * fullmatrix.float() + 0.5 * middlematrix.float()
        _layermatrix = layermatrix
        assert layermatrix.size() == (512, 512)
        downsamples = [8, 16, 32, 64]
        unpacked_layers.append({
            "pos": pos,
            "neg": neg,
            "mode": LayerMode.NORMAL,
            tuple(layermatrix.shape): layermatrix,
        })
        for downsample in downsamples:
            downsampled = torch.nn.functional.avg_pool2d(_layermatrix[None, :, :], downsample, downsample)[0, :, :]
            layermatrix = downsampled
            if trilevel:
                fullmatrix = (layermatrix > 254/255)
                emptymatrix = (layermatrix < 2/256)
                middlematrix = ~(fullmatrix | emptymatrix)
                assert torch.all(torch.ones_like(middlematrix) == (fullmatrix | emptymatrix | middlematrix))
                assert torch.all(torch.zeros_like(middlematrix) == (fullmatrix & emptymatrix))
                assert torch.all(torch.zeros_like(middlematrix) == (fullmatrix & middlematrix))
                assert torch.all(torch.zeros_like(middlematrix) == (emptymatrix & middlematrix))
                layermatrix = 0. * emptymatrix.float() + 1. * fullmatrix.float() + 0.5 * middlematrix.float()
            unpacked_layers[-1][tuple(downsampled.shape)] = layermatrix

    return {"layers": unpacked_layers, "global": global_descriptions}



class StableDiffusionPaintbywordsPipeline(DiffusionPipeline):
    r"""
    Pipeline for text-to-image generation using "MultiDiffusion: Fusing Diffusion Paths for Controlled Image
    Generation".

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.).

    To generate panorama-like images, be sure to pass the `width` parameter accordingly when using the pipeline. Our
    recommendation for the `width` value is 2048. This is the default value of the `width` parameter for this pipeline.

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. The original work
            on Multi Diffsion used the [`DDIMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please, refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for details.
        feature_extractor ([`CLIPFeatureExtractor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """
    _optional_components = ["safety_checker", "feature_extractor"]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: DDIMScheduler,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPFeatureExtractor,
        requires_safety_checker: bool = True,
    ):
        super().__init__()

        if isinstance(scheduler, PNDMScheduler):
            logger.error("PNDMScheduler for this pipeline is currently not supported.")

        if safety_checker is None and requires_safety_checker:
            logger.warning(
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )

        if safety_checker is not None and feature_extractor is None:
            raise ValueError(
                "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.register_to_config(requires_safety_checker=requires_safety_checker)

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.enable_vae_slicing
    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding.

        When this option is enabled, the VAE will split the input tensor in slices to compute decoding in several
        steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.vae.enable_slicing()

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.disable_vae_slicing
    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously invoked, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_slicing()

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.enable_sequential_cpu_offload
    def enable_sequential_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, significantly reducing memory usage. When called, unet,
        text_encoder, vae and safety checker have their state dicts saved to CPU and then are moved to a
        `torch.device('meta') and loaded to GPU only when their specific submodule has its `forward` method called.
        Note that offloading happens on a submodule basis. Memory savings are higher than with
        `enable_model_cpu_offload`, but performance is lower.
        """
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            cpu_offload(cpu_offloaded_model, device)

        if self.safety_checker is not None:
            cpu_offload(self.safety_checker, execution_device=device, offload_buffers=True)

    @property
    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline._execution_device
    def _execution_device(self):
        r"""
        Returns the device on which the pipeline's models will be executed. After calling
        `pipeline.enable_sequential_cpu_offload()` the execution device can only be inferred from Accelerate's module
        hooks.
        """
        if not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def _tokenize_dict(self, d):
        tokenizerout = self.tokenizer(d["pos"], padding="max_length", truncation=True,
                                      max_length=self.tokenizer.model_max_length, return_tensors="pt")
        untruncated_ids = self.tokenizer(d["pos"], padding="max_length", return_tensors="pt").input_ids
        if tokenizerout.input_ids.size(1) > untruncated_ids.size(1):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )

        d["pos_ids"] = tokenizerout.input_ids
        d["pos_attn_mask"] = tokenizerout.attention_mask

        tokenizerout = self.tokenizer(d["neg"], padding="max_length", truncation=True,
                                      max_length=self.tokenizer.model_max_length, return_tensors="pt")
        untruncated_ids = self.tokenizer(d["neg"], padding="max_length", return_tensors="pt").input_ids
        if tokenizerout.input_ids.size(1) > untruncated_ids.size(1):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer.model_max_length - 1: -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )
        d["neg_ids"] = tokenizerout.input_ids
        d["neg_attn_mask"] = tokenizerout.attention_mask
        return d

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline._encode_prompt
    def _encode_prompt(
        self,
        spec,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
             prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds`. instead. If not defined, one has to pass `negative_prompt_embeds`. instead.
                Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
        """
        batch_size = 1
        # V1 Layers to image: encode prompts for each layer separately, then concatenate
        # Run tokenizer on all layers
        globalpos = spec["global"]["pos"]
        globalneg = spec["global"]["neg"]
        for layer in spec["layers"]:
            layer["pos"] += ", " + globalpos
            layer["neg"] += ", " + globalneg
        spec["layers"] = [self._tokenize_dict(layer) for layer in spec["layers"]]
        ds = spec["layers"]

        assert prompt_embeds is None

        pos_text_ids = [x["pos_ids"] for x in ds]
        pos_attn_masks = [x["pos_attn_mask"] for x in ds]
        maxlen = max([x.sum().cpu().item() for x in pos_attn_masks])
        pos_text_ids = [x[:, :maxlen] for x in pos_text_ids]
        pos_attn_masks = [x[:, :maxlen] for x in pos_attn_masks]

        _ids = torch.cat(pos_text_ids, 0)
        _attn_mask = torch.cat(pos_attn_masks, 0)
        pos_text_embeddings = self.text_encoder(_ids.to(device), attention_mask=_attn_mask.to(device))[0]
        object_masks = torch.stack([dse[(64,64)] for dse in ds], 0).to(device)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            assert negative_prompt_embeds is None
            assert negative_prompt is None

            neg_text_ids = [x["neg_ids"] for x in ds]
            neg_attn_masks = [x["neg_attn_mask"] for x in ds]
            maxlen = max([x.sum().cpu().item() for x in neg_attn_masks])
            neg_text_ids = [x[:, :maxlen] for x in neg_text_ids]
            neg_attn_masks = [x[:, :maxlen] for x in neg_attn_masks]

            _ids = torch.cat(neg_text_ids, 0)
            _attn_mask = torch.cat(neg_attn_masks, 0)
            neg_text_embeddings = self.text_encoder(_ids.to(device), attention_mask=_attn_mask.to(device))[0]

            return pos_text_embeddings, neg_text_embeddings, object_masks

        return pos_text_embeddings, None, object_masks

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.run_safety_checker
    def run_safety_checker(self, image, device, dtype):
        if self.safety_checker is not None:
            safety_checker_input = self.feature_extractor(self.numpy_to_pil(image), return_tensors="pt").to(device)
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        else:
            has_nsfw_concept = None
        return image, has_nsfw_concept

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.decode_latents
    def decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.check_inputs
    def check_inputs(
        self,
        prompt,
        height,
        width,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if prompt_embeds is not None or negative_prompt_embeds is not None:
            raise ValueError(f"Prompt embeds or negative prompt embeds are not supported in this pipeline.")
        if negative_prompt is not None:
            raise ValueError(f"Negative prompt is not supported here (must be given in the layers file).")


    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def get_views(self, panorama_height, panorama_width, window_size=64, stride=8):
        # Here, we define the mappings F_i (see Eq. 7 in the MultiDiffusion paper https://arxiv.org/abs/2302.08113)
        panorama_height /= 8
        panorama_width /= 8
        num_blocks_height = (panorama_height - window_size) // stride + 1
        num_blocks_width = (panorama_width - window_size) // stride + 1
        total_num_blocks = int(num_blocks_height * num_blocks_width)
        views = []
        for i in range(total_num_blocks):
            h_start = int((i // num_blocks_width) * stride)
            h_end = h_start + window_size
            w_start = int((i % num_blocks_width) * stride)
            w_end = w_start + window_size
            views.append((h_start, h_end, w_start, w_end))
        return views

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        layers = None, # must be a psd image file
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        num_inference_steps: int = 32,
        guidance_scale: float = 7.5,
        bootstrap_ratio:float=0.4,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            height (`int`, *optional*, defaults to 512:
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to 2048):
                The width in pixels of the generated image. The width is kept to a high number because the
                    pipeline is supposed to be used for generating panorama-like images.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds`. instead. If not defined, one has to pass `negative_prompt_embeds`. instead.
                Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttnProcessor` as defined under
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        spec = unpack_layers(layers, trilevel=False)

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            spec, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )

        # 2. Define call parameters
        batch_size = 1
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        pos_text_embeds, neg_text_embeds, object_masks = self._encode_prompt(
            spec,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            pos_text_embeds.dtype,
            device,
            generator,
            latents,
        )
        init_noise = latents.clone()


        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 8. Denoising loop
        # Each denoising step also includes refinement of the latents with respect to the
        # views.

        t_tight_bootstrap = int(self.scheduler.num_train_timesteps * (1 - bootstrap_ratio))   # before this timestep, do constant mask like in paper
        # t_tight_bootstrap = self.scheduler.num_train_timesteps + 1

        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        #"""
        object_masks_alphamask = object_masks > 254 / 255
        current_alphamask = torch.zeros_like(object_masks[-1]).bool()
        for i in list(range(len(object_masks)))[::-1]:
            object_masks_alphamask[i] = (object_masks[i] > 0) & (~current_alphamask)
            current_alphamask = current_alphamask | (object_masks[i] > 254/255)
        # """
        """
        object_masks_alphamask = object_masks
        current_alphamask = torch.zeros_like(object_masks[-1])
        for i in list(range(len(object_masks)))[::-1]:
            object_masks_alphamask[i] = object_masks[i] - current_alphamask
            current_alphamask = (current_alphamask + object_masks[i]).clamp(0, 1)
        # """

        bootstrapping_backgrounds = self.get_random_background(20)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # generate views
                # Here, we iterate through different spatial crops of the latents and denoise them. These
                # denoised (latent) crops are then averaged to produce the final latent
                # for the current timestep via MultiDiffusion. Please see Sec. 4.1 in the
                # MultiDiffusion paper for more details: https://arxiv.org/abs/2302.08113

                latent_model_input = torch.cat([latents] * len(pos_text_embeds), 0)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                if t.cpu().item() > t_tight_bootstrap:      # do tight masks stuff
                    # sample random rgb (a different one for each mask?)
                    latent_bgr = bootstrapping_backgrounds[torch.randint(0, len(bootstrapping_backgrounds), (latent_model_input.size(0),), device=latent_model_input.device)]
                    latent_bgr = self.scheduler.scale_model_input(latent_bgr, t)
                    latent_bgr = self.scheduler.add_noise(latent_bgr, init_noise, t)
                    # prepare unet input by replacing masked regions with corresponding latent_bgr parts
                    latent_model_input = object_masks_alphamask[:, None].float() * latent_model_input \
                                        + (1 - object_masks_alphamask[:, None].float()) * latent_bgr

                pos_noise_preds = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=pos_text_embeds,
                        cross_attention_kwargs=cross_attention_kwargs,
                    ).sample
                noise_pred = pos_noise_preds

                if do_classifier_free_guidance:
                    neg_noise_preds = self.unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=neg_text_embeds,
                            cross_attention_kwargs=cross_attention_kwargs,
                        ).sample
                    noise_pred = neg_noise_preds + guidance_scale * (pos_noise_preds - neg_noise_preds)

                # compute the previous noisy sample x_t -> x_t-1
                latents_view_denoised = self.scheduler.step(
                    noise_pred, t, latent_model_input, **extra_step_kwargs
                ).prev_sample

                # merge different predictions based on objectmask

                count = object_masks_alphamask.sum(0)
                assert torch.all(count > 0)

                latents_view_masked = latents_view_denoised * object_masks_alphamask[:, None]
                latents = latents_view_masked.sum(0) / count[None]
                latents = latents[None]

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # 8. Post-processing
        image = self.decode_latents(latents)

        # 9. Run safety checker
        # image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
        image, has_nsfw_concept = image, False

        # 10. Convert to PIL
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)

    # copied from region_control.py
    @torch.no_grad()
    def get_random_background(self, n_samples):
        # sample random background with a constant rgb value
        backgrounds = torch.rand(n_samples, 3, device=self.device)[:, :, None, None].repeat(1, 1, 512, 512)
        return torch.cat([self.encode_imgs(bg.unsqueeze(0)) for bg in backgrounds])

    @torch.no_grad()
    def encode_imgs(self, imgs):
        imgs = 2 * imgs - 1
        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents
    """
    @torch.no_grad()
    def decode_latents(self, latents):
        latents = 1 / 0.18215 * latents
        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs
    # """

    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt):
        # Tokenize text and get embeddings
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                    truncation=True, return_tensors='pt')
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # Do the same for unconditional embeddings
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                      return_tensors='pt')

        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # Cat for final embeddings
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

