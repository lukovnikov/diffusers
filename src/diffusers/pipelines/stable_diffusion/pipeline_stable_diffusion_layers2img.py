# Copyright 2022 The HuggingFace Team. All rights reserved.
#
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
import math
from enum import Enum
from typing import Callable, List, Optional, Union
import numpy as np

import torch

from diffusers.utils import is_accelerate_available
from packaging import version
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

from ...configuration_utils import FrozenDict
from ...models import AutoencoderKL, UNet2DConditionModel
from ...models.attention import CrossAttention, BasicTransformerBlock
from ...pipeline_utils import DiffusionPipeline
from ...schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from ...utils import deprecate, logging
from . import StableDiffusionPipelineOutput
from .safety_checker import StableDiffusionSafetyChecker
from ...pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class CrossAttentionWithMask(CrossAttention):
    pass # TODO: implement cross attention that uses a selective cross-attention mask that controls which words we're allowed to attend to from each pixel

    def forward(self, hidden_states, context=None, mask=None):
        batch_size, sequence_length, _ = hidden_states.shape

        query = self.to_q(hidden_states)

        # unpack context, select right size of cross attention mask
        context, cross_attn_mask = context
        size = int(math.sqrt(hidden_states.size(1)))
        cross_attn_mask = cross_attn_mask[(size, size)]
        # reshape mask for heads and flatten over space  # TODO check that flattened same as hidden_states
        batsize, seqlen, height, width = cross_attn_mask.shape
        cross_attn_mask = cross_attn_mask[:, None]\
            .repeat(1, self.heads, 1, 1, 1)\
            .reshape(batsize * self.heads, seqlen, height, width)\
            .reshape(batsize * self.heads, seqlen, height * width)

        context = context if context is not None else hidden_states
        key = self.to_k(context)
        value = self.to_v(context)

        dim = query.shape[-1]

        query = self.reshape_heads_to_batch_dim(query)
        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)

        # TODO(PVP) - mask is currently never used. Remember to re-implement when used

        # attention, what we cannot get enough of
        if self._use_memory_efficient_attention_xformers:
            hidden_states = self._memory_efficient_attention_xformers(query, key, value, mask=cross_attn_mask)
            # Some versions of xformers return output in fp32, cast it back to the dtype of the input
            hidden_states = hidden_states.to(query.dtype)
        else:
            if self._slice_size is None or query.shape[0] // self._slice_size == 1:
                hidden_states = self._attention(query, key, value, mask=cross_attn_mask)
            else:
                hidden_states = self._sliced_attention(query, key, value, sequence_length, dim, mask=cross_attn_mask)

        # linear proj
        hidden_states = self.to_out[0](hidden_states)
        # dropout
        hidden_states = self.to_out[1](hidden_states)
        return hidden_states

    def _attention(self, query, key, value, mask=None):
        attention_scores = torch.baddbmm(
            torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device),
            query,
            key.transpose(-1, -2),
            beta=0,
            alpha=self.scale,
        )
        attention_scores = attention_scores + torch.log(mask.transpose(1, 2).float())

        attention_probs = attention_scores.softmax(dim=-1)
        # compute attention output

        hidden_states = torch.bmm(attention_probs, value)

        # reshape hidden_states
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states


class LayerMode(Enum):
    NORMAL = "normal"


def unpack_layers(layers):
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


class StableDiffusionPipelineLayers2ImageV1(StableDiffusionPipeline):
    r"""
    Pipeline for text-to-image generation using Stable Diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

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
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please, refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for details.
        feature_extractor ([`CLIPFeatureExtractor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """
    _optional_components = ["safety_checker", "feature_extractor"]

    def new__init__todo(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPFeatureExtractor,
        requires_safety_checker: bool = True,
    ):
        # TODO: cast Unet to different unet that takes cross-attention masks
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

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

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely .If you're checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

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

    def _encode_prompt(self, spec, device, num_images_per_prompt, do_classifier_free_guidance):
        # TODO: encode all the prompts for different layers and create cross-attention masks
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `list(int)`):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
        """
        # TODO/ current implementation only supports batch size 1 !!!!
        batch_size = 1 # if not isinstance(prompt, list) else len(prompt)

        # V1 Layers to image: encode prompts for each layer separately, then concatenate
        # Run tokenizer on all layers
        spec["global"] = self._tokenize_dict(spec["global"])
        spec["layers"] = [self._tokenize_dict(layer) for layer in spec["layers"]]
        ds = [spec["global"]] + spec["layers"]

        # FOR POSITIVE PROMPTS:
        # Collect all prompts in one matrix and build gatherer
        pos_text_ids = [x["pos_ids"] for x in ds]
        pos_attn_masks = [x["pos_attn_mask"] for x in ds]
        maxlen = max([x.sum().cpu().item() for x in pos_attn_masks])
        pos_text_ids = [x[:, :maxlen] for x in pos_text_ids]
        pos_attn_masks = [x[:, :maxlen] for x in pos_attn_masks]
        masked_selector = torch.cat(pos_attn_masks, 1).long()
        pos_text_ids_selected = torch.masked_select(torch.cat(pos_text_ids, 1), masked_selector.bool())

        # Run through encoder
        _ids = torch.cat(pos_text_ids, 0)
        _attn_mask = torch.cat(pos_attn_masks, 0)

        _layerids = torch.ones_like(_ids) * torch.arange(0, len(pos_text_ids), dtype=torch.long, device=_ids.device)[:, None]

        text_embeddings = self.text_encoder(_ids.to(device), attention_mask=_attn_mask.to(device))
        text_embeddings = text_embeddings[0]
        _pos_text_embeddings = text_embeddings
        dim = text_embeddings.size(-1)

        # Gather according to structure; keep track of which layer each position in merged sequence belongs to
        text_embeddings = torch.masked_select(text_embeddings.view(-1, text_embeddings.size(-1)),
                                              masked_selector[0, :, None].bool().to(text_embeddings.device))
        text_embeddings = text_embeddings.view(-1, dim)[None]     # (1, seqlen, dim)
        pos_layer_ids = torch.masked_select(_layerids.view(-1), masked_selector[0, :].bool().to(_layerids.device))

        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings[:, None, :, :].repeat(1, num_images_per_prompt, 1, 1)
        text_embeddings = text_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)
        pos_text_embeddings = text_embeddings

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            # Collect all prompts in one matrix and build gatherer
            neg_text_ids = [x["neg_ids"] for x in ds]
            neg_attn_masks = [x["neg_attn_mask"] for x in ds]
            maxlen = max([x.sum().cpu().item() for x in neg_attn_masks])
            neg_text_ids = [x[:, :maxlen] for x in neg_text_ids]
            neg_attn_masks = [x[:, :maxlen] for x in neg_attn_masks]
            masked_selector = torch.cat(neg_attn_masks, 1).long()
            neg_text_ids_selected = torch.masked_select(torch.cat(neg_text_ids, 1), masked_selector.bool())

            # Run through encoder
            _ids = torch.cat(neg_text_ids, 0)
            _attn_mask = torch.cat(neg_attn_masks, 0)

            _layerids = torch.ones_like(_ids) * torch.arange(0, len(neg_text_ids), dtype=torch.long,
                                                             device=_ids.device)[:, None]

            text_embeddings = self.text_encoder(_ids.to(device), attention_mask=_attn_mask.to(device))
            text_embeddings = text_embeddings[0]
            dim = text_embeddings.size(-1)

            # Gather according to structure; keep track of which layer each position in merged sequence belongs to
            text_embeddings = torch.masked_select(text_embeddings.view(-1, text_embeddings.size(-1)),
                                                  masked_selector[0, :, None].bool().to(text_embeddings.device))
            text_embeddings = text_embeddings.view(-1, dim)[None]  # (1, seqlen, dim)
            neg_layer_ids = torch.masked_select(_layerids.view(-1), masked_selector[0, :].bool().to(_layerids.device))

            bs_embed, seq_len, _ = text_embeddings.shape
            text_embeddings = text_embeddings[:, None, :, :].repeat(1, num_images_per_prompt, 1, 1)
            text_embeddings = text_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)
            neg_text_embeddings = text_embeddings

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            # text_embeddings = torch.cat([neg_text_embeddings, pos_text_embeddings], 0)

            return (pos_text_embeddings, pos_layer_ids.to(device)), (neg_text_embeddings, neg_layer_ids.to(device))
        return (pos_text_embeddings, pos_layer_ids.to(device)), None

    def check_inputs(self, spec, height, width, callback_steps):
        # if not isinstance(prompt, str) and not isinstance(prompt, list):
        #     raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    def _compute_cross_attn_masks(self, spec, device):
        layermasks = {}
        for layer in spec["layers"]:
            for k in layer:
                if isinstance(k, tuple):
                    if k not in layermasks:
                        layermasks[k] = []
                    layermasks[k].append(layer[k])
        for k in layermasks:
            # go backwards from the top and subtract 100% opacity parts
            subacc = torch.zeros_like(layermasks[k][0]).to(torch.bool)
            newlayermasks = []
            for layermask in layermasks[k][::-1]:
                newlayermask = layermask > 1/256
                newlayermask = newlayermask & (~subacc)
                opaque = layermask > 254/256
                subacc = subacc | opaque
                newlayermasks.append(newlayermask)
            layermasks[k] = newlayermasks[::-1]
            # Add global mask; this mask does not interact with others
            layermasks[k] = [torch.ones_like(layermasks[k][0])] + layermasks[k]
            layermasks[k] = torch.stack(layermasks[k], 0).to(device)
            # layermasks[k] = torch.index_select(layermasks[k], 0, layer_ids)
        return layermasks

    @torch.no_grad()
    def __call__(
        self,
        layers,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
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
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
                deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
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

        spec = unpack_layers(layers)
        # 1. Check inputs. Raise error if not correct
        self.check_inputs(spec, height, width, callback_steps)

        # 2. Define call parameters
        batch_size = 1 # if isinstance(prompt, str) else len(prompt)
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        output = self._encode_prompt(
            spec, device, num_images_per_prompt, do_classifier_free_guidance
        )

        append_pos_to_neg = True
        text_embeddings, layer_ids = output[0]
        _cross_attention_masks = self._compute_cross_attn_masks(spec, device)
        cross_attention_masks = {res: torch.index_select(_cross_attention_masks[res], 0, layer_ids) for res in _cross_attention_masks}
        cross_attention_masks = {res: m[None].repeat(num_images_per_prompt, 1, 1, 1) for res, m in cross_attention_masks.items()}
        if do_classifier_free_guidance:
            neg_text_embeddings, neg_layer_ids = output[1]
            _neg_cross_attention_masks = _cross_attention_masks
            if append_pos_to_neg:
                neg_text_embeddings = torch.cat([neg_text_embeddings, text_embeddings], 1)
                extra_layer_ids = layer_ids + neg_layer_ids.max() + 1
                neg_layer_ids = torch.cat([neg_layer_ids, extra_layer_ids], 0)
                neg_cross_attention_masks = {
                    res : torch.cat([_neg_cross_attention_masks[res], ~_cross_attention_masks[res]], 0)
                    for res in _neg_cross_attention_masks
                }
            neg_cross_attention_masks = {res: torch.index_select(neg_cross_attention_masks[res], 0, neg_layer_ids) for res in neg_cross_attention_masks}
            neg_cross_attention_masks = {res: m[None].repeat(num_images_per_prompt, 1, 1, 1) for res, m in
                                     neg_cross_attention_masks.items()}

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
            text_embeddings.dtype if not isinstance(text_embeddings, tuple) else text_embeddings[0].dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                # latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latents, t)

                # predict the noise residual
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=(text_embeddings, cross_attention_masks)).sample

                # perform guidance
                if do_classifier_free_guidance:
                    neg_noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=(neg_text_embeddings, neg_cross_attention_masks)).sample
                    noise_pred = guidance_scale * noise_pred + (1 - guidance_scale) * neg_noise_pred
                    # noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    # noise_pred = neg_noise_pred + guidance_scale * (noise_pred - neg_noise_pred)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # 8. Post-processing
        image = self.decode_latents(latents)

        # 9. Run safety checker
        # image, has_nsfw_concept = self.run_safety_checker(image, device, text_embeddings.dtype)
        has_nsfw_concept = False

        # 10. Convert to PIL
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)

    def convert(self):
        # adapt unet to use StructuredCrossAttention
        print("replacing CrossAttention in BasicTransformerBlock with CrossAttentionWithMask")
        _c = 0
        for m in self.unet.modules():
            if isinstance(m, BasicTransformerBlock):
                m.attn2.__class__ = CrossAttentionWithMask
                if m.only_cross_attention:
                    m.attn1.__class__ = CrossAttentionWithMask
                _c += 1
        print(f"Replaced {_c} modules")
        return self