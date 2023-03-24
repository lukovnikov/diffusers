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
import re

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
from ...models.cross_attention import CrossAttnProcessor
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


class MaskedCrossAttention(CrossAttention):

    def get_attention_scores(self, query, key, attention_mask=None, mode="original"):
        dtype = query.dtype
        if self.upcast_attention:
            query = query.float()
            key = key.float()

        attention_scores = torch.bmm(query, key.transpose(-1, -2)) * self.scale

        if mode.startswith("o") or mode.startswith("p"):
            maxqk = torch.max(attention_scores, dim=-1, keepdim=True)[0]
            attention_mask = attention_mask.to(attention_scores.dtype) * maxqk

            attention_scores = attention_scores + attention_mask
        elif mode.startswith("n"):
            attention_scores = attention_scores * attention_mask

        if self.upcast_softmax:
            attention_scores = attention_scores.float()

        attention_probs = attention_scores.softmax(dim=-1)
        attention_probs = attention_probs.to(dtype)

        return attention_probs


class MaskedCrossAttnProcessor(CrossAttnProcessor):
    def __call__(
        self,
        attn: MaskedCrossAttention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,        # self-attention mask
        masks=None,                 # cross-attention masks at different resolutions
        sigma=1.,                   # special scaling factor proportionate to noise level
        wprime=1.,
        threshold=0.3,
        mode="original",            # "o(riginal)" or "n(egative)"
        attention_mask_negation:torch.Tensor=None,  # (bsz, seqlen) bool tensor
        **kwargs,
    ):
        batch_size, sequence_length, _ = hidden_states.shape
        assert attention_mask is None, "cross-attention mask will be computed later based on provided masks"
        # attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.cross_attention_norm:
            encoder_hidden_states = attn.norm_cross(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_mask = masks[query.size(1)]
        attention_mask = attention_mask.permute(0, 2, 3, 1).flatten(1, 2)

        strength = torch.sigmoid(((1 - sigma) - threshold) / torch.tensor(0.025))

        if mode.startswith("o"):
            # attention_mask = torch.zeros(query.size(0), query.size(1), key.size(1), device=query.device)
            attention_mask = wprime * attention_mask        # scale with noise level ==> higher with higher noise level
            attention_mask = attention_mask * torch.log(1 + sigma * 80)
            # attention_mask = attention_mask * math.log(1+80)
            # attention_mask = attention_mask * 0    # DEBUG
        elif mode.startswith("n"):
            # parts of attention mask that are not negated should be flooded so 0->1 when strength rises
            attention_mask = attention_mask.float().clamp_min(strength)
            if attention_mask_negation is not None:
                # negated parts of mask should diminish 1->0 when strength rises
                negated_attention_mask = attention_mask.float().clamp_max(1-strength)
                attention_mask = torch.where(attention_mask_negation[:, None, :], negated_attention_mask, attention_mask)

        elif mode.startswith("p"):
            attention_mask = attention_mask * (1 - strength) * wprime

        # Change: feed position ids
        attention_probs = attn.get_attention_scores(query, key, attention_mask, mode=mode)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states
    

class SelfAttnProcessor(CrossAttnProcessor):
    def __call__(
        self,
        attn: CrossAttention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        **kwargs,
    ):
        return super(SelfAttnProcessor, self).__call__(attn, hidden_states, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask)


class LayerMode(Enum):
    NORMAL = "normal"


def unpack_layers_old(layers):
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
        opaquematrix = (layermatrix >= 1.).float()
        transparentmatrix = (layermatrix > 0).float()
        layermatrix = 0.5 * opaquematrix + 0.5 * transparentmatrix
        _layermatrix = layermatrix
        assert layermatrix.size() == (512, 512)
        downsamples = [8, 16, 32, 64]
        unpacked_layers.append({
            "pos": pos,
            "neg": neg,
            "mode": None,
            tuple(layermatrix.shape): layermatrix,
        })
        for downsample in downsamples:
            _opaquematrix = torch.nn.functional.avg_pool2d(opaquematrix[None, :, :], downsample, downsample)[0, :, :]
            _opaquematrix = (_opaquematrix > 0.).float()
            _transparentmatrix = torch.nn.functional.avg_pool2d(transparentmatrix[None, :, :], downsample, downsample)[0, :, :]
            _transparentmatrix = (_transparentmatrix > 0.).float()
            _layermatrix = 0.5 * _opaquematrix + 0.5 * _transparentmatrix
            unpacked_layers[-1][tuple(_layermatrix.shape)] = _layermatrix

    return {"layers": unpacked_layers, "global": global_descriptions}


def compute_cross_attn_masks(spec, device, mode="original"):
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

    # unlike in exclusionary cross-attention modification, we are not going to promote base layer or background layer
    # so we have to set the masks to False
    if mode.startswith("o") or mode.startswith("p"):
        for res in layermasks:
            layermasks[res][0:2] = torch.zeros_like(layermasks[res][0:2])

    return layermasks


class StableDiffusionPipelineLayers2ImageEdiffi(StableDiffusionPipeline):
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

    def __init__(
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
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            requires_safety_checker=requires_safety_checker
        )
        self._convert()

    def _tokenize_dict(self, d, remove_ediffi=False):
        if remove_ediffi:
            d["pos"] = re.sub(f"\[Ediffi\:[^\]]+\]", "", d["pos"])
            d["neg"] = re.sub(f"\[Ediffi\:[^\]]+\]", "", d["neg"])
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

    def _tokenize_annotated_prompt(self, prompt):
        # find the [ediffi: ...] part in the pos and neg global prompts
        m = re.match(r"\[Ediffi\:(.+)\]", prompt)
        if m:
            prompt = m.groups()[0].strip()

        prompt = re.split(r"(\{[^\}]+\})", prompt)
        _prompt = []
        _level_id = []
        for e in prompt:
            m = re.match(r"\{(.+):(\d+)\}", e)
            if m:
                _prompt.append(m.group(1))
                _level_id.append(int(m.group(2)) + 1)
            else:
                _prompt.append(e)
                _level_id.append(0)

        for i in range(len(_prompt)):
            _prompt[i] = self.tokenizer(_prompt[i], return_tensors="pt").input_ids[0, (0 if i == 0 else 1):(-1 if i < len(_prompt) - 1 else None)]
            _level_id[i] = torch.tensor([_level_id[i]]).repeat(len(_prompt[i]))

        token_ids = torch.cat(_prompt, 0)
        level_ids = torch.cat(_level_id, 0)

        assert len(token_ids) <= self.tokenizer.model_max_length
        return token_ids, level_ids

    def _encode_prompt_one(self, spec, device, num_images_per_prompt, do_classifier_free_guidance):
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

        pos_prompt = spec["global"]["pos"]
        neg_prompt = spec["global"]["neg"]

        pos_token_ids, pos_level_ids = self._tokenize_annotated_prompt(pos_prompt)

        pos_text_embedding = self.text_encoder(pos_token_ids[None].to(device), attention_mask=None)[0]
        bs_embed, seq_len, _ = pos_text_embedding.shape
        pos_text_embedding = pos_text_embedding[:, None, :, :].repeat(1, num_images_per_prompt, 1, 1)
        pos_text_embedding = pos_text_embedding.view(bs_embed * num_images_per_prompt, seq_len, -1)

        if do_classifier_free_guidance:
            neg_token_ids, neg_level_ids = self._tokenize_annotated_prompt(neg_prompt)

            neg_text_embedding = self.text_encoder(neg_token_ids[None].to(device), attention_mask=None)[0]
            bs_embed, seq_len, _ = neg_text_embedding.shape
            neg_text_embedding = neg_text_embedding[:, None, :, :].repeat(1, num_images_per_prompt, 1, 1)
            neg_text_embedding = neg_text_embedding.view(bs_embed * num_images_per_prompt, seq_len, -1)

            return (pos_text_embedding, pos_level_ids.to(device), pos_token_ids.to(device)), (neg_text_embedding, neg_level_ids.to(device), neg_token_ids.to(device))
        return (pos_text_embedding, pos_level_ids.to(device), pos_token_ids.to(device)), None


    def _encode_prompt_many(self, spec, device, num_images_per_prompt, do_classifier_free_guidance):
        # encode all the prompts for different layers and create cross-attention masks
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
        spec["global"] = self._tokenize_dict(spec["global"], remove_ediffi=True)
        spec["layers"] = [self._tokenize_dict(layer, remove_ediffi=True) for layer in spec["layers"]]
        ds = [spec["global"]] + spec["layers"]

        # FOR POSITIVE PROMPTS:
        # Collect all prompts in one matrix and build gatherer
        pos_text_ids = [x["pos_ids"] for x in ds]
        pos_attn_masks = [x["pos_attn_mask"] for x in ds]
        maxlen = max([x.sum().cpu().item() for x in pos_attn_masks])
        pos_text_ids = [x[:, :maxlen] for x in pos_text_ids]
        pos_attn_masks = [x[:, :maxlen] for x in pos_attn_masks]
        masked_selector = torch.cat(pos_attn_masks, 1).long()
        pos_token_ids = torch.masked_select(torch.cat(pos_text_ids, 1), masked_selector.bool())

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
            neg_token_ids = torch.masked_select(torch.cat(neg_text_ids, 1), masked_selector.bool())

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

            return (pos_text_embeddings, pos_layer_ids.to(device), pos_token_ids.to(device)), (neg_text_embeddings, neg_layer_ids.to(device), neg_token_ids.to(device))
        return (pos_text_embeddings, pos_layer_ids.to(device), pos_token_ids.to(device)), None

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
        wprime:float=1./20,
        threshold:float=0.3,
        mode="original",        # "original" or "n(egative)"
        expose_start_end_token=False,
        expand_negative_prompt=False,
            encode_layers=True,
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

        if not isinstance(generator, torch.Generator) and not generator is None:
            _generator = torch.Generator(device)
            _generator.manual_seed(generator)
            generator = _generator
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        encode_fn = self._encode_prompt_many if encode_layers else self._encode_prompt_one
        output = encode_fn(
            spec, device, num_images_per_prompt, do_classifier_free_guidance
        )

        APPEND_POS_TO_NEG = expand_negative_prompt

        text_embeddings, layer_ids, token_ids = output[0]
        if not encode_layers:
            layer_ids = torch.where(layer_ids == 0, torch.ones_like(layer_ids), layer_ids)       # assign tokens that are not part of any layer to the background layer
        _cross_attention_masks = compute_cross_attn_masks(spec, device, mode=mode)
        cross_attention_masks = {res: torch.index_select(_cross_attention_masks[res], 0, layer_ids) for res in _cross_attention_masks}
        cross_attention_masks = {res: m[None].repeat(num_images_per_prompt, 1, 1, 1) for res, m in cross_attention_masks.items()}
        cross_attention_masks = {res[0]*res[1]: mask for res, mask in cross_attention_masks.items() if res[0] <= 64}

        EXPOSE_START_END_TOKEN = expose_start_end_token
        if EXPOSE_START_END_TOKEN and not mode.startswith("o"):      # make sure the start and end token can be attended to from all regions
            cross_attention_masks = {
                res: torch.where(((token_ids == 49406)| (token_ids == 49407))[None, :, None, None],
                            torch.ones_like(mask), mask)
                for res, mask in cross_attention_masks.items()
            }

        nummasks = _cross_attention_masks[min(_cross_attention_masks.keys())].size(0)

        if do_classifier_free_guidance:
            neg_text_embeddings, neg_layer_ids, neg_token_ids = output[1]
            _neg_cross_attention_masks = _cross_attention_masks
            negated_attention_mask = None
            if APPEND_POS_TO_NEG:
                neg_text_embeddings = torch.cat([neg_text_embeddings, text_embeddings], 1)
                negated_attention_mask = torch.cat([
                    torch.zeros_like(neg_token_ids),
                    torch.ones_like(token_ids)
                ], 0).bool()[None]
                neg_token_ids = torch.cat([neg_token_ids, token_ids], 0)
                extra_layer_ids = layer_ids + nummasks
                neg_layer_ids = torch.cat([neg_layer_ids, extra_layer_ids], 0)
                _neg_cross_attention_masks = {
                    res : torch.cat([_neg_cross_attention_masks[res], ~_cross_attention_masks[res]], 0)
                    for res in _neg_cross_attention_masks
                }
            neg_cross_attention_masks = {res: torch.index_select(_neg_cross_attention_masks[res], 0, neg_layer_ids) for res in _neg_cross_attention_masks}
            neg_cross_attention_masks = {res: m[None].repeat(num_images_per_prompt, 1, 1, 1) for res, m in
                                     neg_cross_attention_masks.items()}
            neg_cross_attention_masks = {res[0]*res[1]: mask for res, mask in neg_cross_attention_masks.items() if res[0] <= 64}
            if EXPOSE_START_END_TOKEN and not mode.startswith(
                    "o"):  # make sure the start and end token can be attended to from all regions
                neg_cross_attention_masks = {
                    res: torch.where(((neg_token_ids == 49406) | (neg_token_ids == 49407))[None, :, None, None],
                                     torch.ones_like(mask), mask)
                    for res, mask in neg_cross_attention_masks.items()
                }

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

        # 6. Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                # latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latents, t)

                # predict the noise residual
                noise_pred = self.unet(latent_model_input, t,
                                       encoder_hidden_states=text_embeddings,
                                       cross_attention_kwargs=
                                        {"masks": cross_attention_masks,
                                         "sigma": t / self.scheduler.num_train_timesteps,
                                         "wprime": wprime,
                                         "threshold": threshold,
                                         "mode": mode,
                                         }).sample

                # perform guidance
                if do_classifier_free_guidance:
                    neg_noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=neg_text_embeddings,
                                               cross_attention_kwargs={"masks": neg_cross_attention_masks,
                                                                       "sigma": t / self.scheduler.num_train_timesteps,
                                                                       "wprime": wprime,
                                                                       "threshold": threshold,
                                                                       "attention_mask_negation": negated_attention_mask,
                                                                       "mode": mode
                                                                       }).sample
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

    def _convert(self):
        # adapt unet to use StructuredCrossAttention
        print("replacing CrossAttention in BasicTransformerBlock with CrossAttentionWithMask")
        _c = 0
        for m in self.unet.modules():
            if isinstance(m, BasicTransformerBlock):
                m.attn2.__class__ = MaskedCrossAttention
                m.attn2.set_processor(MaskedCrossAttnProcessor())
                if m.only_cross_attention:
                    m.attn1.__class__ = MaskedCrossAttention
                    m.attn1.set_processor(MaskedCrossAttnProcessor())
                else:
                    m.attn1.set_processor(SelfAttnProcessor())
                _c += 1
        print