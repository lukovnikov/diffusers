from typing import Optional, Tuple, Union, List, Callable

import torch
from torch import nn
from transformers import CLIPTextConfig
from transformers.models.clip.modeling_clip import CLIPTextTransformer

from diffusers.models.attention import CrossAttention
from .unet_2d_condition import UNet2DConditionOutput
from .. import UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
from ..configuration_utils import register_to_config
from ..pipelines.stable_diffusion import StableDiffusionPipelineOutput
from ..utils.import_utils import is_xformers_available
from ..utils import logging


if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class StructuredCrossAttention(CrossAttention):
    r"""
    A cross attention layer.

    Parameters:
        query_dim (`int`): The number of channels in the query.
        cross_attention_dim (`int`, *optional*):
            The number of channels in the context. If not given, defaults to `query_dim`.
        heads (`int`,  *optional*, defaults to 8): The number of heads to use for multi-head attention.
        dim_head (`int`,  *optional*, defaults to 64): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        bias (`bool`, *optional*, defaults to False):
            Set to `True` for the query, key, and value linear layers to contain a bias parameter.
    """

    def forward(self, hidden_states, context=None, mask=None):
        context, structure = context if isinstance(context, tuple) else (context, None)
        batch_size, sequence_length, _ = hidden_states.shape

        query = self.to_q(hidden_states)
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
            hidden_states = self._memory_efficient_attention_xformers(query, key, value)
            # Some versions of xformers return output in fp32, cast it back to the dtype of the input
            hidden_states = hidden_states.to(query.dtype)
        else:
            if self._slice_size is None or query.shape[0] // self._slice_size == 1:
                hidden_states = self._attention(query, key, value, structure=structure)
            else:
                hidden_states = self._sliced_attention(query, key, value, sequence_length, dim)

        # linear proj
        hidden_states = self.to_out[0](hidden_states)
        # dropout
        hidden_states = self.to_out[1](hidden_states)
        return hidden_states

    def _attention(self, query, key, value, structure=None):
        if structure is None:
            return super()._attention(query, key, value)
        else:
            attention_scores = torch.baddbmm(
                torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device),
                query,
                key.transpose(-1, -2),
                beta=0,
                alpha=self.scale,
            )

            if False:
                attention_probs = attention_scores.softmax(dim=-1)
            else:
                # take into account structure
                # structure spec format: [[nummem, memsize, 0, ..., 1, 0, 2, 1, 0, ...]*batsize]
                # extract memory from key and value
                nummem, memsize = structure[0, 0].item(), structure[0, 1].item()
                _structure = structure[:, nummem*memsize:]

                # extract memory scores
                memscores = attention_scores[:, :, :nummem*memsize]             # (B*H, Qlen, Mlen)
                _attention_scores = attention_scores[:, :, nummem*memsize:]     # (B*H, Qlen, Klen)

                # separately normalize memory scores within one memory supercell
                bxh, qlen, nummemxmemsize = memscores.shape
                memscores = memscores.view(bxh, qlen, nummem, memsize)       # (B*H, Qlen, nummem, memsize)
                memprobs = memscores.softmax(dim=-1)

                # normalize attention probs for the actual input
                _attention_probs = _attention_scores.softmax(dim=-1)

                # gather attention probability per supercell according to structure
                supercellprobs = torch.zeros(memprobs.size(0), memprobs.size(1), nummem+1, dtype=memprobs.dtype, device=memprobs.device)
                indexes = _structure[:, None, None, :].repeat(1, self.heads, qlen, 1)
                indexes = indexes.view(-1, qlen, indexes.size(-1))
                supercellprobs.scatter_add_(-1, indexes, _attention_probs)
                supercellprobs = supercellprobs[:, :, 1:]

                # multiply the attention prob of memory token occurrences with the probability of 0-th memory cell prob for the corresponding memory token
                # this is done to be able to use the syntactic info of the memory token as it occurs in the sentence
                memprobs_first = memprobs[:, :, :, 0]
                memprobs_first = torch.cat([torch.ones_like(memprobs_first[:, :, 0:1]), memprobs_first], -1)
                # (BxH, Qlen, 1+nummem)
                _attention_probs = _attention_probs * torch.gather(memprobs_first, 2, indexes)

                # scale memory cell probs with supercellprobs
                memprobs = memprobs * supercellprobs[:, :, :, None]
                # set to zero first scores within a supercell (we just distributed this probability to the memory tokens)
                memprobs[:, :, :, 0] *= 0

                # merge all probs back together
                attention_probs = torch.cat([memprobs.view(bxh, qlen, nummemxmemsize), _attention_probs], -1)
                attention_probs_sums = attention_probs.sum(-1)
                assert torch.allclose(torch.ones_like(attention_probs_sums), attention_probs_sums, rtol=1e-3)

            hidden_states = torch.bmm(attention_probs, value)

            # reshape hidden_states
            hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
            return hidden_states


class StructuredCLIPTextTransformer(CLIPTextTransformer):

    def init_mem(self,
        tokenid_to_mem_map = None,
        mem_to_tokenid_map = None):
        self.register_buffer("tokenid_to_mem_map", tokenid_to_mem_map)
        self.register_buffer("mem_to_tokenid_map", mem_to_tokenid_map)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        ret = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        if return_dict:
            text_embeds = ret.last_hidden_state
        else:
            text_embeds, *_ret = ret

        # prepend the memory token embeddings raw in text embeds
        # compute mem ids: concatenate all memory token ids after each other and repeat across all examples
        mem_to_tokenid_map = self.mem_to_tokenid_map
        numsupercell, supercellsize = mem_to_tokenid_map.shape
        batsize = input_ids.size(0)
        mem_ids = mem_to_tokenid_map.view(-1)[None, :].repeat(batsize, 1)
        # (batsize, nummemids * nummem,)

        # embed mem ids without positioning and concatenate
        mem_embeds = self.embeddings.token_embedding(mem_ids)       # (batsize, nummemids * nummem, embdim)
        embeds = torch.cat([mem_embeds, text_embeds], 1)

        # compute structure spec
        structure = self.tokenid_to_mem_map[input_ids]
        structure = torch.cat([torch.zeros_like(structure[:, 0:1]).repeat(1, numsupercell*supercellsize), structure], 1)
        structure[:, 0] = numsupercell
        structure[:, 1] = supercellsize
        if return_dict:
            ret.last_hidden_state = (embeds, structure)
        else:
            ret = ((embeds, structure)) + _ret
        return ret


class StructuredStableDiffusionPipeline(StableDiffusionPipeline):
    ###### THE CHANGE: output text embeddings AND text structure in encode prompt --> it gets passed on into Unet straight away in superclass

    def _encode_prompt(self, prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt):
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
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        maxlen = (text_input_ids != self.tokenizer.pad_token_id).long().sum(1).max().cpu().item()
        text_input_ids = text_input_ids[:, :maxlen+1]
        untruncated_ids = self.tokenizer(prompt, padding="max_length", return_tensors="pt").input_ids

        if text_input_ids.size(1) > untruncated_ids.size(1):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )

        if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None

        text_embeddings = self.text_encoder(
            text_input_ids.to(device),
            attention_mask=attention_mask,
        )
        text_embeddings, text_structure, *_ = text_embeddings

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_images_per_prompt, 1)
        text_embeddings = text_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            neg_input_ids = uncond_input.input_ids

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            uncond_embeddings, uncond_structure, *_ = uncond_embeddings

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(1, num_images_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(batch_size * num_images_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
            text_structure = torch.cat([uncond_structure, text_structure])

        return text_embeddings, text_structure
