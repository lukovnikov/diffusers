from typing import Optional, Tuple, Union

import torch
from torch import nn

from diffusers.models.attention import CrossAttention
from .unet_2d_condition import UNet2DConditionOutput
from .. import UNet2DConditionModel
from ..configuration_utils import register_to_config
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


class StructuredUNet2DConditionModel(UNet2DConditionModel):

    def init_mem(self,
        nummem: int = 2,
        memsize: int = 32,
        memdim: int = 768,
        mem = None,
        tokenid_to_mem_map = None):
        if mem is None:
            mem = torch.nn.Linear(nummem * memsize, memdim)
            mem = mem.weight.T.view(nummem, memsize, memdim)
        self.mem = torch.nn.Parameter(mem)
        self.register_buffer("tokenid_to_mem_map", tokenid_to_mem_map)

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: Tuple[torch.Tensor, torch.Tensor],
        class_labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[UNet2DConditionOutput, Tuple]:
        r"""
        Args:
            sample (`torch.FloatTensor`): (batch, channel, height, width) noisy inputs tensor
            timestep (`torch.FloatTensor` or `float` or `int`): (batch) timesteps
            encoder_hidden_states (`torch.FloatTensor`): (batch, channel, height, width) encoder hidden states
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain tuple.

        Returns:
            [`~models.unet_2d_condition.UNet2DConditionOutput`] or `tuple`:
            [`~models.unet_2d_condition.UNet2DConditionOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.
        """


        nummem, memsize, memdim = self.mem.shape
        mem = self.mem.view(-1, self.mem.size(-1))[None, :, :].repeat(sample.size(0), 1, 1)
        encoder_hidden_states, inputids = encoder_hidden_states
        structure = self.tokenid_to_mem_map[inputids]
        # print(mem.size(), encoder_hidden_states.size())
        encoder_hidden_states = torch.cat([mem, encoder_hidden_states], 1)
        structure = torch.cat([torch.zeros_like(structure[:, 0:1]).repeat(1, nummem*memsize), structure], 1)
        structure[:, 0] = nummem
        structure[:, 1] = memsize

        # encoder_hidden_states = (encoder_hidden_states, structure)

        return super().forward(sample, timestep, (encoder_hidden_states, structure), class_labels, return_dict)
        ###############################

        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layears).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2**self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            logger.info("Forward upsample size to force interpolation output size.")
            forward_upsample_size = True

        # 0. center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.dtype)
        emb = self.time_embedding(t_emb)

        if self.config.num_class_embeds is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")
            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
            emb = emb + class_emb

        # 2. pre-process
        sample = self.conv_in(sample)

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "attentions") and downsample_block.attentions is not None:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        # 4. mid
        sample = self.mid_block(sample, emb, encoder_hidden_states=encoder_hidden_states)

        # 5. up
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "attentions") and upsample_block.attentions is not None:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    upsample_size=upsample_size,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples, upsample_size=upsample_size
                )
        # 6. post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if not return_dict:
            return (sample,)

        return UNet2DConditionOutput(sample=sample)

