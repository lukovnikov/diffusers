import re

import math
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, LMSDiscreteScheduler, UNet2DConditionModel
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, CLIPFeatureExtractor
import torchvision.transforms as T

from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipelineOutput

import psd_tools


def always_round(x):
    intx = int(x)
    is_even = intx % 2 == 0
    if is_even:
        if x < intx + 0.5:
            return intx
        return intx + 1
    else:
        return round(x)


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
        splits = layer.name.split("*")
        layername = splits[0]
        strength = 1.
        if len(splits) == 2:
            strength = float(splits[1])
        splits = [x.strip() for x in layername.split("|")]
        pos, neg = splits if len(splits) == 2 else (splits[0], "")
        layermatrix = torch.tensor(np.asarray(layer.topil().getchannel("A"))).float() / 255
        layermatrix = (layermatrix > 0.5).float()

        assert layermatrix.size() == (512, 512)
        unpacked_layers.append({
            "pos": pos,
            "neg": neg,
            "strength": strength,
            tuple(layermatrix.shape): layermatrix,
        })

    fullreskey = tuple(layermatrix.shape)

    # subtract masks from each other before downsampling to reproduce ediffi conditions
    subacc = torch.zeros_like(layermatrix).bool()
    for layer in unpacked_layers[::-1]:
        newlayermask = layer[fullreskey] > 0.5
        newlayermask = newlayermask & (~subacc)
        subacc = subacc | newlayermask
        layer[fullreskey] = newlayermask.to(layer[fullreskey].dtype)

    # compute downsampled versions of the layer masks
    downsamples = [8, 16, 32, 64]
    for layer in unpacked_layers[::-1]:
        layermatrix = layer[fullreskey]
        for downsample in downsamples:
            downsampled = _img_importance_flatten(layermatrix, downsample, downsample)
            layer[tuple(downsampled.shape)] = downsampled

    return {"layers": unpacked_layers, "global": global_descriptions}


def preprocess(image):
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0


def _img_importance_flatten(img: torch.tensor, w: int, h: int) -> torch.tensor:
    return F.interpolate(
        img.unsqueeze(0).unsqueeze(1),
        # scale_factor=1 / ratio,
        size=(w, h),
        mode="bilinear",
        align_corners=True,
    ).squeeze()


def _pil_from_latents(vae, latents):
    _latents = 1 / 0.18215 * latents.clone()
    image = vae.decode(_latents).sample

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    ret_pil_images = [Image.fromarray(image) for image in images]

    return ret_pil_images


@torch.autocast("cuda")
def inj_forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, **cross_attn_kwargs):
    is_dict_format = True
    context = encoder_hidden_states
    if context is not None:
        try:
            context_tensor = context["CONTEXT_TENSOR"]
        except:
            context_tensor = context
            is_dict_format = False

    else:
        context_tensor = hidden_states

    batch_size, sequence_length, _ = hidden_states.shape

    query = self.to_q(hidden_states)

    key = self.to_k(context_tensor)
    value = self.to_v(context_tensor)

    dim = query.shape[-1]

    query = self.head_to_batch_dim(query)
    key = self.head_to_batch_dim(key)
    value = self.head_to_batch_dim(value)

    attention_scores = torch.matmul(query, key.transpose(-1, -2))

    attention_size_of_img = attention_scores.shape[-2]
    if context is not None:
        if is_dict_format:
            f: Callable = context["WEIGHT_FUNCTION"]
            try:
                w = context[f"CROSS_ATTENTION_WEIGHT_{attention_size_of_img}"]
            except KeyError:
                w = context[f"CROSS_ATTENTION_WEIGHT_ORIG"]
                if not isinstance(w, int):
                    img_h, img_w, nc = w.shape
                    ratio = math.sqrt(img_h * img_w / attention_size_of_img)
                    w = F.interpolate(w.permute(2, 0, 1).unsqueeze(0), scale_factor=1 / ratio, mode="bilinear",
                                      align_corners=True)
                    w = F.interpolate(w.reshape(1, nc, -1), size=(attention_size_of_img,), mode='nearest').permute(2, 1,
                                                                                                                   0).squeeze()
                else:
                    w = 0
            sigma = context["SIGMA"]

            cross_attention_weight = f(w, sigma, attention_scores)
        else:
            cross_attention_weight = 0.0
    else:
        cross_attention_weight = 0.0

    attention_scores = (attention_scores + cross_attention_weight) * self.scale

    attention_probs = attention_scores.softmax(dim=-1)

    hidden_states = torch.matmul(attention_probs, value)

    hidden_states = self.batch_to_head_dim(hidden_states)

    # linear proj
    hidden_states = self.to_out[0](hidden_states)
    # dropout
    hidden_states = self.to_out[1](hidden_states)

    return hidden_states


def pww_load_tools(
        device: str = "cuda:0",
        scheduler_type=LMSDiscreteScheduler,
        local_model_path: Optional[str] = None,
        hf_model_path: Optional[str] = None,
        model_token: Optional[str] = None,
) -> Tuple[
    UNet2DConditionModel,
    CLIPTextModel,
    CLIPTokenizer,
    AutoencoderKL,
    LMSDiscreteScheduler,
]:
    assert (
            local_model_path or hf_model_path
    ), "either local_model_path or hf_model_path must be provided"

    is_mps = device == 'mps'
    dtype = torch.float16 if not is_mps else torch.float32

    model_path = local_model_path if local_model_path is not None else hf_model_path
    local_path_only = local_model_path is not None
    print(model_path)
    if not is_mps:
        vae = AutoencoderKL.from_pretrained(
            model_path,
            subfolder="vae",
            use_auth_token=model_token,
            torch_dtype=dtype,
            local_files_only=local_path_only,
            revision="fp16"
        )
    else:
        vae = AutoencoderKL.from_pretrained(
            model_path,
            subfolder="vae",
            use_auth_token=model_token,
            torch_dtype=dtype,
            local_files_only=local_path_only,
        )

    tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder")

    if not is_mps:
        unet = UNet2DConditionModel.from_pretrained(
            model_path,
            subfolder="unet",
            use_auth_token=model_token,
            torch_dtype=dtype,
            local_files_only=local_path_only,
            revision="fp16",
        )
    else:
        unet = UNet2DConditionModel.from_pretrained(
            model_path,
            subfolder="unet",
            use_auth_token=model_token,
            torch_dtype=dtype,
            local_files_only=local_path_only,
        )

    vae.to(device), unet.to(device), text_encoder.to(device)

    for _module in unet.modules():
        if _module.__class__.__name__ == "CrossAttention":
            _module.__class__.__call__ = inj_forward

    scheduler = scheduler_type(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
    )

    return vae, unet, text_encoder, tokenizer, scheduler


def _image_context_seperator(
        img: Image.Image, color_context: dict, _tokenizer
) -> List[Tuple[List[int], torch.Tensor]]:
    ret_lists = []

    if img is not None:
        w, h = img.size
        # w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
        # img = img.resize((w, h), resample=PIL.Image.LANCZOS)

        for color, v in color_context.items():
            f = v.split(",")[-1]
            v = ",".join(v.split(",")[:-1])
            f = float(f)
            v_input = _tokenizer(
                v,
                max_length=_tokenizer.model_max_length,
                truncation=True,
            )
            v_as_tokens = v_input["input_ids"][1:-1]
            if isinstance(color, str):
                r, g, b = color[1:3], color[3:5], color[5:7]
                color = (int(r, 16), int(g, 16), int(b, 16))
            img_where_color = (np.array(img) == color).all(axis=-1)

            if not img_where_color.sum() > 0:
                print(f"Warning : not a single color {color} not found in image")

            img_where_color = torch.tensor(img_where_color, dtype=torch.float32) * f

            ret_lists.append((v_as_tokens, img_where_color))
    else:
        w, h = 512, 512

    if len(ret_lists) == 0:
        ret_lists.append(([-1], torch.zeros((w, h), dtype=torch.float32)))
    return ret_lists, w, h


def _tokens_img_attention_weight(
        img_context_seperated, tokenized_texts, ratio: int = 8, original_shape=False
):
    token_lis = tokenized_texts["input_ids"][0].tolist()
    w, h = img_context_seperated[0][1].shape

    w_r, h_r = always_round(w / ratio), always_round(h / ratio)
    ret_tensor = torch.zeros((w_r * h_r, len(token_lis)), dtype=torch.float32)

    for v_as_tokens, img_where_color in img_context_seperated:
        is_in = 0
        for idx, tok in enumerate(token_lis):
            if token_lis[idx: idx + len(v_as_tokens)] == v_as_tokens:
                is_in = 1

                # print(token_lis[idx : idx + len(v_as_tokens)], v_as_tokens)
                ret_tensor[:, idx: idx + len(v_as_tokens)] += (
                    _img_importance_flatten(img_where_color, w_r, h_r)
                    .reshape(-1, 1)
                    .repeat(1, len(v_as_tokens))
                )

        if not is_in == 1:
            print(f"Warning ratio {ratio} : tokens {v_as_tokens} not found in text")

    if original_shape:
        ret_tensor = ret_tensor.reshape((w_r, h_r, len(token_lis)))

    return ret_tensor


def _extract_seed_and_sigma_from_context(color_context, ignore_seed=-1):
    # Split seed and sigma from color_context if provided
    extra_seeds = {}
    extra_sigmas = {}
    for i, (k, _context) in enumerate(color_context.items()):
        _context_split = _context.split(',')
        if len(_context_split) > 2:
            try:
                seed = int(_context_split[-2])
                sigma = float(_context_split[-1])
                _context_split = _context_split[:-2]
                extra_sigmas[i] = sigma
            except ValueError:
                seed = int(_context_split[-1])
                _context_split = _context_split[:-1]
            if seed != ignore_seed:
                extra_seeds[i] = seed
        color_context[k] = ','.join(_context_split)
    return color_context, extra_seeds, extra_sigmas


def _get_binary_mask(seperated_word_contexts, extra_seeds, dtype, size):
    img_where_color_mask = [(seperated_word_contexts[k][1] > 0).type(dtype) for k in extra_seeds.keys()]
    img_where_color_mask = [F.interpolate(mask.unsqueeze(0).unsqueeze(1),
                                          size=size, mode='bilinear') for mask in img_where_color_mask]
    return img_where_color_mask


def _blur_image_mask(seperated_word_contexts, extra_sigmas):
    for k, sigma in extra_sigmas.items():
        blurrer = T.GaussianBlur(kernel_size=(39, 39), sigma=(sigma, sigma))
        v_as_tokens, img_where_color = seperated_word_contexts[k]
        seperated_word_contexts[k] = (v_as_tokens, blurrer(img_where_color[None, None])[0, 0])
    return seperated_word_contexts


def _encode_text_color_inputs(
        text_encoder, tokenizer, device,
        color_map_image, color_context,
        input_prompt, unconditional_input_prompt):
    # Process input prompt text
    text_input = tokenizer(
        [input_prompt],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )

    # Extract seed and sigma from color context
    color_context, extra_seeds, extra_sigmas = _extract_seed_and_sigma_from_context(color_context)
    is_extra_sigma = len(extra_sigmas) > 0

    # Process color map image and context
    separated_word_contexts, width, height = _image_context_seperator(
        color_map_image, color_context, tokenizer
    )

    # separated_word_contexts is a list of tuples, each containing the sequence of token ids for the selected words and corresponding mask
    # Smooth mask with extra sigma if applicable
    if is_extra_sigma:
        print('Use extra sigma to smooth mask', extra_sigmas)
        separated_word_contexts = _blur_image_mask(separated_word_contexts, extra_sigmas)

    # Compute cross-attention weights
    cross_attention_weight_1 = _tokens_img_attention_weight(
        separated_word_contexts, text_input, ratio=1, original_shape=True
    ).to(device)
    cross_attention_weight_8 = _tokens_img_attention_weight(
        separated_word_contexts, text_input, ratio=8
    ).to(device)
    cross_attention_weight_16 = _tokens_img_attention_weight(
        separated_word_contexts, text_input, ratio=16
    ).to(device)
    cross_attention_weight_32 = _tokens_img_attention_weight(
        separated_word_contexts, text_input, ratio=32
    ).to(device)
    cross_attention_weight_64 = _tokens_img_attention_weight(
        separated_word_contexts, text_input, ratio=64
    ).to(device)

    # Compute conditional and unconditional embeddings
    cond_embeddings = text_encoder(text_input.input_ids.to(device))[0]
    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer(
        [unconditional_input_prompt],
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]

    encoder_hidden_states = {
        "CONTEXT_TENSOR": cond_embeddings,
        f"CROSS_ATTENTION_WEIGHT_ORIG": cross_attention_weight_1,
        f"CROSS_ATTENTION_WEIGHT_{always_round(height / 8) * always_round(width / 8)}": cross_attention_weight_8,
        f"CROSS_ATTENTION_WEIGHT_{always_round(height / 16) * always_round(width / 16)}": cross_attention_weight_16,
        f"CROSS_ATTENTION_WEIGHT_{always_round(height / 32) * always_round(width / 32)}": cross_attention_weight_32,
        f"CROSS_ATTENTION_WEIGHT_{always_round(height / 64) * always_round(width / 64)}": cross_attention_weight_64,
    }

    uncond_encoder_hidden_states = {
        "CONTEXT_TENSOR": uncond_embeddings,
        f"CROSS_ATTENTION_WEIGHT_ORIG": 0,
        f"CROSS_ATTENTION_WEIGHT_{always_round(height / 8) * always_round(width / 8)}": 0,
        f"CROSS_ATTENTION_WEIGHT_{always_round(height / 16) * always_round(width / 16)}": 0,
        f"CROSS_ATTENTION_WEIGHT_{always_round(height / 32) * always_round(width / 32)}": 0,
        f"CROSS_ATTENTION_WEIGHT_{always_round(height / 64) * always_round(width / 64)}": 0,
    }

    return extra_seeds, separated_word_contexts, encoder_hidden_states, uncond_encoder_hidden_states


def _tokenize_annotated_prompt(prompt, tokenizer):
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
        if i == len(_prompt) - 1:
            tokenized = tokenizer([_prompt[i]],
                                  padding="max_length",
                                  max_length=tokenizer.model_max_length,
                                  truncation=True,
                                  return_tensors="pt")
        else:
            tokenized = tokenizer([_prompt[i]], return_tensors="pt")
        _prompt[i] = tokenized.input_ids[0, (0 if i == 0 else 1):(-1 if i < len(_prompt) - 1 else None)]
        _level_id[i] = torch.tensor([_level_id[i]]).repeat(len(_prompt[i]))

    token_ids = torch.cat(_prompt, 0)
    token_ids = token_ids[:min(len(token_ids), tokenizer.model_max_length)]
    level_ids = torch.cat(_level_id, 0)
    level_ids = level_ids[:min(len(level_ids), tokenizer.model_max_length)]

    assert len(token_ids) <= tokenizer.model_max_length
    return token_ids, level_ids


def _encode_prompt(spec, device, num_images_per_prompt, do_classifier_free_guidance, text_encoder=None, tokenizer=None):
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

    pos_token_ids, pos_level_ids = _tokenize_annotated_prompt(pos_prompt, tokenizer)

    pos_text_embedding = text_encoder(pos_token_ids[None].to(device), attention_mask=None)[0]
    bs_embed, seq_len, _ = pos_text_embedding.shape
    pos_text_embedding = pos_text_embedding[:, None, :, :].repeat(1, num_images_per_prompt, 1, 1)
    pos_text_embedding = pos_text_embedding.view(bs_embed * num_images_per_prompt, seq_len, -1)

    if do_classifier_free_guidance:
        neg_token_ids, neg_level_ids = _tokenize_annotated_prompt(neg_prompt, tokenizer)

        neg_text_embedding = text_encoder(neg_token_ids[None].to(device), attention_mask=None)[0]
        bs_embed, seq_len, _ = neg_text_embedding.shape
        neg_text_embedding = neg_text_embedding[:, None, :, :].repeat(1, num_images_per_prompt, 1, 1)
        neg_text_embedding = neg_text_embedding.view(bs_embed * num_images_per_prompt, seq_len, -1)

        return (pos_text_embedding, pos_level_ids.to(device), pos_token_ids.to(device)), (neg_text_embedding, neg_level_ids.to(device), neg_token_ids.to(device))
    return (pos_text_embedding, pos_level_ids.to(device), pos_token_ids.to(device)), None


def compute_cross_attn_masks(spec, device, base_strength=0.):
    layermasks = {}
    for layer in spec["layers"]:
        for k in layer:
            if isinstance(k, tuple):
                if k not in layermasks:
                    layermasks[k] = []
                layermasks[k].append(layer[k] * layer["strength"])

    for k in layermasks:
        # Add global mask; this mask does not interact with others
        layermasks[k] = [torch.ones_like(layermasks[k][0]) * base_strength] + layermasks[k]
        layermasks[k] = torch.stack(layermasks[k], 0).to(device)

    return layermasks


@torch.no_grad()
@torch.autocast("cuda")
def paint_with_words(
        specfile:str=None,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        seed: int = 0,
        scheduler_type=LMSDiscreteScheduler,
        device: str = "cuda:0",
        weight_function: Callable = lambda w, sigma, qk: 0.1
                                                         * w
                                                         * math.log(sigma + 1)
                                                         * qk.max(),
        local_model_path: Optional[str] = None,
        hf_model_path: Optional[str] = "CompVis/stable-diffusion-v1-4",
        preloaded_utils: Optional[Tuple] = None,
        init_image=None,
        strength: float = 0.5,
):
    vae, unet, text_encoder, tokenizer, scheduler = (
        pww_load_tools(
            device,
            scheduler_type,
            local_model_path=local_model_path,
            hf_model_path=hf_model_path,
            model_token=None,
        )
        if preloaded_utils is None
        else preloaded_utils
    )

    layers = psd_tools.PSDImage.open(specfile)
    width, height = layers.size
    init_image = layers.composite() if init_image is True else init_image

    spec = unpack_layers(layers)        # TODO: add support for extra seeds and sigmas
    extra_seeds = {}

    # encode texts
    do_cfg = True
    output = _encode_prompt(spec, device, num_images_per_prompt=1, do_classifier_free_guidance=do_cfg, text_encoder=text_encoder, tokenizer=tokenizer)

    text_embeddings, layer_ids, token_ids = output[0]
    _cross_attention_masks = compute_cross_attn_masks(spec, device)
    cross_attention_masks = {res: torch.index_select(_cross_attention_masks[res], 0, layer_ids) for res in
                             _cross_attention_masks}
    cross_attention_masks = {res[0] * res[1]: mask.view(mask.size(0), -1).transpose(0, 1) for res, mask in cross_attention_masks.items() if res[0] <= 64}

    if do_cfg:
        neg_text_embeddings, neg_layer_ids, neg_token_ids = output[1]
        neg_cross_attention_masks = {res: torch.index_select(_cross_attention_masks[res], 0, neg_layer_ids) for res
                                     in _cross_attention_masks}
        neg_cross_attention_masks = {res[0] * res[1]: mask.view(mask.size(0), -1).transpose(0, 1) for res, mask in neg_cross_attention_masks.items() if
                                     res[0] <= 64}

    # extra_seeds: ??
    # separated_word_contexts: ??

    encoder_hidden_states = {
        "CONTEXT_TENSOR": text_embeddings,
        f"CROSS_ATTENTION_WEIGHT_ORIG": None,
    }
    uncond_encoder_hidden_states = {
        "CONTEXT_TENSOR": neg_text_embeddings,
        f"CROSS_ATTENTION_WEIGHT_ORIG": 0,
    }
    for downscle in (8, 16, 32, 64):
        k = always_round(width/downscle) * always_round(height/downscle)
        encoder_hidden_states[f"CROSS_ATTENTION_WEIGHT_{k}"] = cross_attention_masks[k]
        uncond_encoder_hidden_states[f"CROSS_ATTENTION_WEIGHT_{k}"] = 0

    # TODO: verify that we get the same masks and everything compared to original implementation

    # old code below:
    scheduler.set_timesteps(num_inference_steps)
    if init_image is None:
        timesteps = scheduler.timesteps
    else:
        offset = scheduler.config.get("steps_offset", 0)
        init_timestep = int(num_inference_steps * strength) + offset
        init_timestep = min(init_timestep, num_inference_steps)
        t_start = max(num_inference_steps - init_timestep + offset, 0)
        timesteps = scheduler.timesteps[t_start:]
        num_inference_steps = num_inference_steps - t_start
        latent_timestep = timesteps[:1]

    # Latent:
    if init_image is None:  # txt2img
        latent_size = (1, unet.in_channels, height // 8, width // 8)
        latents = torch.randn(latent_size, generator=torch.manual_seed(seed))
        if len(extra_seeds) > 0:
            print('Use region based seeding: ', extra_seeds)
            multi_latents = [torch.randn(latent_size,
                                         generator=torch.manual_seed(_seed)) for _seed in extra_seeds.values()]
            img_where_color_mask = _get_binary_mask(seperated_word_contexts, extra_seeds, dtype=latents[0].dtype,
                                                    size=latent_size[-2:])
            foreground = (sum(img_where_color_mask) > 0).squeeze()
            # sum seeds weighted by masks
            summed_multi_latents = sum(_latents * _mask for _latents, _mask in zip(multi_latents, img_where_color_mask))
            latents[:, :, foreground] = summed_multi_latents[:, :, foreground]
        latents = latents.to(device)
        latents = latents * scheduler.init_noise_sigma
    else:
        init_image = preprocess(init_image)
        image = init_image.to(device=device)
        init_latent_dist = vae.encode(image).latent_dist
        init_latents = init_latent_dist.sample()
        init_latents = 0.18215 * init_latents
        noise = torch.randn(init_latents.shape).to(device)

        # get latents
        init_latents = scheduler.add_noise(init_latents, noise, latent_timestep)
        latents = init_latents

    is_mps = device == "mps"
    for t in tqdm(timesteps):
        # sigma for pww
        step_index = (scheduler.timesteps == t).nonzero().item()
        sigma = scheduler.sigmas[step_index]

        latent_model_input = scheduler.scale_model_input(latents, t)

        _t = t if not is_mps else t.float()
        encoder_hidden_states.update({
            "SIGMA": sigma,
            "WEIGHT_FUNCTION": weight_function,
        })
        noise_pred_text = unet(
            latent_model_input,
            _t,
            encoder_hidden_states=encoder_hidden_states,
        ).sample

        latent_model_input = scheduler.scale_model_input(latents, t)

        uncond_encoder_hidden_states.update({
            "SIGMA": sigma,
            "WEIGHT_FUNCTION": lambda w, sigma, qk: 0.0,
        })
        noise_pred_uncond = unet(
            latent_model_input,
            _t,
            encoder_hidden_states=uncond_encoder_hidden_states,
        ).sample

        noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
        )

        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    ret_pil_images = _pil_from_latents(vae, latents)

    return ret_pil_images[0]


class PaintWithWord_StableDiffusionPipeline(StableDiffusionPipeline):
    def __init__(self,
                 vae: AutoencoderKL,
                 text_encoder: CLIPTextModel,
                 tokenizer: CLIPTokenizer,
                 unet: UNet2DConditionModel,
                 scheduler: LMSDiscreteScheduler,
                 safety_checker: None,
                 feature_extractor: CLIPFeatureExtractor,
                 requires_safety_checker: bool = False,
                 ):
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            requires_safety_checker=requires_safety_checker)
        self.scheduler = LMSDiscreteScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
        )
        self.plugin_cross_attention()

    @classmethod
    def from_pretrained(self, save_dir, **kwargs):
        sd = StableDiffusionPipeline.from_pretrained(save_dir, **kwargs)
        self = PaintWithWord_StableDiffusionPipeline(
            vae=sd.vae,
            text_encoder=sd.text_encoder,
            tokenizer=sd.tokenizer,
            unet=sd.unet,
            scheduler=sd.scheduler,
            safety_checker=sd.safety_checker,
            feature_extractor=sd.feature_extractor,
            requires_safety_checker=sd.requires_safety_checker
        )
        return self

    def plugin_cross_attention(self):
        for _module in self.unet.modules():
            if _module.__class__.__name__ == "CrossAttention":
                _module.__class__.__call__ = inj_forward

    def _encode_text_color_inputs(
            self, device, color_map_image, color_context,
            input_prompt, unconditional_input_prompt):
        # Process input prompt text
        text_input = self.tokenizer(
            [input_prompt],
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        # Extract seed and sigma from color context
        color_context, extra_seeds, _ = _extract_seed_and_sigma_from_context(color_context)

        # Process color map image and context
        separated_word_contexts, width, height = _image_context_seperator(
            color_map_image, color_context, self.tokenizer
        )

        # Compute cross-attention weights
        cross_attention_weight_1 = _tokens_img_attention_weight(
            separated_word_contexts, text_input, ratio=1, original_shape=True
        ).to(device)
        cross_attention_weight_8 = _tokens_img_attention_weight(
            separated_word_contexts, text_input, ratio=8
        ).to(device)
        cross_attention_weight_16 = _tokens_img_attention_weight(
            separated_word_contexts, text_input, ratio=16
        ).to(device)
        cross_attention_weight_32 = _tokens_img_attention_weight(
            separated_word_contexts, text_input, ratio=32
        ).to(device)
        cross_attention_weight_64 = _tokens_img_attention_weight(
            separated_word_contexts, text_input, ratio=64
        ).to(device)

        # Compute conditional and unconditional embeddings
        cond_embeddings = self.text_encoder(text_input.input_ids.to(device))[0]
        max_length = text_input.input_ids.shape[-1]
        uncond_input = self.tokenizer(
            [unconditional_input_prompt],
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(device))[0]

        encoder_hidden_states = {
            "CONTEXT_TENSOR": cond_embeddings,
            f"CROSS_ATTENTION_WEIGHT_ORIG": cross_attention_weight_1,
            f"CROSS_ATTENTION_WEIGHT_{always_round(height / 8) * always_round(width / 8)}": cross_attention_weight_8,
            f"CROSS_ATTENTION_WEIGHT_{always_round(height / 16) * always_round(width / 16)}": cross_attention_weight_16,
            f"CROSS_ATTENTION_WEIGHT_{always_round(height / 32) * always_round(width / 32)}": cross_attention_weight_32,
            f"CROSS_ATTENTION_WEIGHT_{always_round(height / 64) * always_round(width / 64)}": cross_attention_weight_64,
        }

        uncond_encoder_hidden_states = {
            "CONTEXT_TENSOR": uncond_embeddings,
            f"CROSS_ATTENTION_WEIGHT_ORIG": 0,
            f"CROSS_ATTENTION_WEIGHT_{always_round(height / 8) * always_round(width / 8)}": 0,
            f"CROSS_ATTENTION_WEIGHT_{always_round(height / 16) * always_round(width / 16)}": 0,
            f"CROSS_ATTENTION_WEIGHT_{always_round(height / 32) * always_round(width / 32)}": 0,
            f"CROSS_ATTENTION_WEIGHT_{always_round(height / 64) * always_round(width / 64)}": 0,
        }

        return extra_seeds, cond_embeddings.dtype, separated_word_contexts, encoder_hidden_states, uncond_encoder_hidden_states

    @torch.no_grad()
    @torch.autocast("cuda")
    def __call__(
            self,
            prompt: Union[str, List[str]],
            color_map_image: Optional[Image.Image] = None,
            color_context: Dict[Tuple[int, int, int], str] = {},
            weight_function: Callable = lambda w, sigma, qk: 0.1 * w * math.log(sigma + 1) * qk.max(),
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 30,
            guidance_scale: float = 7.5,
            negative_prompt: Optional[Union[str, List[str]]] = "",
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.5,
            seed: Optional[int] = 0,
            generator: Optional[torch.Generator] = None,
            image: Optional[Image.Image] = None,
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

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(prompt, height, width, callback_steps)

        # 2. Define call parameters
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        # text_embeddings = self._encode_prompt(
        #     prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
        # )
        extra_seeds, text_embeddings_type, seperated_word_contexts, encoder_hidden_states, uncond_encoder_hidden_states = \
            self._encode_text_color_inputs(device, color_map_image, color_context, prompt, negative_prompt)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        if image is None:
            timesteps = self.scheduler.timesteps
        else:
            offset = self.scheduler.config.get("steps_offset", 0)
            init_timestep = int(num_inference_steps * eta) + offset
            init_timestep = min(init_timestep, num_inference_steps)
            t_start = max(num_inference_steps - init_timestep + offset, 0)
            timesteps = self.scheduler.timesteps[t_start:]
            num_inference_steps = num_inference_steps - t_start
            latent_timestep = timesteps[:1]

        # 5. Prepare latent variables
        num_channels_latents = self.unet.in_channels
        # latents = self.prepare_latents(
        #     batch_size * num_images_per_prompt,
        #     num_channels_latents,
        #     height,
        #     width,
        #     text_embeddings_type,
        #     device,
        #     generator,
        #     latents,
        # )
        # Latent:
        if image is None:  # txt2img
            latent_size = (1, self.unet.in_channels, height // 8, width // 8)
            latents = torch.randn(latent_size, generator=torch.manual_seed(seed))
            if len(extra_seeds) > 0:
                print('Use region based seeding: ', extra_seeds)
                multi_latents = [torch.randn(latent_size,
                                             generator=torch.manual_seed(_seed)) for _seed in extra_seeds.values()]
                img_where_color_mask = _get_binary_mask(seperated_word_contexts, extra_seeds, dtype=latents[0].dtype,
                                                        size=latent_size[-2:])
                foreground = (sum(img_where_color_mask) > 0).squeeze()
                # sum seeds weighted by masks
                summed_multi_latents = sum(
                    _latents * _mask for _latents, _mask in zip(multi_latents, img_where_color_mask))
                latents[:, :, foreground] = summed_multi_latents[:, :, foreground]
            latents = latents.to(device)
            latents = latents * self.scheduler.init_noise_sigma
        else:
            image = preprocess(image)
            image = image.to(device=device)
            init_latent_dist = self.vae.encode(image).latent_dist
            init_latents = init_latent_dist.sample()
            init_latents = 0.18215 * init_latents
            noise = torch.randn(init_latents.shape).to(device)
            # get latents
            init_latents = self.scheduler.add_noise(init_latents, noise, latent_timestep)
            latents = init_latents

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                step_index = (self.scheduler.timesteps == t).nonzero().item()
                sigma = self.scheduler.sigmas[step_index]

                latent_model_input = self.scheduler.scale_model_input(latents, t)

                # _t = t if not is_mps else t.float()
                encoder_hidden_states.update({
                    "SIGMA": sigma,
                    "WEIGHT_FUNCTION": weight_function,
                })
                noise_pred_text = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=encoder_hidden_states,
                ).sample

                latent_model_input = self.scheduler.scale_model_input(latents, t)

                uncond_encoder_hidden_states.update({
                    "SIGMA": sigma,
                    "WEIGHT_FUNCTION": lambda w, sigma, qk: 0.0,
                })
                noise_pred_uncond = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=uncond_encoder_hidden_states,
                ).sample

                noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                )

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # 8. Post-processing
        image = self.decode_latents(latents)

        # 9. Run safety checker
        # image, has_nsfw_concept = self.run_safety_checker(image, device, text_embeddings_type)
        has_nsfw_concept = False

        # 10. Convert to PIL
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)


if __name__ == '__main__':

    img = paint_with_words(
        specfile="images/test_image.psd",
        num_inference_steps=30,
        guidance_scale=7.5,
        device="cuda:0",
        # hf_model_path="runwayml/stable-diffusion-v1-5",
        seed=420,
        weight_function=lambda w, sigma, qk: 1.8 * w * math.log(1 + sigma**2) * qk.std(),
    )

    img = np.array(img)[:,:,::-1]
    print(img)