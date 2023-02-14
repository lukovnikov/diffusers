import os
import pathlib

import fire
from diffusers import StableDiffusionPipeline, DDIMScheduler, DPMSolverMultistepScheduler
import torch
from PIL import Image
import shelve

from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.customnn import StructuredCrossAttention, StructuredCLIPTextTransformer, \
    StructuredStableDiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_layers2img import \
    StableDiffusionPipelineLayers2ImageV1


def load_model(path, dtype=torch.float16, use_ddim=False, use_dpm=False):
    extra_token_map = None
    affects_prior = False
    if pathlib.Path(os.path.join(path, "custom.pth")).is_file():
        d = torch.load(os.path.join(path, "custom.pth"))
        pipe = StableDiffusionPipeline.from_pretrained(d["source"], torch_dtype=dtype)
        print(f"Adding 3000 extra tokens to tokenizer")
        pipe.tokenizer.add_tokens([f"<extra-token-{i}>" for i in range(3000)])
        print(f"Extending token embedder with 3000 extra token vectors, randomly initialized")
        embed = pipe.text_encoder.text_model.embeddings.token_embedding
        extra_embed = torch.nn.Embedding(3000, embed.embedding_dim)
        embed.weight.data = torch.cat([embed.weight.data, extra_embed.weight.data.to(embed.weight.dtype)])
        embed.num_embeddings += 3000
        # pipe.unet.dtype = dtype

        if "custom_embeddings" in d:
            print(f"Reloading {len(d['custom_embeddings'])} custom vectors into embedding layer")
            extra_token_map = {}
            extra_embed_vectors = d["custom_embeddings"]
            tokenizervocab = pipe.tokenizer.get_vocab()
            for source_extra_token, extra_vectors in extra_embed_vectors.items():
                extra_token_map[source_extra_token] = []
                print(f"Reloading {len(extra_vectors)} vectors for {source_extra_token}")
                for i, extra_vector in enumerate(extra_vectors):
                    embed.weight.data[tokenizervocab[f"<extra-token-{i}>"]] = extra_vector
                    extra_token_map[source_extra_token].append(f"<extra-token-{i}>")

            if "use_mem" in d:
                print("casting CrossAttentions to Structured")
                unet = pipe.unet
                _c = 0
                for m in unet.modules():
                    if isinstance(m, BasicTransformerBlock):
                        m.attn2.__class__ = StructuredCrossAttention
                        if m.only_cross_attention:
                            m.attn1.__class__ = StructuredCrossAttention
                        _c += 1
                print(f"Recast {_c} modules")
                print(f"Casting CLIPTextTransformer to StructuredCLIPTextTransformer")
                pipe.text_encoder.text_model.__class__ = StructuredCLIPTextTransformer

                print(f"Reloading saved memory specs")
                # recreate tokenid to mem map and mem to tokenid based on the loaded vectors
                new_extra_token_map = {}
                mem_token_map = {}
                for source_extra_token, extra_tokens in extra_token_map.items():
                    new_extra_token_map[source_extra_token] = [extra_tokens[0]]
                    mem_token_map[extra_tokens[0]] = extra_tokens[1:]
                extra_token_map = new_extra_token_map       # only used to replace text before running through model

                supercellsize = len(list(mem_token_map.values())[0])
                numsupercells = len(extra_token_map)
                _c = 0
                tokenid_to_mem_map = torch.zeros(embed.num_embeddings, dtype=torch.long)
                mem_to_tokenid_map = torch.zeros(numsupercells, supercellsize, dtype=torch.long)
                for source_extra_token, extra_tokens in extra_token_map.items():
                    tokenid_to_mem_map[tokenizervocab[extra_tokens[0]]] = _c + 1
                    mem_to_tokenid_map[_c, :] = torch.tensor(
                        [tokenizervocab[mem_token] for mem_token in mem_token_map[extra_tokens[0]]])
                    _c += 1
                pipe.text_encoder.text_model.init_mem(tokenid_to_mem_map=tokenid_to_mem_map,
                                                 mem_to_tokenid_map=mem_to_tokenid_map)

                print(f"Casting Pipeline to Structured")
                pipe.__class__ = StructuredStableDiffusionPipeline

        if "unet_state_dict" in d:
            print(f"Reloading (partial) state dict for Unet")
            pipe.unet.load_state_dict(d["unet_state_dict"], strict=False)

        if "text_encoder_state_dict" in d:
            print(f"Reloading (partial) state dict for text encoder")
            pipe.unet.load_state_dict(d["text_encoder_state_dict"], strict=False)

    else:
        pipe = StableDiffusionPipeline.from_pretrained(path, torch_dtype=torch.float16)
        print(f"Adding 300 extra tokens to tokenizer")
        pipe.tokenizer.add_tokens([f"<extra-token-{i}>" for i in range(300)])
        affects_prior = True

    if use_ddim:
        oldsched = pipe.scheduler
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.scheduler.set_timesteps(100)
        assert torch.allclose(oldsched.alphas_cumprod, pipe.scheduler.alphas_cumprod)

    if use_dpm:
        oldsched = pipe.scheduler
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.scheduler.set_timesteps(30)
        assert torch.allclose(oldsched.alphas_cumprod, pipe.scheduler.alphas_cumprod)
    return pipe, extra_token_map, affects_prior


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def get_prompts(subject="the xvw dog", instancetype="thing"):
    if instancetype == "thing":
        prompts = [
            f"a photo of {subject}",
            f"a photo of {subject} in a bucket, 4k, highly detailed, realistic, olympus, fujifilm",
            f"a photo of {subject} in front of the eifel tower, 4k, highly detailed, realistic, olympus, fujifilm",
             f"a plush toy version of {subject}",
            f"a dramatic oil painting of {subject}, artistic, greg rutkowski, dramatic harsh light, trending on artstation",
            f"an oil painting of {subject} in the style of vincent van gogh",
        ]
    elif instancetype == "portrait":
        prompts = [
            f"a photo of {subject}",
            f"a portrait photo of {subject} smiling, 4k, highly detailed, realistic, olympus, fujifilm",
            f"a portrait of {subject} in anime style, manga style, studio ghibli, 2D, masterpiece, detailed",
            f"a dramatic digital portrait painting of {subject}, artistic, greg rutkowski, dramatic harsh light, 4k, trending on artstation",
            f"an oil portrait painting of {subject} in the style of vincent van gogh",
            f"a bust statue of {subject} head, 4k 8k 5k, olympus, canon r3, fujifilm xt3",
        ]
    return prompts


def main(outputdir:str="none",
         concept:str="xvw",
         conceptclass:str="xvw",
         step:int=0,
         gpu:int=0,
         instancetype:str="thing"):
    loc = locals()
    logfile = open("log.txt", "w")
    print(f"generation script called with args: {loc}")
    logfile.write(f"generation script called with args: {loc}")
    device = torch.device("cuda", gpu)
    pipe, extra_token_map, affects_prior = load_model(outputdir, use_ddim=False, use_dpm=True)  #StableDiffusionPipeline.from_pretrained(outputdir, torch_dtype=torch.float16).to(device)
    pipe = pipe.to(device)

    if False and affects_prior:
        prompts = get_prompts(concept, instancetype) + get_prompts(conceptclass, instancetype)
        prompts = list(zip(prompts[:len(prompts)//2], prompts[len(prompts)//2:]))     # get instance and class versions next to each other
        prompts = [prompt for prompts_ in prompts for prompt in prompts_]  # flatten
    else:
        prompts = get_prompts(conceptclass, instancetype)[0:1] + get_prompts(concept, instancetype)
    print(f"prompts: {prompts}")
    imgperprompt = 5
    allimages = []
    for i, prompt in enumerate(prompts):
        print(f"running prompt: {prompt}")
        logfile.write(f"running prompt: {prompt}")
        generator = torch.Generator(device=device)
        generator.manual_seed(42)
        # replace source extra tokens with ones from model (how many?)
        if extra_token_map is not None:
            for source_extra_token, repl_tokens in extra_token_map.items():
                prompt = prompt.replace(source_extra_token, " ".join(repl_tokens))
        images = pipe(prompt, num_inference_steps=32, guidance_scale=7.5, eta=0.,
                      generator=generator, num_images_per_prompt=imgperprompt, output_type="pil").images
        allimages.append(list(images))
        # imggrid = image_grid(images, 1, imgperprompt)

    print("done generating")
    # allimages = list(zip(allimages[:len(allimages)//2], allimages[len(allimages)//2:]))     # get instance and class versions next to each other
    # allimages = [img for imgs in allimages for img in imgs] # flatten
    allimages = list(zip(*allimages))  # transpose
    allimages = [img for imgs in allimages for img in imgs]  # flatten
    print(len(allimages), len(images), len(prompts))
    allgrid = image_grid(allimages, imgperprompt, len(prompts))

    allgrid.save(os.path.join(outputdir, f"grid_at_step_{step}") + ".png")
    print("grid saved")


if __name__ == '__main__':
   fire.Fire(main)