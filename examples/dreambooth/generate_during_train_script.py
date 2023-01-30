import os
import pathlib

import fire
from diffusers import StableDiffusionPipeline, DDIMScheduler, StructuredUNet2DConditionModel, \
    DPMSolverMultistepScheduler
import torch
from PIL import Image
import shelve

from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.customnn import StructuredCrossAttention


def load_model(path, dtype=torch.float16, use_ddim=False, use_dpm=False):
    if pathlib.Path(os.path.join(path, "custom.pth")).is_file():
        d = torch.load(os.path.join(path, "custom.pth"))
        pipe = StableDiffusionPipeline.from_pretrained(d["source"], torch_dtype=dtype)
        # pipe.unet.dtype = dtype
        for k in d["replace"]:
            m = getattr(pipe, k)
            m.load_state_dict(d["replace"][k], strict=False)

        if "convert" in d and d["convert"] == "structured":
            print("casting CrossAttentions, Unet to Structured")
            unet = pipe.unet
            _c = 0
            for m in unet.modules():
                if isinstance(m, BasicTransformerBlock):
                    m.attn2.__class__ = StructuredCrossAttention
                    if m.only_cross_attention:
                        m.attn1.__class__ = StructuredCrossAttention
                    _c += 1
            print(f"Replaced {_c} modules")
            print("Replacing Unet with StructuredUnet and reloading mem")
            unet.__class__ = StructuredUNet2DConditionModel
            mem = d["unet-mem"].to(dtype)
            tokenid_to_mem_map = d["unet-tokenid_to_mem_map"]
            unet.init_mem(mem=mem, tokenid_to_mem_map=tokenid_to_mem_map)
    else:
        pipe = StableDiffusionPipeline.from_pretrained(path, torch_dtype=torch.float16)

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
    return pipe


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
            f"a photo of {subject}, 4k, highly detailed, realistic, olympus, fujifilm",
            f"a photo of {subject} in a bucket, 4k, highly detailed, realistic, olympus, fujifilm",
            f"a photo of {subject} in front of the eifel tower, 4k, highly detailed, realistic, olympus, fujifilm",
             f"a plush toy version of {subject}",
            f"a dramatic oil painting of {subject}, artistic, greg rutkowski, dramatic harsh light, trending on artstation",
            f"an oil painting of {subject} in the style of vincent van gogh",
        ]
    elif instancetype == "portrait":
        prompts = [
            f"a portrait photo of {subject}, 4k, highly detailed, realistic, olympus, fujifilm, gigapixel, award-winning photography, instagram",
            f"a portrait photo of {subject} in front of the eifel tower, 4k, highly detailed, realistic, olympus, fujifilm",
            f"a cute portrait of {subject} in anime style, manga style, studio ghibli, 2D, cute, masterpiece",
            f"a dramatic oil portrait painting of {subject}, artistic, greg rutkowski, dramatic harsh light, 4k, trending on artstation",
            # f"an oil portrait painting of {subject} in the style of vincent van gogh",
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
    pipe = load_model(outputdir, use_ddim=False, use_dpm=True).to(device)  #StableDiffusionPipeline.from_pretrained(outputdir, torch_dtype=torch.float16).to(device)

    prompts = get_prompts(concept, instancetype) + get_prompts(conceptclass, instancetype)
    print(f"prompts: {prompts}")
    imgperprompt = 5
    allimages = []
    for i, prompt in enumerate(prompts):
        print(f"running prompt: {prompt}")
        logfile.write(f"running prompt: {prompt}")
        generator = torch.Generator(device=device)
        generator.manual_seed(42)
        images = pipe(prompt, num_inference_steps=32, guidance_scale=7.5, eta=0.,
                      generator=generator, num_images_per_prompt=imgperprompt, output_type="pil").images
        allimages.append(list(images))
        # imggrid = image_grid(images, 1, imgperprompt)

    print("done generating")
    allimages = list(zip(allimages[:len(allimages)//2], allimages[len(allimages)//2:]))     # get instance and class versions next to each other
    allimages = [img for imgs in allimages for img in imgs] # flatten
    allimages = list(zip(*allimages))  # transpose
    allimages = [img for imgs in allimages for img in imgs]  # flatten
    print(len(allimages), len(images), len(prompts))
    allgrid = image_grid(allimages, imgperprompt, len(prompts))

    allgrid.save(os.path.join(outputdir, f"grid_at_step_{step}") + ".png")
    print("grid saved")


if __name__ == '__main__':
    fire.Fire(main)