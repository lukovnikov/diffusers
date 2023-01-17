import os
import fire
from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch
from PIL import Image


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def get_prompts(subject="the xvw dog"):
    prompts = [
        f"a photo of {subject}, 4k, highly detailed, realistic, super sharp, high fidelity",
        f"a photo of {subject} in a bucket, 4k, highly detailed, realistic, super sharp, high fidelity",
        f"a photo of {subject} in front of the eifel tower, 4k, highly detailed, realistic, super sharp, high fidelity",
        f"a plush toy version of {subject}",
        f"a bronze statue of {subject} standing in the street",
        f"a dramatic oil painting of {subject}, artistic, greg rutkowski, dramatic harsh light, 4k, trending on artstation",
        f"an oil painting of {subject} in the style of vincent van gogh",
    ]
    return prompts


def main(outputdir:str="none", concept:str="xvw", conceptclass:str="xvw", step:int=0, gpu:int=0):
    loc = locals()
    logfile = open("log.txt", "w")
    print(f"generation script called with args: {loc}")
    logfile.write(f"generation script called with args: {loc}")
    device = torch.device("cuda", gpu)
    pipe = StableDiffusionPipeline.from_pretrained(outputdir, torch_dtype=torch.float16).to(device)

    use_ddim = False
    if use_ddim:
        ddimsched = DDIMScheduler(
            num_train_timesteps = pipe.scheduler.num_train_timesteps,
            beta_start = pipe.scheduler.beta_start,
            beta_end = pipe.scheduler.beta_end,
            beta_schedule = pipe.scheduler.beta_schedule,
        )
        assert torch.allclose(ddimsched.alphas_cumprod, pipe.scheduler.alphas_cumprod)
        pipe.scheduler = ddimsched

    prompts = get_prompts(concept) + get_prompts(conceptclass)
    print(f"prompts: {prompts}")
    imgperprompt = 6
    allimages = []
    for prompt in prompts:
        print(f"running prompt: {prompt}")
        logfile.write(f"running prompt: {prompt}")
        generator = torch.Generator(device=device)
        generator.manual_seed(42)
        images = pipe(prompt, num_inference_steps=50, guidance_scale=7.5, eta=0.,
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