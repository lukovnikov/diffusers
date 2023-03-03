import torch
from diffusers import StableDiffusionPanoramaPipeline, DDIMScheduler
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_multilevel_multidiffusion import \
    StableDiffusionHighresPipeline

if __name__ == '__main__':
    # for debugging
    model_ckpt = "runwayml/stable-diffusion-v1-5"
    scheduler = DDIMScheduler.from_pretrained(model_ckpt, subfolder="scheduler")
    pipe = StableDiffusionHighresPipeline.from_pretrained(model_ckpt, scheduler=scheduler, torch_dtype=torch.float16)

    pipe = pipe.to("cuda")

    prompt = "a photo of the dolomites"
    image = pipe(prompt).images[0]