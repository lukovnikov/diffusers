import torch
from diffusers import StableDiffusionPanoramaPipeline, DDIMScheduler, StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_multidiffusion_img2img import \
    StableDiffusionPipelineMultidiffusionImageToImage
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_multilevel_multidiffusion import \
    StableDiffusionHighresPipeline

if __name__ == '__main__':
    # for debugging
    model_ckpt = "runwayml/stable-diffusion-v1-5"
    scheduler = DDIMScheduler.from_pretrained(model_ckpt, subfolder="scheduler")
    pipe1 = StableDiffusionPipeline.from_pretrained(model_ckpt, scheduler=scheduler, torch_dtype=torch.float16).to("cuda")
    pipe2 = StableDiffusionPipelineMultidiffusionImageToImage.from_pretrained(model_ckpt, scheduler=scheduler, torch_dtype=torch.float16).to("cuda")

    prompt = "a photo of the dolomites"
    image = pipe1(prompt).images[0]

    print(image)

    srimage = pipe2(prompt, image, upscale=2, overlap=256).images[0]