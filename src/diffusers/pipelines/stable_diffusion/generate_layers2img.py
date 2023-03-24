import torch

from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_multidiffusion_paintbywords import \
    StableDiffusionPaintbywordsPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_paintbywords import \
    StableDiffusionPipelineLayers2ImageV1
import psd_tools as psd

from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_paintbywords_ediffi import \
    StableDiffusionPipelineLayers2ImageEdiffi

if __name__ == '__main__':
    # pipe = StableDiffusionPipelineLayers2ImageV1.from_pretrained("runwayml/stable-diffusion-v1-5")
    pipe = StableDiffusionPaintbywordsPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    #pipe = StableDiffusionPipelineLayers2ImageEdiffi.from_pretrained("runwayml/stable-diffusion-v1-5")
    device = torch.device("cuda", 0)
    pipe = pipe.to(device)

    # pipe._encode_prompt(spec, device, 2, do_classifier_free_guidance=True)

    psdimg = psd.PSDImage.open("lion.psd")
    ret = pipe(psdimg, bootstrap_ratio=0.3, num_inference_steps=50)     # for multidiffusion
    #ret = pipe(psdimg, mode="neg", wprime=1, threshold=0.3, expand_negative_prompt=True, expose_start_end_token=False)     # for attention-based
    print(ret)


# TODO: check that the cross-attention mask has the desired behavior
# TODO: how does ControlNet work and how do we integrate?
