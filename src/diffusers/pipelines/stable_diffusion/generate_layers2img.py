import torch

from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_multidiffusion_paintbywords import \
    StableDiffusionPaintbywordsPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_paintbywords import \
    StableDiffusionPipelineLayers2ImageV1
import psd_tools as psd


if __name__ == '__main__':
    # pipe = StableDiffusionPipelineLayers2ImageV1.from_pretrained("runwayml/stable-diffusion-v1-5")
    pipe = StableDiffusionPaintbywordsPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    device = torch.device("cuda", 0)
    pipe = pipe.to(device)

    # pipe._encode_prompt(spec, device, 2, do_classifier_free_guidance=True)

    psdimg = psd.PSDImage.open("lion.psd")
    ret = pipe(psdimg)
    print(ret)


# TODO: check that the cross-attention mask has the desired behavior
# TODO: how does ControlNet work and how do we integrate?
