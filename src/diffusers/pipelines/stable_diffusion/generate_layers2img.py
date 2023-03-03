import torch
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_paintbywords import \
    StableDiffusionPipelineLayers2ImageV1
import psd_tools as psd


if __name__ == '__main__':
    spec = {
        "global": {
            "pos": "global description",
            "neg": "negative prompt for global description"
        },
        "layers": [
            {"pos": "first layer description", "neg": "negative prompt for first layer"},
            {"pos": "second layer description", "neg": ""}

        ]
    }

    pipe = StableDiffusionPipelineLayers2ImageV1.from_pretrained("runwayml/stable-diffusion-v1-5")
    device = torch.device("cuda", 0)
    pipe = pipe.to(device)

    # pipe._encode_prompt(spec, device, 2, do_classifier_free_guidance=True)

    psdimg = psd.PSDImage.open("lion.psd")
    ret = pipe(psdimg, num_images_per_prompt=2)
    print(ret)


# TODO: check that the cross-attention mask has the desired behavior
# TODO: how does ControlNet work and how do we integrate?
