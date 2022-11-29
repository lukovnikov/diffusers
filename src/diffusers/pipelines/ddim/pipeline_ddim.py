# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from typing import Optional, Tuple, Union

import torch

from diffusers.models.unet_2d import UNet2DModel
from diffusers.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.schedulers.scheduling_ddim import DDIMExtendedScheduler, DDIMScheduler, DistilledDDIMScheduler


class DDIMPipeline(DiffusionPipeline):
    r"""
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Parameters:
        unet ([`UNet2DModel`]): U-Net architecture to denoise the encoded image.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image. Can be one of
            [`DDPMScheduler`], or [`DDIMScheduler`].
    """

    def __init__(self, unet, scheduler):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        generator: Optional[torch.Generator] = None,
        eta: float = 0.0,
        num_inference_steps: int = 50,
        use_clipped_model_output: Optional[bool] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        predict_epsilon: bool = True,
        **kwargs,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            generator (`torch.Generator`, *optional*):
                A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
                deterministic.
            eta (`float`, *optional*, defaults to 0.0):
                The eta parameter which controls the scale of the variance (0 is DDIM and 1 is one type of DDPM).
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            use_clipped_model_output (`bool`, *optional*, defaults to `None`):
                if `True` or `False`, see documentation for `DDIMScheduler.step`. If `None`, nothing is passed
                downstream to the scheduler. So use `None` for schedulers which don't support this argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipeline_utils.ImagePipelineOutput`] instead of a plain tuple.
            predict_epsilon (`bool`, *optional*, defaults to True):
                Whether the Unet model should be used to predict eps (as opposed to x0).
        Returns:
            [`~pipeline_utils.ImagePipelineOutput`] or `tuple`: [`~pipelines.utils.ImagePipelineOutput`] if
            `return_dict` is True, otherwise a `tuple. When returning a tuple, the first element is a list with the
            generated images.
        """

        # Sample gaussian noise to begin loop
        image = torch.randn(
            (batch_size, self.unet.in_channels, self.unet.sample_size, self.unet.sample_size),
            generator=generator,
        )
        image = image.to(self.device)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        # Ignore use_clipped_model_output if the scheduler doesn't accept this argument
        accepts_use_clipped_model_output = "use_clipped_model_output" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_kwargs = {}
        if accepts_use_clipped_model_output:
            extra_kwargs["use_clipped_model_output"] = use_clipped_model_output

        timesteps = self.scheduler.timesteps
        timesteps = torch.cat([timesteps, -1 * torch.ones_like(timesteps)[:1]], 0)
        timepairs = list(zip(timesteps[:-1], timesteps[1:]))
        for t in self.progress_bar(timepairs):
            start_t, end_t = t
            # 1. predict noise model_output
            model_output = self.unet(image, start_t).sample

            # 2. predict previous mean of image x_t-1 and add variance depending on eta
            # eta corresponds to η in paper and should be between [0, 1]
            # do x_t -> x_t-1
            image = self.scheduler.step(
                model_output, t, image, eta, **extra_kwargs
            ).prev_sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)


class DistilledDDIMPipeline(DDIMPipeline):
    pass


def tst_ddim():
    model_id = "google/ddpm-cifar10-32"

    unet = UNet2DModel.from_pretrained(model_id)
    ddim_scheduler = DDIMScheduler()

    ddim = DDIMPipeline(unet=unet, scheduler=ddim_scheduler)
    ddim.scheduler.__class__ = DDIMExtendedScheduler
    ddim.to(torch.device("cpu"))
    ddim.set_progress_bar_config(disable=None)


    generator = torch.manual_seed(42)
    ddim_images = ddim(
        batch_size=4,
        generator=generator,
        num_inference_steps=1000,
        eta=0.0,
        output_type="numpy",
        use_clipped_model_output=True,  # Need this to make DDIM match DDPM
    ).images


def tst_distilled_ddim():
    model_id = "google/ddpm-cifar10-32"

    unet = UNet2DModel.from_pretrained(model_id)
    sched = DistilledDDIMScheduler()

    ddim = DDIMPipeline(unet=unet, scheduler=sched)
    ddim.to(torch.device("cpu"))
    ddim.set_progress_bar_config(disable=None)

    # Sample gaussian noise to begin loop
    image = torch.randn(
        (1, unet.in_channels, unet.sample_size, unet.sample_size)
    )

    # set step values
    sched.set_timesteps(512)
    sched.set_timesteps(256, prevtimesteps=sched.timesteps)
    sched.set_timesteps(128, prevtimesteps=sched.timesteps)
    sched.set_timesteps(64, prevtimesteps=sched.timesteps)
    sched.set_timesteps(32, prevtimesteps=sched.timesteps)
    sched.set_timesteps(16, prevtimesteps=sched.timesteps)
    sched.set_timesteps(8, prevtimesteps=sched.timesteps)
    sched.set_timesteps(4, prevtimesteps=sched.timesteps)

    timesteps = sched.timesteps
    timesteps = torch.cat([timesteps, -1 * torch.ones_like(timesteps)[:1]], 0)
    timepairs = list(zip(timesteps[:-1], timesteps[1:]))
    for t in timepairs:
        start_t, end_t = t
        # 1. predict noise model_output
        model_output = unet(image, start_t).sample

        # 2. predict previous mean of image x_t-1 and add variance depending on eta
        # eta corresponds to η in paper and should be between [0, 1]
        # do x_t -> x_t-1
        image = sched.step(model_output, t, image, 0.).prev_sample

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()


if __name__ == '__main__':
    tst_distilled_ddim()

