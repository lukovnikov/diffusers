import argparse
import json
import os

import torch

from diffusers import DDIMPipeline, DDIMScheduler, DDPMPipeline, DiffusionPipeline
from diffusers import LDMPipeline
from PIL import Image

from diffusers.schedulers.scheduling_ddim import _ddim_scheduler_from_ddpm_scheduler


def main(args):
    DEBUG = True
    pipeline = DiffusionPipeline.from_pretrained(args.loadmodel)
    if args.sampler == "ddim":
        ddimsched = _ddim_scheduler_from_ddpm_scheduler(pipeline.scheduler)
        pipeline = DDIMPipeline(pipeline.unet, ddimsched)
    pipeline = pipeline.to(torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 else "cpu"))

    pipeline.scheduler.config.clip_sample = args.clip_sample

    num_steps = args.numsteps if args.numsteps != -1 else pipeline.scheduler.num_train_timesteps
    print(f"Number of timesteps used for sampling: {num_steps}")

    generator = torch.manual_seed(42)


    if args.outputsubdir is None:
        args.outputsubdir = f"samples_{args.sampler}" + (f"_{args.numsteps}" if args.numsteps is not None else "")
    os.makedirs(os.path.join(args.savedir, args.outputsubdir), exist_ok=True)
    with open(os.path.join(args.savedir, args.outputsubdir, "config.json"), "w") as f:
        json.dump(args.__dict__, f, indent=4)
    os.makedirs(os.path.join(args.savedir, args.outputsubdir, "imgs"), exist_ok=True)

    cnt = 0  # how many images generated so far
    numdigits = len(str(int(args.numsamples)))

    while cnt < args.numsamples:
        # run pipeline in inference (sample random noise and denoise)
        if isinstance(pipeline, DDPMPipeline):
            images = pipeline(
                generator=generator,
                batch_size=args.batchsize,
                output_type="numpy",
                num_inference_steps=num_steps,
            ).images
        elif isinstance(pipeline, DDIMPipeline):
            images = pipeline(
                generator=generator,
                batch_size=args.batchsize,
                output_type="numpy",
                num_inference_steps=num_steps,
                use_clipped_model_output=True,
                eta=args.ddimeta,
            ).images
        elif isinstance(pipeline, LDMPipeline):
            images = pipeline(
                generator=generator,
                batch_size=args.batchsize,
                output_type="numpy",
                num_inference_steps=num_steps,
                eta=args.ddimeta,
            ).images

        # denormalize the images and save to tensorboard
        images_processed = (images * 255).round().astype("uint8")  # .transpose(0, 3, 1, 2)

        for img in list(images_processed):
            pilimage = Image.fromarray(img, mode="RGB")
            pilimage.save(
                os.path.join(args.savedir, args.outputsubdir, "imgs", f"{{:0{numdigits}d}}.png".format(cnt))
            )
            cnt += 1

        print(f"generated {cnt} images")

    print(f"generated {cnt} images and saved in {os.path.join(args.savedir, args.outputsubdir, 'imgs')}")

    print(pipeline)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--loadmodel", type=str, default="ddpm-model-64")
    parser.add_argument("--savedir", type=str, default=None)
    parser.add_argument("--sampler", type=str, default="ddpm")  # can be ddpm or ddim
    parser.add_argument("--numsteps", type=int, default=None)
    parser.add_argument("--ddimeta", type=float, default=0.0)
    parser.add_argument("--batchsize", type=int, default=16)
    parser.add_argument("--outputsubdir", type=str, default=None)
    parser.add_argument("--numsamples", type=int, default=10)
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--clip_sample", action="store_true", default=False)

    args = parser.parse_args()
    if args.savedir is None:
        args.savedir = args.loadmodel
    main(args)
