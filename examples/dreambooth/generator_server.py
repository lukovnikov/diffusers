import json
import os
import pathlib

import fire
from diffusers import StableDiffusionPipeline, DDIMScheduler, DPMSolverMultistepScheduler
import torch
from PIL import Image
import time

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


def load_model(path, dtype=torch.float16, use_ddim=False, use_dpm=False):
    pipe = StableDiffusionPipeline.from_pretrained(path, torch_dtype=dtype)

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


def get_prompts(subject="the hta dog", instancetype="thing"):
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
            # f"a portrait of {subject} in anime style, manga style, studio ghibli, 2D, masterpiece, detailed",
            f"a dramatic digital portrait painting of {subject}, artistic, greg rutkowski, dramatic harsh light, 4k, trending on artstation",
            # f"an oil portrait painting of {subject} in the style of vincent van gogh",
            f"a bust statue of {subject} head, 4k 8k 5k, olympus, canon r3, fujifilm xt3",
        ]
    elif instancetype == "style":
        prompts = [
            f"a painting of a pretty young woman with red hair and blues eyes in the style of {subject}, dramatic light, masterpiece, detailed",
            f"a painting of a grumpy old man in the style of {subject}, dramatic light, masterpiece, detailed",
            f"a painting of a husky dog in the style of {subject}, dramatic light, masterpiece, detailed",
            f"a painting of a white mayonaise jar in the style of {subject}, dramatic light, masterpiece, detailed",
            f"a painting of an old church in the style of {subject}, dramatic light, masterpiece, detailed",
        ]
    return prompts


def run_generation(
         modeldir:str="none",
         outputdir:str="none",
         concept:str="hta",
         conceptclass:str="hta",
         step:int=0,
         gpu:int=0,
         instancetype:str="thing",
         numimgsperprompt:int=4,
         savegrid=True):
    loc = locals()

    logfile = open("log.txt", "w")
    print(f"generation script called with args: {loc}")
    logfile.write(f"generation script called with args: {loc}")
    device = torch.device("cuda", gpu)
    pipe = load_model(modeldir, use_ddim=False, use_dpm=True)  #StableDiffusionPipeline.from_pretrained(outputdir, torch_dtype=torch.float16).to(device)
    pipe = pipe.to(device)

    prompts = get_prompts(concept, instancetype) + get_prompts(conceptclass, instancetype)
    prompts = list(zip(prompts[:len(prompts)//2], prompts[len(prompts)//2:]))     # get instance and class versions next to each other
    prompts = [prompt for prompts_ in prompts for prompt in prompts_]  # flatten

    print(f"prompts: {prompts}")

    allimages = []

    for i, prompt in enumerate(prompts):
        print(f"running prompt: {prompt}")
        logfile.write(f"running prompt: {prompt}")
        generator = torch.Generator(device=device)
        generator.manual_seed(42)
        images = pipe(prompt, num_inference_steps=32, guidance_scale=7.5, eta=0.,
                      generator=generator, num_images_per_prompt=numimgsperprompt, output_type="pil").images
        allimages.append(list(images))
        # imggrid = image_grid(images, 1, imgperprompt)

    print("done generating")
    # allimages = list(zip(allimages[:len(allimages)//2], allimages[len(allimages)//2:]))     # get instance and class versions next to each other
    # allimages = [img for imgs in allimages for img in imgs] # flatten
    if savegrid:
        allimages = list(zip(*allimages))  # transpose
        allimages = [img for imgs in allimages for img in imgs]  # flatten
        print(len(allimages), len(images), len(prompts))
        allgrid = image_grid(allimages, numimgsperprompt, len(prompts))

        allgrid.save(os.path.join(outputdir, f"grid_at_step_{step}") + ".png")
        print("grid saved")
    else:
        os.makedirs(os.path.join(outputdir, f"withtoken"), exist_ok=True)
        os.makedirs(os.path.join(outputdir, f"notoken"), exist_ok=True)
        allimages = [img for imgs in allimages for img in imgs]  # flatten
        for i, img in enumerate(allimages):
            img.save(os.path.join(outputdir, "withtoken" if i % 2 == 0 else "notoken",
                                  f"sampled_image_{i}.png"))
        print("images saved")


class MyHandler(FileSystemEventHandler):
    def __init__(self, outputdir:str="none",
                 watchdir:str="none",
         concept:str="hta",
         conceptclass:str="hta",
         gpu:int=0,
         instancetype:str="thing",
         numimgsperprompt:int=4):
        self.outputdir = outputdir
        self.watchdir = watchdir
        self.concept = concept
        self.conceptclass = conceptclass
        self.numimgsperprompt = numimgsperprompt
        self.gpu = gpu
        self.instancetype = instancetype
        self.last_event_timestamp = None
        self.event_timeout = 5
        self.logfile = open(os.path.join(self.outputdir, "generator.logfile"), "a+")

    def on_modified(self, event):
        self.process_event(event)
    def on_created(self, event):
        self.process_event(event)

    def process_event(self, event):
        try:
            if self.last_event_timestamp is not None and self.last_event_timestamp > time.time() - self.event_timeout:
                pass
            else:
                self.last_event_timestamp = time.time()
                progresspath = os.path.join(self.watchdir, "progress.json")
                if os.path.exists(progresspath):
                    progressinfo = json.load(open(progresspath))
                    step = progressinfo["global_step"]
                    self.logfile.write(f"File changed: {event.src_path}, step: {step}\n")
                else:
                    self.logfile.write("progress does not exist\n")
                self.logfile.write("running generator\n")
            self.run_generator(step)
            self.last_event_timestamp = time.time()
        except Exception as e:
            self.logfile.write(f"Exception happened: {e}\n")

    def run_generator(self, step):
        run_generation(self.watchdir, self.outputdir, self.concept, self.conceptclass, step=step, gpu=self.gpu,
                       instancetype=self.instancetype, numimgsperprompt=self.numimgsperprompt, savegrid=True)


def main(
         outputdir:str="none",
         watchdir:str="__none",
         modeldir:str="__none",
         concept:str="hta",
         conceptclass:str="hta",
         gpu:int=0,
         instancetype:str="thing",
         numimgsperprompt:int=4):
    if watchdir != "__none":
        if not os.path.exists(watchdir):
            os.makedirs(watchdir, exist_ok=True)
        if not os.path.exists(outputdir):
            os.makedirs(outputdir, exist_ok=True)
        event_handler = MyHandler(outputdir=outputdir, watchdir=watchdir, concept=concept, conceptclass=conceptclass,
                                  gpu=gpu,
                                  instancetype=instancetype)
        observer = Observer()
        observer.schedule(event_handler, path=watchdir, recursive=True)
        observer.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
        observer.join()
    else:
        print(f"Generating images")
        assert modeldir != "__none"
        run_generation(modeldir, outputdir, concept, conceptclass, -1, gpu, instancetype, numimgsperprompt, savegrid=False)


if __name__ == '__main__':
   fire.Fire(main)