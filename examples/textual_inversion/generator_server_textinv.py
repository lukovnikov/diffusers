import collections
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


def load_model(paths, dtype=torch.float16, use_ddim=False, use_dpm=False, basepipe=None):
    pipe = basepipe
    if isinstance(paths, str):
        paths = [paths]

    repldict = {}
    totalextratokens = 0
    globalstep = None
    for path in paths:
        spec = torch.load(os.path.join(path, "learned_embeddings.bin"))
        if pipe is None:
            pipe = StableDiffusionPipeline.from_pretrained(spec["pretrained_model_name_or_path"], torch_dtype=dtype)
        if globalstep is None:
            globalstep = spec["global_step"]
        # create replacement dictionary, extend tokenizer and embeddings and copy trained vectors
        for token, vectors in spec["tokens"].items():
            num_token_vectors = vectors.size(0)
            print(f"Adding {num_token_vectors} extra tokens for token '{token}' to tokenizer.")
            extratokens = [f"<extra-token-{i+totalextratokens}>" for i in range(num_token_vectors)]
            num_added_tokens = pipe.tokenizer.add_tokens(extratokens)
            assert num_added_tokens == len(extratokens), "Extra tokens must not be present in tokenizer."
            repldict[token] = " ".join(extratokens)
            tokenizervocab = pipe.tokenizer.get_vocab()

            print(f"Extending token embedder with {num_token_vectors} extra token vectors and loading saved vectors.")
            pipe.text_encoder.resize_token_embeddings(len(pipe.tokenizer))
            token_embeds = pipe.text_encoder.get_input_embeddings().weight.data
            for i, extratoken in enumerate(extratokens):
                token_embeds[tokenizervocab[extratoken], :] = vectors[i, :]

            totalextratokens += len(extratokens)
    print(f"Loaded a total of {totalextratokens} vectors for {len(paths)} new concepts.")

    if use_ddim:
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    if use_dpm:
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    return pipe, repldict, globalstep


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def get_prompts(subject="the hta dog", instancetype="object"):
    if instancetype == "object":
        prompts = [
            f"a photo of {subject}",
            f"a photo of {subject} in a bucket, 4k, highly detailed, realistic, olympus, fujifilm",
            f"a photo of {subject} in front of the eifel tower, 4k, highly detailed, realistic, olympus, fujifilm",
             f"a plush toy version of {subject}",
            f"a dramatic oil painting of {subject}, artistic, greg rutkowski, dramatic harsh light, trending on artstation",
            f"an oil painting of {subject} in the style of vincent van gogh",
        ]
    elif instancetype == "person":
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
         gpu:int=0,
         instancetype:str="object"):
    loc = locals()

    logfile = open("log.txt", "w")
    print(f"generation script called with args: {loc}")
    logfile.write(f"generation script called with args: {loc}")
    device = torch.device("cuda", gpu)
    pipe, repldict, globalstep = load_model(modeldir, use_ddim=False, use_dpm=True)  #StableDiffusionPipeline.from_pretrained(outputdir, torch_dtype=torch.float16).to(device)
    pipe = pipe.to(device)

    prompts = get_prompts(concept, instancetype) + get_prompts(conceptclass, instancetype)[0:1]
    print(f"prompts: {prompts}")

    imgperprompt = 4
    allimages = []

    for i, prompt in enumerate(prompts):
        print(f"running prompt: {prompt}")
        logfile.write(f"running prompt: {prompt}")
        for k, v in repldict.items():
            prompt = prompt.replace(k, v)
        print(f"Actual prompt: {prompt}")
        generator = torch.Generator(device=device)
        generator.manual_seed(42)
        images = pipe(prompt, num_inference_steps=32, guidance_scale=7.5, eta=0.,
                      generator=generator, num_images_per_prompt=imgperprompt, output_type="pil").images
        allimages.append(list(images))
        # imggrid = image_grid(images, 1, imgperprompt)

    print("done generating")
    # allimages = list(zip(allimages[:len(allimages)//2], allimages[len(allimages)//2:]))     # get instance and class versions next to each other
    # allimages = [img for imgs in allimages for img in imgs] # flatten
    allimages = list(zip(*allimages))  # transpose
    allimages = [img for imgs in allimages for img in imgs]  # flatten
    print(len(allimages), len(images), len(prompts))
    allgrid = image_grid(allimages, imgperprompt, len(prompts))

    allgrid.save(os.path.join(outputdir, f"grid_at_step_{globalstep}") + ".png")
    print("grid saved")


class MyHandler(FileSystemEventHandler):
    def __init__(self, outputdir:str="none",
                 watchdir:str="none",
         concept:str="hta",
         conceptclass:str="hta",
         gpu:int=0,
         instancetype:str="thing"):
        self.outputdir = outputdir
        self.watchdir = watchdir
        self.concept = concept
        self.conceptclass = conceptclass
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
        if event.src_path.startswith(os.path.join(self.watchdir, "logs")):
            pass #self.logfile.write(f"A log file changed. Ignoring. {event.src_path}\n")
        else:
            try:
                if self.last_event_timestamp is not None and self.last_event_timestamp > time.time() - self.event_timeout:
                    pass
                else:
                    self.last_event_timestamp = time.time()
                    self.logfile.write(f"File changed: {event.src_path}\n")
                    self.logfile.write("running generator\n")
                    self.logfile.flush()
                    self.run_generator()
                    self.last_event_timestamp = time.time()
            except Exception as e:
                self.logfile.write(f"Exception happened: {e}\n")
                self.logfile.flush()

    def run_generator(self):
        run_generation(self.watchdir, self.outputdir, self.concept, self.conceptclass, gpu=self.gpu,
                       instancetype=self.instancetype)


def start_generator_server(
         outputdir:str="none",
         watchdir:str="none",
         concept:str="hta",
         conceptclass:str="hta",
         gpu:int=0,
         instancetype:str="thing"):
    if __name__ == "__main__":
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


def main(outputdir:str="none",
         watchdir:str="none",
         concept:str="hta",
         conceptclass:str="hta",
         gpu:int=0,
         instancetype:str="thing"):
    start_generator_server(outputdir=outputdir,
                           watchdir=watchdir,
                           concept=concept,
                           conceptclass=conceptclass,
                           gpu=gpu,
                           instancetype=instancetype)


if __name__ == '__main__':
   fire.Fire(main)