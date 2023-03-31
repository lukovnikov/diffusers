import math
from PIL import Image, ImageDraw, ImageFont

from typing import Callable
import json
import os
from copy import deepcopy


class ImgDB:
    def __init__(self, path=None):
        super().__init__()
        self.d = []
        self.path = path

    def add(self, data: dict):
        self.d.append(data)

    def filter(self, attrs: dict):
        ret = []
        for datarow in self.d:
            discard = False
            for k, v in attrs.items():
                if isinstance(v, Callable):
                    if k not in datarow or not v(datarow[k]):
                        discard = True
                        break
                else:
                    if k not in datarow or datarow[k] != v:
                        discard = True
                        break
            if not discard:
                ret.append(datarow)
        return ret

    def save(self, path=None, force_overwrite=False):
        path = self.path if path is None else path
        d = deepcopy(self.d)
        for data in d:
            data["img"] = data["img"].tobytes().decode("latin1")
        if path is None:
            raise Exception("no save path specified")
        if os.path.exists(path) and not force_overwrite:
            raise Exception("path exists!")
        with open(path, "w") as f:
            json.dump(d, f)


def image_grid(imgs, cols):
    numrows = round(math.ceil(len(imgs) / cols))
    margin = 5

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w + margin * (cols - 1), numrows * h + margin * (numrows - 1)),
                     color=(255, 255, 255))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * (w + margin), i // cols * (h + margin)))
    return grid


def draw_text(text, width, height, fontsize=50):
    img = Image.new('RGB', size=(width, height), color=(255,255,255))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", fontsize)

    textwidth, textheight = draw.textsize(text, font)
    width, height = img.size
    x = width / 2 - textwidth / 2
    y = height / 2 - textheight / 2

    draw.text((x, y), text, font=font, fill='black')
    return img