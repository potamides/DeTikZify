from base64 import decodebytes
from io import BytesIO

from PIL import Image, ImageChops, ImageOps
import requests
from transformers.utils.hub import is_remote_url

def convert(img, filetype):
    img.save(imgbytes:=BytesIO(), format=filetype)
    return Image.open(imgbytes)

def _preprocess(raw, bg):
    if not isinstance(raw, Image.Image):
        raw = Image.open(raw)
    filetype = raw.format
    # https://stackoverflow.com/a/62414364
    background = Image.new('RGBA', raw.size, bg)
    alpha_composite = Image.alpha_composite(background, raw.convert("RGBA"))
    return alpha_composite.convert("RGB"), filetype

def _postprocess(img, filetype):
    if filetype: # preserve filetype
        return convert(img, filetype)
    return img

# https://stackoverflow.com/a/10616717
def _trim(img, border):
    bg = Image.new(img.mode, img.size, border)
    diff = ImageChops.difference(img, bg)
    #diff = ImageChops.add(diff, diff, 2.0, -10)
    return img.crop(bbox) if (bbox:=diff.getbbox()) else img

def trim(raw, border="white"):
    img, filetype = _preprocess(raw, bg=border)
    return _postprocess(_trim(img, border), filetype)

def expand(raw, size, trim=False, border="white"):
    """Expand image to a square of size {size}. Optionally trims borders first."""
    img, filetype = _preprocess(raw, bg=border)
    img = _trim(img, border=border) if trim else img
    img = ImageOps.pad(img, 2 * (max(img.size),), color=border, method=Image.Resampling.LANCZOS)
    img = img.resize((size, size), Image.Resampling.LANCZOS)
    return _postprocess(img, filetype)

def load(image: Image.Image | str, bg="white"):
    if isinstance(image, str):
        if image.startswith("data:image/"):
            image = BytesIO(decodebytes(image.split(",")[1].encode()))
        elif is_remote_url(image):
            # https://stackoverflow.com/a/69791396
            headers = {'user-agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:68.0) Gecko/20100101 Firefox/68.0'}
            image = requests.get(image, stream=True, headers=headers).raw
        image = Image.open(image)
    image = ImageOps.exif_transpose(image)
    return _postprocess(*_preprocess(image, bg))
