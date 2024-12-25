from base64 import b64decode
from io import BytesIO
from os.path import isfile

from PIL import Image, ImageChops, ImageOps
import requests
from transformers.utils.hub import is_remote_url

DUMMY_IMAGE = Image.new("RGB", (24, 24), color="white")

def convert(image, filetype):
    image.save(imgbytes:=BytesIO(), format=filetype)
    return Image.open(imgbytes)

def remove_alpha(image, bg):
    # https://stackoverflow.com/a/62414364
    background = Image.new('RGBA', image.size, bg)
    alpha_composite = Image.alpha_composite(background, image.convert("RGBA"))
    return alpha_composite.convert("RGB")

# https://stackoverflow.com/a/10616717
def trim(image, bg="white"):
    bg = Image.new(image.mode, image.size, bg)
    diff = ImageChops.difference(image, bg)
    #diff = ImageChops.add(diff, diff, 2.0, -10)
    return image.crop(bbox) if (bbox:=diff.getbbox()) else image

def expand(image, size, do_trim=False, bg="white"):
    """Expand image to a square of size {size}. Optionally trims borders first."""
    image = trim(image, bg=bg) if do_trim else image
    return ImageOps.pad(image, (size, size), color=bg, method=Image.Resampling.LANCZOS)

#  based on transformers/image_utils.py (added support for rgba images)
def load(image: Image.Image | str | bytes, bg="white", timeout=None):
    if isinstance(image, bytes):
        # assume image bytes and open
        image = Image.open(BytesIO(image))
    elif isinstance(image, str):
        if is_remote_url(image):
            # https://stackoverflow.com/a/69791396
            headers = {'user-agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:68.0) Gecko/20100101 Firefox/68.0'}
            image = Image.open(BytesIO(requests.get(image, timeout=timeout, headers=headers).content))
        elif isfile(image):
            image = Image.open(image)
        else:
            try:
                image.removeprefix("data:image/")
                image = Image.open(BytesIO(b64decode(image)))
            except Exception as e:
                raise ValueError(
                    "Incorrect image source. "
                    "Must be a valid URL starting with `http://` or `https://`, "
                    "a valid path to an image file, bytes, or a base64 encoded "
                    f"string. Got {image}. Failed with {e}"
                )

    image = ImageOps.exif_transpose(image) # type: ignore
    return  remove_alpha(image, bg=bg)
