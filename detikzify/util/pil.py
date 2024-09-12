from io import BytesIO

from PIL import Image, ImageChops, ImageOps
from transformers.image_utils import load_image

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

def load(image: Image.Image | str, bg="white"):
    return  remove_alpha(load_image(image), bg=bg)
