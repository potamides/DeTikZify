#!/usr/bin/env python
from argparse import ArgumentParser
from sys import flags

from PIL import UnidentifiedImageError
from peft.utils import infer_device # type: ignore
from transformers import set_seed

from detikzify.infer import DetikzifyPipeline
from detikzify.model import load

try:
    import readline # patches input()
except:
    pass

def parse_args():
    argument_parser = ArgumentParser(
        description="Inference helper for fine-tuned models."
    )
    argument_parser.add_argument(
        "--model_name_or_path",
        required=True,
        help="the model checkpoint for weights initialization (local or hub)",
    )
    return argument_parser.parse_args()

if __name__ == "__main__":
    set_seed(0)
    model, tokenizer = load(parse_args().model_name_or_path)
    pipe = DetikzifyPipeline(model.to(infer_device()), tokenizer, stream=True) # type: ignore

    if flags.interactive:
        print("generate(*args, **kwargs):", str(DetikzifyPipeline.generate.__doc__).strip())
    else:
        print("Specify the path to an image (locally or as URL) to detikzify it!")
        while True:
            try:
                image = input("Image: ")
            except (KeyboardInterrupt, EOFError):
                break
            try:
                pipe(image=image)
            except KeyboardInterrupt:
                pass
            except (UnidentifiedImageError, FileNotFoundError, AttributeError):
                print("Error: Cannot identify image file!")
