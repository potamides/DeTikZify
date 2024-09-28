#!/usr/bin/env python
from argparse import ArgumentParser
from sys import flags

from PIL import UnidentifiedImageError
from torch import bfloat16, float16
from torch.cuda import is_available as is_cuda_available, is_bf16_supported
from transformers import TextStreamer, set_seed
from transformers.utils import is_flash_attn_2_available

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
    model, processor = load(
        **vars(parse_args()),
        device_map="auto",
        torch_dtype=bfloat16 if is_cuda_available() and is_bf16_supported() else float16,
        attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
    )
    pipe = DetikzifyPipeline(
        model=model,
        processor=processor,
        streamer=TextStreamer(
            tokenizer=processor.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )
    )

    if flags.interactive:
        print("pipe(*args, **kwargs):", str(DetikzifyPipeline.sample.__doc__).strip())
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
