#!/usr/bin/env -S torchrun --nproc_per_node gpu

from argparse import ArgumentParser
from functools import cached_property
from itertools import chain
from math import ceil, floor
from os import environ
from random import choice, gauss, random, sample

from datasets import load_dataset
from diffusers import DiffusionPipeline
import torch

from detikzify.util import convert

# performance optimizations: https://huggingface.co/blog/sd3
torch.set_float32_matmul_precision("high")
torch._inductor.config.conv_1x1_as_mm = True
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.epilogue_fusion = False
torch._inductor.config.coordinate_descent_check_all_directions = True

WORLD_SIZE = int(environ.get("WORLD_SIZE", 1))
RANK = int(environ.get("RANK", 0))

class Sketchifier:
    def __init__(
        self,
        model="nllg/sketch-pix2pix",
        device=torch.device("cuda", RANK),
        grayscale_ratio=0.1,
    ):
        self.model = model
        self.device = torch.device(device)
        self.grayscale_ratio = grayscale_ratio

    @cached_property
    def pipe(self):
        pipe = DiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path="nllg/ultrasketch",
            custom_pipeline="nllg/ultrasketch",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        pipe.set_progress_bar_config(disable=True)

        # speed up inference
        pipe.to(self.device)
        pipe.transformer.to(memory_format=torch.channels_last)
        pipe.vae.to(memory_format=torch.channels_last)

        pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune", fullgraph=True)
        pipe.vae.decode = torch.compile(pipe.vae.decode, mode="max-autotune", fullgraph=True)

        return pipe

    def __call__(self, *args, **kwargs):
        return self.sketchify(*args, **kwargs)

    def sketchify(self, image):
        with torch.inference_mode(), torch.autocast(self.device.type, enabled=False): # type: ignore
            sketch = self.pipe(
                prompt="Turn it into a hand-drawn sketch",
                image=image,
                mask_img=Image.new("RGB", image.size, "white"),
                num_inference_steps=50,
                image_guidance_scale=1.7,
                guidance_scale=1.5,
                strength=max(.85, min(.95, gauss(.9, .5)))
            ).images[0]
            sketch = sketch if random() > self.grayscale_ratio else sketch.convert("L")
            return convert(sketch, "png")

def sketchify(dataset, num_epochs, ratio, sketchifier):
    """
    Randomly sketchify <ratio> of all examples in <dataset> for each epoch
    given with <num_epochs>.
    """
    # prepare the sketches (distribute load among all workers)
    worker_sketches, all_sketches = list(), WORLD_SIZE * [None]
    for i in torch.arange(len(dataset['image'])).tensor_split(WORLD_SIZE)[RANK]:
        # randomize in which epochs how many images should be sketchified
        num_sketches = choice([floor(product:=ratio*num_epochs), ceil(product)])
        sketch_epochs = sample(range(num_epochs), k=num_sketches)
        # generate the sketches
        sketches = [sketchifier(dataset['image'][i.item()]) for _ in range(num_sketches)]
        worker_sketches.append([sketches.pop() if epoch in sketch_epochs else None for epoch in range(num_epochs)])

    torch.distributed.all_gather_object(all_sketches, worker_sketches) # type: ignore
    dataset['sketches'] = list(chain.from_iterable(all_sketches)) # type: ignore
    return dataset

def parse_args():
    argument_parser = ArgumentParser(
        description="Sketchify an existing DaTikZ dataset."
    )
    argument_parser.add_argument(
        "--path",
        default="nllg/datikz-v3",
        help="Path or name of the DaTikZ dataset.",

    )
    return argument_parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    datikz = load_dataset(args.path)
    torch.distributed.init_process_group()

    train = datikz['train'].map(
        function=sketchify,
        batched=True,
        batch_size=WORLD_SIZE * 1000,
        desc="Sketchify (train)",
        fn_kwargs=dict(
            num_epochs=3,
            ratio=0.5,
            sketchifier=(sketchifier:=Sketchifier())
        )
    )

     # test split is small so keep it simple and do it only on the main process
    if RANK == 0:
        test = datikz['test'].map(
            function=lambda ex: ex | {"sketch": sketchifier(ex['image'])},
            desc="Sketchify (test)"
        )

        train.to_parquet("datikz-train-sketches.parquet", compression="GZIP")
        test.to_parquet("datikz-test-sketches.parquet", compression="GZIP")
