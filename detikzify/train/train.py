from functools import cached_property
from io import BytesIO
from itertools import chain
from math import ceil, floor
import os
from random import choice, sample
from typing import Dict

from PIL import Image

from diffusers import (
    EulerAncestralDiscreteScheduler, # type: ignore
    StableDiffusionInstructPix2PixPipeline, # type: ignore
)
import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainerCallback, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import logging

from ..util import convert, infer_device, HalfEpochSaveCallback
from .pretrain import DataCollatorForImageTextTraining, preprocess

logger = logging.get_logger("transformers")

WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
RANK = int(os.environ.get("RANK", 0))

class Sketchifier:
    def __init__(self, model="nllg/sketch-pix2pix", device=torch.device(infer_device(), RANK)):
        self.model, self.device = model, torch.device(device)

    @cached_property
    def pipe(self):
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(self.model,
            torch_dtype=torch.float16,
            safety_checker=None
        )
        pipe.set_progress_bar_config(disable=True)

        # speed up inference
        pipe.to(self.device)
        pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.unet.to(memory_format=torch.channels_last)

        return pipe

    def __call__(self, *args, **kwargs):
        return self.sketchify(*args, **kwargs)

    def sketchify(self, imgs):
        with torch.inference_mode(), torch.autocast(self.device.type, enabled=False): # type: ignore
            return [convert(img, "png") for img in self.pipe(
                prompt=["turn it into a doodle"] * len(imgs),
                image=imgs,
                num_inference_steps=10,
                image_guidance_scale=1.2,
                guidance_scale=15,
            ).images]

class ImageSketchDataset(Dataset, TrainerCallback):
    """
    Dataset which samples sketches instead of images, when a sketch exists
    for the current epoch.
    """
    def __init__(self, dataset, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.cur_epoch = 0

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        item = self.dataset[i]

        if (sketch:=item['sketches'][self.cur_epoch]):
            image = Image.open(BytesIO(sketch['bytes']))
        else:
            image = item['image']
        return dict(
            input_ids=torch.tensor(item["input_ids"]),
            labels=torch.tensor(item["labels"]),
            images=self.tokenizer.image(image)
        )

    def on_epoch_end(self, *args, **kwargs):
        self.cur_epoch += 1

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
        sketches = sketchifier(num_sketches * [dataset['image'][i.item()]])
        worker_sketches.append([sketches.pop() if epoch in sketch_epochs else None for epoch in range(num_epochs)])

    torch.distributed.all_gather_object(all_sketches, worker_sketches) # type: ignore
    dataset['sketches'] = list(chain.from_iterable(all_sketches)) # type: ignore
    return dataset

def train(
    output_dir: str,
    model,
    tokenizer,
    dataset,
    overwrite=False,
    deepspeed=None,
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 1,
    num_epochs: int = 3,
    learning_rate: float = 4e-5,
    sketchification_ratio: float = 0.5,
    gradient_checkpointing: bool = False,
    group_by_length: bool = False,
):
    gradient_accumulation_steps = batch_size // micro_batch_size
    if ddp := WORLD_SIZE != 1:
        gradient_accumulation_steps = gradient_accumulation_steps // WORLD_SIZE

    def prepare_dataset(dataset):
        patch_token = tokenizer.text.convert_ids_to_tokens(model.config.patch_token_id)
        max_len = tokenizer.text.model_max_length
        dataset = dataset.map(
            sketchify,
            batched=True,
            desc="Sketchify",
            fn_kwargs=dict(
                num_epochs=num_epochs,
                ratio=sketchification_ratio,
                sketchifier=Sketchifier()
            )
        )
        dataset = dataset.map(
            lambda exs, **kwargs: preprocess(exs['text'], **kwargs) | {"image": exs['image']},
            batched=True,
            desc="Tokenize",
            fn_kwargs=dict(
                tokenizer=tokenizer.text,
                num_patches=model.config.num_patches,
                patch_token=patch_token,
                truncation=False
            )
        )
        logger.info(f"Dataset size before filtering out too long examples: {len(dataset)}")
        dataset = dataset.filter(lambda ex: len(ex['input_ids']) <= max_len and patch_token not in ex['text'])
        logger.info(f"Dataset size after filtering out too long examples: {len(dataset)}")
        return ImageSketchDataset(tokenizer=tokenizer, dataset=dataset)

    last_checkpoint = None
    if os.path.isdir(output_dir) and not overwrite:
        last_checkpoint = get_last_checkpoint(output_dir)
        if last_checkpoint is None and len(os.listdir(output_dir)) > 0:
            raise ValueError(
                f"Output directory ({output_dir}) already exists and is not empty. "
                "Use `overwrite` to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `output_dir` or add `overwrite` to train from scratch."
            )

    trainer = Trainer(
        model=model,
        train_dataset=prepare_dataset(dataset),
        args=TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_checkpointing=gradient_checkpointing,
            warmup_ratio=0.03,
            weight_decay=0,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            bf16=True,
            logging_steps=10,
            lr_scheduler_type="cosine",
            optim="adamw_torch" if deepspeed else "adamw_torch_fused",
            save_strategy="epoch",
            save_total_limit=1,
            output_dir=output_dir,
            ddp_find_unused_parameters=False if ddp else None,
            remove_unused_columns=False,
            group_by_length=group_by_length,
            deepspeed=deepspeed,
        ),
        callbacks=[HalfEpochSaveCallback()],
        data_collator=DataCollatorForImageTextTraining(
            tokenizer=tokenizer.text,
            pad_to_multiple_of=8
        )
    )

    model.config.use_cache = False
    trainer.add_callback(trainer.train_dataset)
    trainer.train(resume_from_checkpoint=last_checkpoint)

    if deepspeed:
        from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
        last_checkpoint = get_last_checkpoint(output_dir)
        load_state_dict_from_zero_checkpoint(trainer.model, last_checkpoint)

    trainer.save_model(output_dir)
    trainer.save_state()

    return model, tokenizer
