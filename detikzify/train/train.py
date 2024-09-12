from io import BytesIO
import os
from typing import Dict, List

from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import (
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import logging

from ..util import SplitEpochSaveCallback
from .pretrain import tokenize

logger = logging.get_logger("transformers")

WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
RANK = int(os.environ.get("RANK", 0))

class ImageSketchDataset(Dataset, TrainerCallback):
    """
    Dataset which samples sketches instead of images, when a sketch exists
    for the current epoch.
    """
    def __init__(self, dataset, processor):
        super().__init__()
        self.processor = processor
        self.dataset = dataset.with_transform(self.tokenize)
        self.cur_epoch = 0

    def __len__(self):
        return len(self.dataset)

    def tokenize(self, batch):
        for idx, sketches in enumerate(batch['sketches']):
            if (sketch:=sketches[self.cur_epoch]):
                batch[idx]['image'] = Image.open(BytesIO(sketch['bytes']))

        return tokenize(
            batch=batch,
            processor=self.processor,
            return_tensors="pt",
            truncation=False,
            padding=True
        )

    def filter(self, *args, **kwargs):
        self.dataset = self.dataset.filter(*args, **kwargs)

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        return self.dataset[index]

    def __getitems__(self, indices) -> Dict[str, List[torch.Tensor]]:
        return self.dataset[*indices]

    def on_epoch_end(self, *args, **kwargs):
        self.cur_epoch += 1

def train(
    output_dir: str,
    model,
    processor,
    dataset,
    overwrite=False,
    deepspeed=None,
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 1,
    num_epochs: int = 3,
    learning_rate: float = 3e-5,
    gradient_checkpointing: bool = False,
):
    assert num_epochs == len(dataset[0]['sketches'])
    gradient_accumulation_steps = batch_size // micro_batch_size
    if WORLD_SIZE != 1:
        gradient_accumulation_steps = gradient_accumulation_steps // WORLD_SIZE

    dataset = ImageSketchDataset(dataset, processor)
    logger.info(f"Dataset size before filtering out too long examples: {len(dataset)}")
    dataset.filter(lambda ex: len(ex['input_ids'].squeeze()) <= processor.tokenizer.model_max_length)
    logger.info(f"Dataset size after filtering out too long examples: {len(dataset)}")

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
        train_dataset=dataset,
        args=TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_checkpointing=gradient_checkpointing,
            warmup_ratio=0.03,
            weight_decay=0,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            torch_compile=True,
            bf16=True,
            logging_steps=10,
            lr_scheduler_type="cosine",
            optim="adamw_torch" if deepspeed else "adamw_torch_fused",
            remove_unused_columns=False,
            save_strategy="epoch",
            report_to="none",
            save_total_limit=1,
            output_dir=output_dir,
            deepspeed=deepspeed,
        ),
        callbacks=[SplitEpochSaveCallback(step_size=0.25)]
    )

    model.config.use_cache = False
    trainer.add_callback(trainer.train_dataset)
    trainer.train(resume_from_checkpoint=last_checkpoint)

    if deepspeed:
        from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
        last_checkpoint = get_last_checkpoint(output_dir)
        load_state_dict_from_zero_checkpoint(trainer.model.float(), last_checkpoint)

    trainer.save_model(output_dir)
    trainer.save_state()

    return model, processor
