from copy import deepcopy
from datetime import timedelta
import os
from typing import Dict, List

from accelerate import Accelerator, InitProcessGroupKwargs
from datasets import Dataset
import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import logging

from ...util import SplitEpochSaveCallback, unwrap_processor

logger = logging.get_logger("transformers")

IGNORE_INDEX = -100
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
RANK = int(os.environ.get("RANK", 0))

def tokenize(
    batch,
    processor,
    caption_condition=False,
    **kwargs
):
    unwrapped_processor = unwrap_processor(processor)
    image_token = unwrapped_processor.image_token
    image_token_id = unwrapped_processor.tokenizer.convert_tokens_to_ids(image_token)
    bos_token = unwrapped_processor.tokenizer.bos_token

    input_ids = processor(
        text=batch['caption'],
        images_kwargs=dict(
            text=[bos_token.join(text) for text in zip(batch['caption'], batch['code'])] if caption_condition else batch['code'],
            max_length=unwrapped_processor.tokenizer.model_max_length,
            pad_to_multiple_of=8,
            add_eos_token=True,
            truncation=False,
            padding=True
        ),
        text_kwargs=dict(
            padding=True,
            truncation=True,
        ),
        **kwargs
    )
    input_ids['labels'] = deepcopy(input_ids['input_ids'])

    if caption_condition:
        # do not train on caption and pad tokens
        for label_ids in input_ids['labels']:
            after_bos_token = False
            for idx, label_id in enumerate(label_ids):
                if not after_bos_token or label_id in {image_token_id, unwrapped_processor.tokenizer.pad_token_id}:
                    if label_id == unwrapped_processor.tokenizer.bos_token_id:
                        after_bos_token = True
                    label_ids[idx] = IGNORE_INDEX
                elif label_id == unwrapped_processor.tokenizer.bos_token_id:
                    after_bos_token = True
    else:
        # do not train on image and pad tokens
        for label_ids in input_ids['labels']:
            for idx, label_id in enumerate(label_ids):
                if label_id in {image_token_id, processor.tokenizer.pad_token_id}:
                    label_ids[idx] = IGNORE_INDEX

    return input_ids

class AdapterDataset(Dataset):
    def __init__(self, dataset, processor, caption_condition=False):
        super().__init__()
        self.processor = processor
        self.dataset = dataset.with_transform(self.tokenize)
        self.caption_condition = caption_condition

    def __len__(self):
        return len(self.dataset)

    def tokenize(self, batch):
        return tokenize(
            batch=batch,
            processor=self.processor,
            caption_condition=self.caption_condition,
            return_tensors="pt",
        )

    def filter(self, *args, **kwargs):
        self.dataset = self.dataset.filter(*args, **kwargs)

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        return self.dataset[index]

    def __getitems__(self, indices) -> Dict[str, List[torch.Tensor]]:
        return self.dataset[*indices]

def train(
    output_dir: str,
    model,
    processor,
    dataset,
    overwrite=False,
    deepspeed=None,
    # training hyperparams
    caption_condition: bool = False,
    batch_size: int = 128,
    micro_batch_size: int = 1,
    num_epochs: int = 5,
    learning_rate: float = 5e-5,
    gradient_checkpointing: bool = False,
):
    gradient_accumulation_steps = batch_size // micro_batch_size
    if WORLD_SIZE != 1:
        gradient_accumulation_steps = gradient_accumulation_steps // WORLD_SIZE

    for _, param in model.model.vision_model.named_parameters():
        param.requires_grad = False
    for _, param in model.adapter.named_parameters():
        param.requires_grad = False
    for _, param in model.embedding_model.named_parameters():
        param.requires_grad = False
    model.enable_input_require_grads()
    model.embedding_model.enable_input_require_grads()

    dataset = AdapterDataset(dataset, processor, caption_condition=caption_condition)
    logger.info(f"Dataset size before filtering out too long examples: {len(dataset)}")
    eos_token_id, model_max_length = unwrap_processor(processor).tokenizer.eos_token_id, unwrap_processor(processor).tokenizer.model_max_length
    with Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(days=3))]).main_process_first():
        dataset.filter(lambda ex: (ex['input_ids'] == eos_token_id).nonzero() < model_max_length, num_proc=64, batch_size=16)
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
            # https://github.com/huggingface/transformers/issues/32576
            #gradient_checkpointing_kwargs={'use_reentrant':False},
            dataloader_num_workers=WORLD_SIZE,
            warmup_ratio=0.03,
            weight_decay=0,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            torch_compile=True,
            bf16=True,
            tf32=True,
            logging_steps=10,
            lr_scheduler_type="cosine",
            optim="adamw_torch" if deepspeed else "adamw_torch_fused",
            ddp_find_unused_parameters=False,
            remove_unused_columns=False,
            save_strategy="epoch",
            report_to="none",
            save_total_limit=1,
            output_dir=output_dir,
            deepspeed=deepspeed,
        ),
        callbacks=[SplitEpochSaveCallback(step_size=0.25)],
        data_collator=lambda batch: batch
    )

    trainer.add_callback(trainer.train_dataset)
    trainer.train(resume_from_checkpoint=last_checkpoint)

    if trainer.is_deepspeed_enabled:
        # https://huggingface.co/docs/accelerate/v0.11.0/en/deepspeed#saving-and-loading
        from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
        last_checkpoint = get_last_checkpoint(output_dir)
        load_state_dict_from_zero_checkpoint(trainer.model.float(), last_checkpoint)

    trainer.model.unload_cross_attn_adapter()
    trainer.save_model(output_dir)
    trainer.save_state()
    processor.processor.save_pretrained(output_dir)

    return model, processor
