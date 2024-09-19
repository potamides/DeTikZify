import copy
from functools import partial
import os
from typing import List

from transformers import Trainer, TrainingArguments

IGNORE_INDEX = -100
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))

def tokenize(
    batch,
    processor,
    **kwargs
):
    image_token = processor.image_token
    image_token_id = processor.tokenizer.convert_tokens_to_ids(image_token)

    input_ids = processor(
        text=batch['text'],
        images=batch['image'],
        max_length=processor.tokenizer.model_max_length,
        pad_to_multiple_of=8,
        add_eos_token=True,
        **kwargs
    )
    input_ids['labels'] = copy.deepcopy(input_ids['input_ids'])

    # do not train on image and pad tokens
    for label_ids in input_ids['labels']:
        for idx, label_id in enumerate(label_ids):
            if label_id in {image_token_id, processor.tokenizer.pad_token_id}:
                label_ids[idx] = IGNORE_INDEX

    return input_ids


def train(
    output_dir: str,
    model,
    processor,
    dataset,
    deepspeed=None,
    # training hyperparams
    batch_size: int = 256,
    micro_batch_size: int = 1,
    num_epochs: int = 1,
    learning_rate: float = 1e-3,
    gradient_checkpointing: bool = False,
    full_finetune_modules: List[str] = [
        "modality_projection",
    ],
):
    gradient_accumulation_steps = batch_size // micro_batch_size

    if WORLD_SIZE != 1:
        gradient_accumulation_steps = gradient_accumulation_steps // WORLD_SIZE
    for name, param in model.named_parameters():
        if not any(module in name for module in full_finetune_modules):
            param.requires_grad = False

    dataset.set_transform(partial(
        tokenize,
        processor=processor,
        return_tensors="pt",
        truncation=True,
        padding=True
    ))

    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        args=TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_checkpointing=gradient_checkpointing,
            # https://github.com/huggingface/transformers/issues/21381
            gradient_checkpointing_kwargs={'use_reentrant':False},
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
            save_strategy="no",
            report_to="none",
            output_dir=output_dir,
            deepspeed=deepspeed,
        )
    )

    if trainer.is_deepspeed_enabled and trainer.accelerator.state.deepspeed_plugin.hf_ds_config.is_zero3():
        raise ValueError("Pretraining with zero stage 3 is not yet supported.")

    trainer.train()

    model.save_pretrained(
        output_dir,
        state_dict={
            name: weight
            for name, weight in model.state_dict().items()
            if any(key_match in name for key_match in full_finetune_modules)
        },
    )
    trainer.save_state()

    return model, processor
