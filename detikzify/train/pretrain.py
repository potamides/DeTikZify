import copy
from dataclasses import dataclass
import os
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments

IGNORE_INDEX = -100

def preprocess(
    texts: str | List[str],
    tokenizer,
    patch_token,
    num_patches,
    truncation=True,
    return_tensors=None,
):
    patch_token_id = tokenizer.convert_tokens_to_ids(patch_token)
    patch_tokens = num_patches * patch_token
    input_ids = tokenizer(
        patch_tokens + texts if isinstance(texts, str) else [patch_tokens + text for text in texts],
        max_length=tokenizer.model_max_length,
        truncation=truncation,
        return_tensors=return_tensors,
    )
    input_ids['labels'] = copy.deepcopy(input_ids['input_ids'])

    # do not train on image patch tokens
    for label_ids in input_ids['labels']:
        for idx, label_id in enumerate(label_ids):
            if label_id == patch_token_id != label_ids[idx + 1]:
                label_ids[idx] = IGNORE_INDEX
                break
            label_ids[idx] = IGNORE_INDEX

    return input_ids


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, dataset, tokenizer, num_patches, patch_token):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.num_patches = num_patches
        self.patch_token = patch_token

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        item = self.dataset[i]
        data_dict = preprocess(
            texts=item['text'],
            tokenizer=self.tokenizer.text,
            patch_token=self.patch_token,
            num_patches=self.num_patches,
            return_tensors="pt",
        )
        return dict(
            input_ids=data_dict["input_ids"][0],
            labels=data_dict["labels"][0],
            images=self.tokenizer.image(item['image'])
        )


@dataclass
class DataCollatorForImageTextTraining(DataCollatorWithPadding):
    """Collate examples for supervised image-text fine-tuning."""

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        images = [feat.pop("images") for feat in features]
        labels = super().__call__([{"input_ids": feat.pop("labels")} for feat in features])['input_ids']
        labels[labels == self.tokenizer.pad_token_id] = IGNORE_INDEX

        batch = super().__call__(features)
        batch['images'] = torch.stack(images)
        batch['labels'] = labels

        return batch


def train(
    output_dir: str,
    model,
    tokenizer,
    dataset,
    # training hyperparams
    batch_size: int = 256,
    micro_batch_size: int = 1,
    num_epochs: int = 1,
    learning_rate: float = 1e-3,
    gradient_checkpointing = False,
    group_by_length: bool = False,
    full_finetune_modules: List[str] = [
        "mm_projector",
    ],
):

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    gradient_accumulation_steps = batch_size // micro_batch_size
    if ddp := world_size != 1:
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    dataset = LazySupervisedDataset(
        tokenizer=tokenizer,
        dataset=dataset,
        num_patches=model.config.num_patches,
        patch_token=tokenizer.text.convert_ids_to_tokens(model.config.patch_token_id)
    )

    for name, param in model.named_parameters():
        if not any(module in name for module in full_finetune_modules):
            param.requires_grad = False

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
            bf16=True,
            logging_steps=10,
            lr_scheduler_type="cosine",
            optim="adamw_torch_fused",
            save_strategy="no",
            output_dir=output_dir,
            ddp_find_unused_parameters=False if ddp else None,
            remove_unused_columns=False,
            group_by_length=group_by_length,
        ),
        data_collator=DataCollatorForImageTextTraining(
            tokenizer=tokenizer.text,
            pad_to_multiple_of=8
        )
    )

    model.config.use_cache = False
    trainer.train()

    model.save_pretrained(
        output_dir,
        state_dict={
            k.split(".", 1)[-1]: v
            for k, v in model.state_dict().items()
            if any(key_match in k for key_match in full_finetune_modules)
        },
    )
    trainer.save_state()

    return model, tokenizer
