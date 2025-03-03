import os
from typing import Dict, Literal

import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from transformers import (
    Trainer,
    TrainerCallback,
    TrainingArguments,
    is_torch_xla_available,
)
from transformers.trainer_utils import SaveStrategy, get_last_checkpoint
from transformers.utils import logging

from ...model.adapter.modeling_adapter import CrossAttentionAdapterMixin
from ...util import (
    EditCutMix,
    EditCutOut,
    EditMixUp,
    FullErase,
    SketchAugment,
    SplitEpochSaveCallback,
    unwrap_processor,
)

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm # type: ignore

logger = logging.get_logger("transformers")

WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))

class EmbeddingSimilarityLoss():
    def __init__(self, elementwise: bool = True, use_mse: bool = False):
        self.cosine = torch.nn.CosineSimilarity(dim=-1)
        self.mae = torch.nn.L1Loss(reduction="none")
        self.mse = torch.nn.MSELoss(reduction="none")
        self.elementwise = elementwise
        self.use_mse = use_mse

    # https://github.com/pytorch/pytorch/issues/104564#issuecomment-1651575112
    @torch.compile
    def cosine_loss(self, x, y):
        if self.elementwise:
            return self.mae(cos:=self.cosine(x, y), torch.ones_like(cos)).mean()
        else:
            X = self.cosine(x.unsqueeze(2), y.unsqueeze(1))
            Y = self.cosine(y.unsqueeze(2), y.unsqueeze(1))
            return self.mae(X, Y).max(dim=-1)[0].mean()

    @torch.compile
    def l2_loss(self, x, y):
        if self.elementwise:
            return self.mse(x, y).mean()
        else:
            X, Y = torch.cdist(x, y), torch.cdist(y, y)
            return self.mae(X, Y).max(dim=-1)[0].mean()

    def __call__(self, x, y):
        if self.use_mse:
            return self.l2_loss(x, y)
        else:
            return self.cosine_loss(x, y)

# https://huggingface.co/docs/transformers/main/en/tasks/knowledge_distillation_for_image_classification
class AdapterTrainer(Trainer):
    def __init__(
        self,
        model: CrossAttentionAdapterMixin,
        loss_term: Literal["avg", "pool", "patch", "layer"] = "patch",
        elementwise_loss: bool = True,
        mse_loss: bool = False,
        pool_train_head: bool = False,
        multimodal: bool = False,
        *args,
        **kwargs,
    ):
        self.term = loss_term
        self.loss_function = EmbeddingSimilarityLoss(elementwise=elementwise_loss, use_mse=mse_loss)
        train_head = self.term == "pool" and pool_train_head
        super().__init__(self.prepare_model(model, train_head, multimodal), *args, **kwargs) # type: ignore

        if self.term == "layer":
            self.loss_layers = sorted({len(self.model.adapter.layers)} | {
                idx for idx, layer in enumerate(self.model.adapter.layers, 1) if layer is not None
            })
            self.control.layer_losses = {layer: 0 for layer in self.loss_layers}

    def prepare_model(self, model, train_head=False, multimodal=False):
        for name, param in model.named_parameters():
            if not "adapter" in name and (not train_head or not "head" in name):
                param.requires_grad = False
            elif multimodal and "dummy_input" in name:
                param.requires_grad = False
            elif model.dtype != torch.float32:
                param.data = param.data.to(torch.float32)

        if train_head: # in this case we also want gradients for the teacher
            model.vision_model.head.forward = torch.enable_grad(model.vision_model.head.forward)
        if self.term != "pool":
            model.vision_model.use_head = False

        model.embedding_model.enable_input_require_grads()
        model.enable_input_require_grads()

        return model

    def compute_loss(self, model, inputs, return_outputs=False, **_):
        with torch.no_grad():
            teacher_output = model(
                pixel_values=inputs.pop("labels"),
                output_hidden_states=self.term=="layer"
            )
        student_output = model(
            output_hidden_states=self.term=="layer",
            **inputs,
        )

        if self.term == "avg":
            loss = self.loss_function(
                student_output.last_hidden_state.mean(dim=1),
                teacher_output.last_hidden_state.mean(dim=1),
            )
        elif self.term == "pool":
            loss = self.loss_function(
                student_output.pooler_output,
                teacher_output.pooler_output,
            )
        elif self.term == "patch":
            loss = self.loss_function(
                student_output.last_hidden_state,
                teacher_output.last_hidden_state
            )
        else:
            loss = 0
            for layer in self.loss_layers:
                last_layer = layer == self.loss_layers[-1]
                layer_loss = self.loss_function(
                    student_output.last_hidden_state if last_layer else student_output.hidden_states[layer],
                    teacher_output.last_hidden_state if last_layer else teacher_output.hidden_states[layer]
                )
                loss += .5 * (1 if last_layer else 1/(len(self.loss_layers)-1)) * layer_loss

                log_layer_loss = layer_loss.mean() if self.args.n_gpu > 1 else layer_loss
                log_layer_loss = log_layer_loss.detach() / self.args.gradient_accumulation_steps
                self.control.layer_losses[layer] += log_layer_loss

        return (loss, student_output) if return_outputs else loss

    # https://github.com/naba89/custom_hf_trainer
    def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time):
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            if is_torch_xla_available():
                xm.mark_step() # type: ignore

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item() # type: ignore

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            if grad_norm is not None:
                logs["grad_norm"] = grad_norm.detach().item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            logs["learning_rate"] = self._get_learning_rate()
            if self.term == "layer":
                for k, v in self.control.layer_losses.items():
                    layer_loss = self._nested_gather(v).mean().item() # type: ignore
                    logs[f"layer_loss_{k}"] = round(layer_loss / (self.state.global_step - self._globalstep_last_logged), 4)
                    self.control.layer_losses[k] -= v # reset the loss

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs, start_time)

        metrics = None
        if self.control.should_evaluate:
            metrics = self._evaluate(trial, ignore_keys_for_eval)
            is_new_best_metric = self._determine_best_metric(metrics=metrics, trial=trial)

            if self.args.save_strategy == SaveStrategy.BEST:
                self.control.should_save = is_new_best_metric

        if self.control.should_save:
            self._save_checkpoint(model, trial)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

class AdapterDataset(Dataset, TrainerCallback):
    def __init__(self, dataset, processor, multimodal=False):
        super().__init__()
        self.processor = processor
        self.dataset = dataset
        self.multimodal = multimodal
        self.offset = 1

        self.sketchify = SketchAugment(intensity=2)
        self.mixup = EditMixUp()
        self.cutmix = EditCutMix()
        self.cutout = EditCutOut()
        self.erase = FullErase()

    def __len__(self):
        return len(self.dataset)

    def __getitems__(self, indices) -> Dict[str, torch.Tensor]:
        batch, images = self.dataset[indices], None
        labels = torch.stack([v2.functional.pil_to_tensor(img) for img in batch['image']])

        if self.multimodal:
            partition, images = torch.randint(3, (len(indices),)), torch.empty_like(labels)
            if len(sketch_ids:=torch.argwhere(partition == 0).flatten()):
                images[sketch_ids] = self.sketchify(labels[sketch_ids])
            if len(blank_ids:=torch.argwhere(partition == 1).flatten()):
                images[blank_ids] = self.erase(labels[blank_ids])
            if len(edit_ids:=torch.argwhere(partition == 2).flatten()):
                edit_partition = torch.randint(3, (len(edit_ids),))
                if len(cutout_ids:=edit_ids[torch.argwhere(edit_partition == 0)].flatten()):
                    images[cutout_ids] = self.cutout(labels[cutout_ids])
                if len(mixup_ids:=edit_ids[torch.argwhere(edit_partition == 1)].flatten()):
                    mixup_imgs = self.dataset[[(indices[idx] + self.offset) % len(self) for idx in mixup_ids]]['image']
                    mixup_imgs = torch.stack([v2.functional.pil_to_tensor(img) for img in mixup_imgs])
                    interleaved_imgs = torch.stack([labels[mixup_ids], mixup_imgs], dim=1).view(-1, *mixup_imgs.shape[1:])
                    images[mixup_ids] = self.mixup(interleaved_imgs)[::2]
                if len(cutmix_ids:=edit_ids[torch.argwhere(edit_partition == 2)].flatten()):
                    cutmix_imgs = self.dataset[[(indices[idx] + self.offset) % len(self) for idx in cutmix_ids]]['image']
                    cutmix_imgs = torch.stack([v2.functional.pil_to_tensor(img) for img in cutmix_imgs])
                    interleaved_imgs = torch.stack([labels[cutmix_ids], cutmix_imgs], dim=1).view(-1, *cutmix_imgs.shape[1:])
                    images[cutmix_ids] = self.cutmix(interleaved_imgs)[::2]

        input_ids = self.processor(
            images=images,
            text=batch['text'],
            return_tensors="pt",
            text_kwargs=dict(
                padding=True,
                truncation=True,
            )
        )
        label_ids = unwrap_processor(self.processor)(images=labels, return_tensors="pt")
        input_ids['labels'] = label_ids['pixel_values']

        return input_ids

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        return self.__getitems__([index] if isinstance(index, int) else index)

    def on_epoch_end(self, *args, **kwargs):
        self.offset += 1

def train(
    output_dir: str,
    model,
    processor,
    dataset,
    overwrite=False,
    deepspeed=None,
    # training hyperparams
    multimodal: bool = False,
    batch_size: int = 512,
    micro_batch_size: int = 8,
    num_epochs: int = 3,
    learning_rate: float = 1e-4,
    gradient_checkpointing: bool = False,
    **loss_kwargs
):
    dataset = AdapterDataset(dataset, processor=processor, multimodal=multimodal)
    gradient_accumulation_steps = batch_size // micro_batch_size

    if WORLD_SIZE != 1:
        gradient_accumulation_steps = gradient_accumulation_steps // WORLD_SIZE

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

    trainer = AdapterTrainer(
        model=model,
        train_dataset=dataset,
        multimodal=multimodal,
        callbacks=[SplitEpochSaveCallback(step_size=0.5)],
        data_collator=lambda batch: batch,
        **loss_kwargs,
        args=TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_checkpointing=gradient_checkpointing,
            # https://github.com/huggingface/transformers/issues/21381
            gradient_checkpointing_kwargs={'use_reentrant': False},
            dataloader_num_workers=WORLD_SIZE,
            warmup_steps=500,
            weight_decay=0.1,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            torch_compile=True,
            bf16=True,
            tf32=True,
            logging_steps=250,
            logging_first_step=True,
            lr_scheduler_type="cosine",
            optim="adamw_torch" if deepspeed else "adamw_torch_fused",
            ddp_find_unused_parameters=False,
            remove_unused_columns=False,
            save_strategy="epoch",
            report_to="none",
            output_dir=output_dir,
            deepspeed=deepspeed,
        )
    )

    trainer.add_callback(trainer.train_dataset)
    trainer.train(resume_from_checkpoint=last_checkpoint)

    if trainer.is_deepspeed_enabled:
        # https://huggingface.co/docs/accelerate/v0.11.0/en/deepspeed#saving-and-loading
        from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
        last_checkpoint = get_last_checkpoint(output_dir)
        load_state_dict_from_zero_checkpoint(trainer.model.float(), last_checkpoint)

    model.save_cross_attn_adapter(output_dir)
    trainer.save_state()

    return model, processor
