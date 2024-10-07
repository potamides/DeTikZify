from functools import partial
import os
from typing import Dict

from PIL import Image
import torch
from transformers import Trainer, TrainingArguments, is_torch_xla_available
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import logging

from ...model.adapter.modeling_adapter import CrossAttentionAdapterMixin
from ...util import SplitEpochSaveCallback

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm # type: ignore

logger = logging.get_logger("transformers")

IGNORE_INDEX = -100
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))


# https://huggingface.co/docs/transformers/main/en/tasks/knowledge_distillation_for_image_classification
class AdapterTrainer(Trainer):
    def __init__(self, model: CrossAttentionAdapterMixin, *args, **kwargs):
        super().__init__(self.prepare_model(model), *args, **kwargs) # type: ignore
        self.loss_function = torch.nn.MSELoss()
        self.loss_layers = sorted({len(self.model.adapter.layers)} | {
            idx for idx, layer in enumerate(self.model.adapter.layers, 1) if layer is not None
        })
        self.loss_weights = [
            w / sum(range(1, len(self.loss_layers) + 1))
            for w in range(1, len(self.loss_layers) + 1)
        ]
        self.control.layer_losses = {layer: 0 for layer in self.loss_layers}

    def prepare_model(self, model):
        for name, param in model.named_parameters():
            if not "adapter" in name:
                param.requires_grad = False
            elif model.dtype != torch.float32:
                param.data = param.data.to(torch.float32)

        model.embedding_model.enable_input_require_grads()
        model.enable_input_require_grads()

        return model

    def compute_loss(self, model, inputs, return_outputs=False, **_):
        with torch.no_grad():
            teacher_output = model(pixel_values=inputs.pop("labels"), output_hidden_states=True)
        student_output = model(**inputs, output_hidden_states=True)

        loss = 0
        for layer, weight in zip(self.loss_layers, self.loss_weights):
            layer_loss = self.loss_function(
                student_output.hidden_states[layer],
                teacher_output.hidden_states[layer]
            )
            loss += weight * layer_loss

            log_layer_loss = layer_loss.mean() if self.args.n_gpu > 1 else layer_loss
            log_layer_loss = log_layer_loss.detach() / self.args.gradient_accumulation_steps
            self.control.layer_losses[layer] += log_layer_loss

        return (loss, student_output) if return_outputs else loss

    # https://github.com/naba89/custom_hf_trainer
    def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval):
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
            for k, v in self.control.layer_losses.items():
                layer_loss = self._nested_gather(v).mean().item() # type: ignore
                logs[f"layer_loss_{k}"] = round(layer_loss / (self.state.global_step - self._globalstep_last_logged), 4)
                self.control.layer_losses[k] -= v # reset the loss

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self._evaluate(trial, ignore_keys_for_eval)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

def transform_images(batch, processor, bg="white"):
    label_ids = processor(images=batch['image'], return_tensors="pt")
    input_ids = processor(
        images=[Image.new('RGB', img.size, color=bg) for img in batch['image']],
        text=batch['text'],
        return_tensors="pt",
        text_kwargs=dict(
            padding=True,
            truncation=True,
        )
    )

    input_ids['labels'] = label_ids['pixel_values']
    return input_ids

def train(
    output_dir: str,
    model,
    processor,
    dataset,
    overwrite=False,
    deepspeed=None,
    # training hyperparams
    batch_size: int = 256,
    micro_batch_size: int = 8,
    num_epochs: int = 10,
    learning_rate: float = 1e-4,
    gradient_checkpointing: bool = False,
):
    dataset.set_transform(partial(transform_images, processor=processor))
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
        callbacks=[SplitEpochSaveCallback(step_size=0.25)],
        args=TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_checkpointing=gradient_checkpointing,
            # https://github.com/huggingface/transformers/issues/21381
            gradient_checkpointing_kwargs={'use_reentrant': False},
            warmup_steps=500,
            weight_decay=0.1,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            torch_compile=True,
            bf16=True,
            tf32=True,
            logging_steps=250,
            logging_first_step=True,
            lr_scheduler_type="constant_with_warmup",
            optim="adamw_torch" if deepspeed else "adamw_torch_fused",
            ddp_find_unused_parameters=False,
            remove_unused_columns=False,
            save_strategy="epoch",
            report_to="none",
            output_dir=output_dir,
            deepspeed=deepspeed,
        )
    )

    trainer.train(resume_from_checkpoint=last_checkpoint)

    if trainer.is_deepspeed_enabled:
        # https://huggingface.co/docs/accelerate/v0.11.0/en/deepspeed#saving-and-loading
        from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
        last_checkpoint = get_last_checkpoint(output_dir)
        load_state_dict_from_zero_checkpoint(trainer.model.float(), last_checkpoint)

    model.save_cross_attn_adapter(output_dir)
    trainer.save_state()

    return model, processor
