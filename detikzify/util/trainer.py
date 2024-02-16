from transformers import (
    IntervalStrategy,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from transformers.trainer_utils import has_length


class HalfEpochSaveCallback(TrainerCallback):
    """
    If save_strategy==EPOCH also save a checkpoint after completing 50% of an
    epoch (default).
    """

    def __init__(self, ratio: float = 0.5):
        self.ratio = ratio

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if has_length(train_dataloader:=kwargs['train_dataloader']):
            self.num_update_steps_per_epoch = max(len(train_dataloader) // args.gradient_accumulation_steps, 1)
        else:
            self.num_update_steps_per_epoch = args.max_steps

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if (
            state.global_step % self.num_update_steps_per_epoch == round(self.num_update_steps_per_epoch * self.ratio)
            and args.save_strategy == IntervalStrategy.EPOCH
        ):
            control.should_save = True

        return control
