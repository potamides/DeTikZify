from numpy import arange
from torchvision.transforms import v2
from transformers import (
    IntervalStrategy,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from transformers.trainer_utils import has_length

class SplitEpochSaveCallback(TrainerCallback):
    """
    If save_strategy==EPOCH also save checkpoints at arbitrary fractions of an
    epoch (controlled by step_size).
    """

    def __init__(self, step_size: float = 0.5):
        self.steps = arange(step_size, 1, step_size)

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if has_length(train_dataloader:=kwargs['train_dataloader']):
            self.num_update_steps_per_epoch = max(len(train_dataloader) // args.gradient_accumulation_steps, 1)
        else:
            self.num_update_steps_per_epoch = args.max_steps

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        steps = [round(self.num_update_steps_per_epoch * step) for step in self.steps]
        if (
            state.global_step % self.num_update_steps_per_epoch in steps
            and args.save_strategy == IntervalStrategy.EPOCH
        ):
            control.should_save = True

        return control

class SketchAugment(v2.Compose):
    def __init__(self):
        super().__init__([
            v2.RandomOrder([
                v2.ElasticTransform(fill=255),
                v2.JPEG((40, 100)),
                v2.ColorJitter(brightness=(1, 1.75)),
                v2.RandomEqualize(),
                v2.RandomGrayscale()
            ]),
            v2.RGB()
        ])
