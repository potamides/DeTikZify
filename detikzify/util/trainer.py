from functools import partial

from numpy import arange
import torch
from torchvision import tv_tensors
from torchvision.transforms import v2
from transformers import (
    IntervalStrategy,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from transformers.trainer_utils import has_length
from torchvision.transforms.v2._utils import query_size

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

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs): # type: ignore
        steps = [round(self.num_update_steps_per_epoch * step) for step in self.steps]
        if (
            state.global_step % self.num_update_steps_per_epoch in steps
            and args.save_strategy == IntervalStrategy.EPOCH
        ):
            control.should_save = True

        return control

class SketchAugment(v2.Compose):
    def __init__(self, intensity=1):
        super().__init__([
            v2.RandomOrder([
                v2.ElasticTransform(alpha=50. * intensity, fill=255),
                v2.JPEG((40 * intensity, 100)),
                v2.ColorJitter(brightness=(.75 + .25 * intensity, 1.75)),
                v2.RandomEqualize(),
                v2.RandomGrayscale()
            ]),
            v2.RGB()
        ])

class FullErase(v2.Lambda):
    def __init__(self, value=255):
        super().__init__(partial(v2.functional.erase, i=0, j=0, h=-1, w=-1, v=torch.tensor(value)))

class EditBase(v2.Transform):
    def __init__(self, *, alpha: float = 1.0) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self._dist = torch.distributions.Beta(torch.tensor([alpha]), torch.tensor([alpha]))

    def _get_boxes(self, flat_inputs):
        lam = self._dist.sample((len(flat_inputs),)).squeeze() # type: ignore

        H, W = query_size(flat_inputs)

        r_x = torch.randint(W, size=(len(flat_inputs),))
        r_y = torch.randint(H, size=(len(flat_inputs),))

        r = 0.5 * torch.sqrt(1.0 - lam)
        r_w_half = (r * W).int()
        r_h_half = (r * H).int()

        x1 = torch.clamp(r_x - r_w_half, min=0)
        y1 = torch.clamp(r_y - r_h_half, min=0)
        x2 = torch.clamp(r_x + r_w_half, max=W)
        y2 = torch.clamp(r_y + r_h_half, max=H)

        grid_x, grid_y = torch.meshgrid(torch.arange(W), torch.arange(H), indexing="ij")

        grid_x = grid_x.unsqueeze(0).expand(len(flat_inputs), -1, -1)
        grid_y = grid_y.unsqueeze(0).expand(len(flat_inputs), -1, -1)

        mask = (grid_x >= x1.unsqueeze(1).unsqueeze(2)) & (grid_x < x2.unsqueeze(1).unsqueeze(2)) & \
               (grid_y >= y1.unsqueeze(1).unsqueeze(2)) & (grid_y < y2.unsqueeze(1).unsqueeze(2))

        return mask.unsqueeze(1).expand(-1, 3, -1, -1)

class EditCutMix(EditBase):
    def _transform(self, inpt, params):
        output = inpt.clone()
        rolled = inpt.roll(1, 0)
        box = self._get_boxes(inpt)
        output[box] = rolled[box]

        if isinstance(inpt, (tv_tensors.Image, tv_tensors.Video)):
            output = tv_tensors.wrap(output, like=inpt)

        return output

class EditMixUp(EditBase):
    def _transform(self, inpt, params):
        lam = self._dist.sample((len(inpt),)).view(-1, *([1] * len(inpt.shape[1:]))) # type: ignore
        output = inpt.roll(1, 0).mul(1.0 - lam).add_(inpt.mul(lam)).to(inpt.dtype)

        if isinstance(inpt, (tv_tensors.Image, tv_tensors.Video)):
            output = tv_tensors.wrap(output, like=inpt)

        return output

class EditCutOut(EditBase):
    def __init__(self, *args, value=255, **kwargs):
        self.value = value
        super().__init__(*args, **kwargs)

    def _transform(self, inpt, params):
        output = inpt.clone()
        box = self._get_boxes(inpt)
        output[box] = self.value

        if isinstance(inpt, (tv_tensors.Image, tv_tensors.Video)):
            output = tv_tensors.wrap(output, like=inpt)

        return output
