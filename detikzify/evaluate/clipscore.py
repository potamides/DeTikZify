from functools import cached_property
from typing import List

from PIL import Image
import torch
from torch.cuda import is_available as is_cuda_available, is_bf16_supported
from torchmetrics import Metric
from transformers import AutoModel, AutoProcessor

from ..util import expand, infer_device, load

class ClipScore(Metric):
    """Calculates CLIPScore which is a text-to-image similarity metric."""

    higher_is_better = True

    def __init__(
        self,
        model_name: str = "google/siglip-so400m-patch14-384",
        preprocess: bool = True,
        device: str = infer_device(),
        dtype=torch.bfloat16 if is_cuda_available() and is_bf16_supported() else torch.float16,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.preprocess = preprocess
        self._device = device
        self.set_dtype(dtype)

        self.add_state("score", torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum")
        self.add_state("n_samples", torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    def __str__(self):
        return self.__class__.__name__

    @cached_property
    def model(self):
        model = AutoModel.from_pretrained(self.model_name, torch_dtype=self.dtype)
        return model.to(self.device)

    @cached_property
    def processor(self):
        return AutoProcessor.from_pretrained(self.model_name)

    def update(
        self,
        images: Image.Image | str | List[Image.Image | str],
        text: str | List[str]
    ):
        images = images if isinstance(images, List) else [images]
        text = text if isinstance(text, List) else [text]

        for img, txt in zip(images, text):
            img = load(img)
            if self.preprocess:
                img = expand(img, max(img.size), do_trim=True)

            with torch.inference_mode():
                inputs = self.processor(text=txt, images=img, truncation=True, return_tensors="pt")
                outputs = self.model(
                    input_ids=inputs.input_ids.to(self.device),
                    pixel_values=inputs.pixel_values.to(self.device, self.dtype)
                )
                self.score += torch.sigmoid(outputs.logits_per_image).item()
                self.n_samples += 1

    def compute(self):
        return (self.score / self.n_samples).item()
