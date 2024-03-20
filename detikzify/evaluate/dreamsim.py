from functools import cached_property
from typing import List
from dreamsim import dreamsim
from PIL import Image
from timm import create_model as create_model
import torch
from torch.cuda import is_available as is_cuda_available, is_bf16_supported
from torchmetrics import Metric
from huggingface_hub import cached_assets_path

from ..util import expand, infer_device

class DreamSim(Metric):
    """Perceptual image similarity using DreamSim"""

    higher_is_better = True

    def __init__(
        self,
        model_name: str = "ensemble",
        pretrained: bool = True,
        normalize: bool = True,
        preprocess: bool = True,
        device: str = infer_device(),
        dtype=torch.bfloat16 if is_cuda_available() and is_bf16_supported() else torch.float16,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.pretrained = pretrained
        self.normalize = normalize
        self._device = device
        self.dtype = dtype
        self.preprocess = preprocess

        self.add_state("score", torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum")
        self.add_state("n_samples", torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    def __str__(self):
        return self.__class__.__name__

    @cached_property
    def dreamsim(self):
        model, processor = dreamsim(
            dreamsim_type=self.model_name,
            pretrained = self.pretrained,
            normalize_embeds=self.normalize,
            device=str(self.device),
            cache_dir=str(cached_assets_path(library_name="evaluate", namespace=self.__class__.__name__.lower()))
        )
        for extractor in model.extractor_list:
            extractor.model = extractor.model.to(self.dtype)
            extractor.proj = extractor.proj.to(self.dtype)
        return dict(
            model=model.to(self.dtype),
            processor=processor
        )

    @property
    def model(self):
        return self.dreamsim['model']

    @property
    def processor(self):
        return self.dreamsim['processor']

    def update(
        self,
        img1: Image.Image | str | List[Image.Image | str],
        img2: Image.Image | str | List[Image.Image | str],
    ):
        if isinstance(img1, List) or isinstance(img2, List):
            assert type(img1) == type(img2) and len(img1) == len(img2) # type: ignore
        else:
            img1, img2 = [img1], [img2]

        for i1, i2 in zip(img1, img2): # type: ignore
            if self.preprocess:
                i1 = expand(i1, max(i1.size), trim=True)
                i2 = expand(i2, max(i2.size), trim=True)
            i1 = self.processor(i1).to(self.device, self.dtype)
            i2 = self.processor(i2).to(self.device, self.dtype)
            with torch.inference_mode():
                self.score += 1 - self.model(i1, i2).item() # type: ignore
            self.n_samples += 1

    def compute(self):
        return (self.score / self.n_samples).item()
