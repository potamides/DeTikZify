from functools import cached_property
from typing import List

from PIL import Image
import torch
from torch import nn
from torch.cuda import is_available as is_cuda_available, is_bf16_supported
from torchmetrics.image.kid import KernelInceptionDistance as KID
from transformers import AutoModel, AutoImageProcessor

from ..util import expand, infer_device, load

class FeatureWrapper(nn.Module):
    def __init__(self, model_name, device, dtype):
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.dtype = dtype

    @cached_property
    def model(self):
        model = AutoModel.from_pretrained(self.model_name, torch_dtype=self.dtype)
        return model.to(self.device)

    def forward(self, pixel_values):
        with torch.inference_mode():
            return self.model.get_image_features(pixel_values.to(self.device, self.dtype))

class KernelInceptionDistance(KID):
    """Wrapper around torchmetrics Kernel Inception Distance with CLIP"""

    def __init__(
        self,
        model_name: str = "google/siglip-so400m-patch14-384",
        subset_size: int = 50,
        preprocess: bool = True,
        device: str = infer_device(),
        dtype=torch.bfloat16 if is_cuda_available() and is_bf16_supported() else torch.float16,
        **kwargs
    ):
        super().__init__(
            subset_size=subset_size,
            feature=FeatureWrapper(
                model_name=model_name,
                device=device,
                dtype=dtype),
            **kwargs
            )
        self.preprocess = preprocess

    def __str__(self):
        return self.__class__.__name__

    @cached_property
    def processor(self):
        return AutoImageProcessor.from_pretrained(self.inception.model_name)

    def open(self, img):
        img = load(img)
        if self.preprocess:
            return expand(img, max(img.size), do_trim=True)
        return img

    def update(self, imgs: Image.Image | str | List[Image.Image | str], *args, **kwargs):
        if not isinstance(imgs, List):
            imgs = [imgs]
        super().update(
            self.processor([self.open(img) for img in imgs], return_tensors="pt")["pixel_values"],
            *args,
            **kwargs
        )

    def compute(self, *args, **kwargs): # type: ignore
        return tuple(tensor.item() for tensor in super().compute(*args, **kwargs))
