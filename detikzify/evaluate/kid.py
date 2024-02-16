from functools import cached_property
from typing import List

from PIL import Image
from timm import create_model as create_model
from timm.data import create_transform, resolve_data_config
import torch
from torch import nn
from torch.cuda import is_available as is_cuda_available, is_bf16_supported
from torchmetrics.image.kid import KernelInceptionDistance as KID

from ..util import expand, infer_device, load

class TimmFeatureWrapper(nn.Module):
    def __init__(self, model_name, device, dtype):
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.dtype = dtype

    @cached_property
    def model(self):
        model = create_model(self.model_name, pretrained=True)
        return model.to(self.device, self.dtype).requires_grad_(False)

    def forward(self, pixel_values):
        return self.model(pixel_values.to(self.device, self.dtype))

class KernelInceptionDistance(KID):
    """Wrapper around torchmetrics Kernel Inception Distance with CLIP"""

    def __init__(
        self,
        model_name: str = "vit_so400m_patch14_siglip_384.webli",
        subset_size: int = 50,
        preprocess: bool = True,
        device: str = infer_device(),
        dtype=torch.bfloat16 if is_cuda_available() and is_bf16_supported() else torch.float16,
        **kwargs
    ):
        super().__init__(
            subset_size=subset_size,
            feature=TimmFeatureWrapper(
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
        vision_config = self.inception.model.pretrained_cfg
        data_config = resolve_data_config(vision_config) | dict(crop_pct=1) # we don't want a resize crop
        return create_transform(**data_config, is_training=False)

    def open(self, img):
        img = load(img)
        if self.preprocess:
            return expand(img, max(img.size), trim=True)
        return img

    def update(self, imgs: Image.Image | str | List[Image.Image | str], *args, **kwargs):
        if not isinstance(imgs, List):
            imgs = [imgs]
        super().update(
            torch.stack([self.processor(self.open(img)) for img in imgs]), # type: ignore
            *args,
            **kwargs
        )

    def compute(self, *args, **kwargs):
        return tuple(tensor.item() for tensor in super().compute(*args, **kwargs))
