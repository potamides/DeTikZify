from functools import cached_property
from math import tanh
from typing import List, Literal

from PIL import Image
from numpy import clip
from ot.lp import emd2
from timm import create_model as create_model
from timm.data import create_transform, resolve_data_config
import torch
from torch.cuda import is_available as is_cuda_available, is_bf16_supported
import torch.nn.functional as F
from torchmetrics import Metric
from torchmetrics.functional import pairwise_cosine_similarity
from transformers import PreTrainedModel

from ..util import expand, infer_device, load

class ImageSim(Metric):
    """Perceptual image similarity using visual encoders."""

    higher_is_better = True

    def __init__(
        self,
        model_name: str = "vit_so400m_patch14_siglip_384.webli",
        mode: Literal["cos", "emd"] = "cos",
        emd_layer: int = -3,
        preprocess: bool = True,
        device: str = infer_device(),
        dtype=torch.bfloat16 if is_cuda_available() and is_bf16_supported() else torch.float16,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.emd_layer = emd_layer
        self.preprocess = preprocess
        self.mode = mode
        self._device = device
        self.set_dtype(dtype)

        self.add_state("score", torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum")
        self.add_state("n_samples", torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    def __str__(self):
        return self.__class__.__name__ + ("(EMD)" if self.mode == "emd" else "")

    @cached_property
    def model(self):
        model = create_model(self.model_name, pretrained=True)
        return model.to(self.device, self.dtype).requires_grad_(False)

    @cached_property
    def processor(self):
        vision_config = self.model.pretrained_cfg
        data_config = resolve_data_config(vision_config)
        return create_transform(**data_config, is_training=False)

    @classmethod
    def from_detikzify(cls, model: PreTrainedModel, *args, **kwargs):
        derived_kwargs = dict(
            emd_layer = model.config.feature_layer,
            model_name = model.config.vision_config['architecture'],
            device = model.device,
            dtype = model.dtype,
        )

        imagesim = cls(*args, **(derived_kwargs | kwargs))
        imagesim.model = model.get_model().get_vision_tower()
        return imagesim

    def get_vision_features(self, image: Image.Image | str):
        image = load(image)
        if self.preprocess:
            image = expand(image, max(image.size), trim=True)

        with torch.inference_mode():
            pixels = self.processor(image).unsqueeze(0).to(self.device, self.dtype) # type: ignore
            if self.mode == "cos":
                return self.model(pixels)[0]
            else:
                layers = [clip(self.emd_layer, -(depth:=len(self.model.blocks)), depth-1) % depth]
                return self.model.get_intermediate_layers(pixels, n=layers, norm=True)[0][0]

    def get_similarity(self, img1: Image.Image | str, img2: Image.Image | str):
        img1_feats = self.get_vision_features(img1)
        img2_feats = self.get_vision_features(img2)

        if img1_feats.is_mps: # mps backend does not support dtype double
            img1_feats, img2_feats = img1_feats.cpu(), img2_feats.cpu()
        if img1_feats.ndim > 1:
            dists = 1 - pairwise_cosine_similarity(img1_feats.double(), img2_feats.double()).cpu().numpy()
            return 2 * tanh(-emd2(M=dists, a=list(), b=list())) + 1 # type: ignore
        else:
            return F.cosine_similarity(img1_feats.double(), img2_feats.double(), dim=0).item()

    def update(
        self,
        img1: Image.Image | str | List[Image.Image | str],
        img2: Image.Image | str | List[Image.Image | str],
    ):
        if isinstance(img1, List) or isinstance(img2, List):
            assert type(img1) == type(img2) and len(img1) == len(img2) # type: ignore
        else:
            img1, img2 = [img1], [img2]

        for pair in zip(img1, img2): # type: ignore
            self.score += self.get_similarity(*pair)
            self.n_samples += 1

    def compute(self):
        return (self.score / self.n_samples).item()
