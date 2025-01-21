from functools import cached_property
from math import tanh
from typing import List, Literal, Optional

from PIL import Image
from ot.lp import emd2
import torch
from torch.cuda import is_available as is_cuda_available, is_bf16_supported
import torch.nn.functional as F
from torchmetrics import Metric
from torchmetrics.functional import pairwise_cosine_similarity
from transformers import AutoImageProcessor, AutoModel, PreTrainedModel, ProcessorMixin

from ..model.adapter import (
    AdapterProcessor,
    CrossAttentionAdapterMixin as AdapterMixin,
    has_adapter,
)
from ..util import cast, expand, infer_device, load, unwrap_processor

class ImageSim(Metric):
    """Perceptual image similarity using visual encoders."""

    higher_is_better = True

    def __init__(
        self,
        model_name: str = "google/siglip-so400m-patch14-384",
        mode: Literal["emd", "cos", "cos_avg"] = "cos",
        preprocess: bool = True,
        device: str = infer_device(),
        dtype=torch.bfloat16 if is_cuda_available() and is_bf16_supported() else torch.float16,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.preprocess = preprocess
        self.mode = mode
        self._device = device
        self.set_dtype(dtype)

        self.add_state("score", torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum")
        self.add_state("n_samples", torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    def __str__(self):
        return self.__class__.__name__ + f" ({self.mode.upper().replace('_', '-')})"

    @cached_property
    def model(self):
        # even if we instantiate with from_detikzify we still end up in this function
        if (model:=dict(self.named_children()).get("model")) is None:
            model = AutoModel.from_pretrained(self.model_name, torch_dtype=self.dtype)
            model = model.vision_model.to(self.device)
        return model

    @cached_property
    def processor(self):
        return AutoImageProcessor.from_pretrained(self.model_name)

    @classmethod
    def from_detikzify(cls, model: PreTrainedModel, processor: ProcessorMixin, *args, **kwargs):
        derived_kwargs = dict(
            model_name = model.name_or_path,
            mode = getattr(model.config, "pooling_mode", "emd"),
            device = model.device,
            dtype = model.dtype,
        )
        imagesim = cls(*args, **(derived_kwargs | kwargs))

        if has_adapter(model):
            class AdapterVisionModel(type(model.model.vision_model), AdapterMixin):
                embedding_model=model.embedding_model
                adapter=model.adapter

                @classmethod
                def cast(cls, vision_model):
                    adapter_vision_model = cast(cls, vision_model)
                    adapter_vision_model.add_hooks()
                    return adapter_vision_model

            imagesim.model = AdapterVisionModel.cast(model.model.vision_model)
            imagesim.processor = AdapterProcessor(
                processor=unwrap_processor(processor).image_processor,
                tokenizer=processor.tokenizer # type: ignore
            )
        else:
            imagesim.model = model.model.vision_model
            imagesim.processor = unwrap_processor(processor).image_processor
        return imagesim

    def get_vision_features(self, image: Optional[Image.Image | str] = None, text: Optional[str] = None):
        if image is not None:
            image = load(image)
            if self.preprocess:
                image = expand(image, max(image.size), do_trim=True)

        with torch.inference_mode():
            if text is not None:
                encoding = self.processor(text=text, images=image, return_tensors="pt").to(self.device, self.dtype)
            else:
                encoding = self.processor(images=image, return_tensors="pt").to(self.device, self.dtype)
            if self.mode == "cos":
                return self.model(**encoding).pooler_output.squeeze()
            elif self.mode == "cos_avg":
                return self.model(**encoding).last_hidden_state.squeeze().mean(dim=0)
            else:
                return self.model(**encoding).last_hidden_state.squeeze()

    def get_similarity(
        self,
        img1: Optional[Image.Image | str] = None,
        img2: Optional[Image.Image | str] = None,
        text1: Optional[str] = None,
        text2: Optional[str] = None,
    ):
        img1_feats = self.get_vision_features(img1, text1)
        img2_feats = self.get_vision_features(img2, text2)

        if img1_feats.is_mps: # mps backend does not support dtype double
            img1_feats, img2_feats = img1_feats.cpu(), img2_feats.cpu()
        if img1_feats.ndim > 1:
            dists = 1 - pairwise_cosine_similarity(img1_feats.double(), img2_feats.double()).cpu().numpy()
            return 2 * tanh(-emd2(M=dists, a=list(), b=list())) + 1 # type: ignore
        else:
            return F.cosine_similarity(img1_feats.double(), img2_feats.double(), dim=0).item()

    def update(
        self,
        img1: Optional[Image.Image | str | List[Image.Image | str]] = None,
        img2: Optional[Image.Image | str | List[Image.Image | str]] = None,
        text1: Optional[str | List[str]] = None,
        text2: Optional[str | List[str]] = None,
    ):
        inputs = dict()
        for key, value in dict(img1=img1, img2=img2, text1=text1, text2=text2).items():
            if value is not None:
                inputs[key] = value if isinstance(value, List) else [value]

        assert not ({"img1", "text1"}.isdisjoint(inputs.keys()) or {"img2", "text2"}.isdisjoint(inputs.keys()))
        assert len(set(map(len, inputs.values()))) == 1

        for inpt in zip(*inputs.values()):
            self.score += self.get_similarity(**dict(zip(inputs.keys(), inpt)))
            self.n_samples += 1

    def compute(self):
        return (self.score / self.n_samples).item()
