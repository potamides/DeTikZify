from transformers import SiglipVisionModel

from ...model.adapter import CrossAttentionAdapterMixin
from .pretrain import train as pretrain
#from .train import train


class CrossAttentionSiglipVisionModel(SiglipVisionModel, CrossAttentionAdapterMixin):
    ...
