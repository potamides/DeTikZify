from datasets import DownloadManager
from safetensors.torch import load_file
from transformers import (
    AutoConfig,
    AutoModelForVision2Seq,
    AutoProcessor,
)
from transformers.utils.hub import is_remote_url

from .configuration_detikzify import *
from .modeling_detikzify import *
from .processing_detikzify import *

def register():
    try:
        AutoConfig.register("detikzify", DetikzifyConfig)
        AutoModelForVision2Seq.register(DetikzifyConfig, DetikzifyForConditionalGeneration)
        AutoProcessor.register(DetikzifyConfig, DetikzifyProcessor)
    except ValueError:
        pass # already registered

def load(model_name_or_path, modality_projector=None, **kwargs):
    register()
    processor = AutoProcessor.from_pretrained(model_name_or_path)
    model = AutoModelForVision2Seq.from_pretrained(model_name_or_path, **kwargs)

    if modality_projector is not None:
        if is_remote_url(modality_projector):
            modality_projector = DownloadManager().download(modality_projector)
        model.load_state_dict(
            state_dict=load_file(
                filename=modality_projector, # type: ignore
                device=str(model.device)
            ),
            strict=False
        )

    return model, processor
