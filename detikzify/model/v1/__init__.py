from datasets import DownloadManager
from transformers import AutoConfig, AutoModel
from transformers import AutoTokenizer, PretrainedConfig
from transformers.utils.hub import is_remote_url

from .configuration_detikzify import *
from .modeling_detikzify import *
from .processing_detikzify import *

models = [
    "nllg/detikzify-ds-1.3b",
    "nllg/detikzify-ds-7b",
    "nllg/detikzify-tl-1.1b",
    "nllg/detikzify-cl-7b",
]

def register():
    try:
        AutoConfig.register("detikzify", DetikzifyConfig)
        AutoModel.register(DetikzifyConfig, DetikzifyForCausalLM)
    except ValueError:
        pass # already registered

def load(model_name_or_path, vision_tower="vit_so400m_patch14_siglip_384.webli", modality_projector=None, **kwargs):
    base_tokenizer = PretrainedConfig.from_pretrained(model_name_or_path).name_or_path or model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=base_tokenizer,
        model_max_length=2048,
        add_bos_token=False,
        add_eos_token=True,
        pad_token="<pad>",
        padding_side="right", # NOTE: only for training, need to change to "left" for batched inference
        legacy=False
    )
    model = DetikzifyForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name_or_path,
        use_cache=True,
        **kwargs
    )
    model.config.model_type = DetikzifyConfig.model_type # type: ignore
    model.generation_config.pad_token_id = tokenizer.pad_token_id # type: ignore

    if len(tokenizer) > model.config.vocab_size: # type: ignore
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8) # type: ignore
    if modality_projector and is_remote_url(modality_projector):
        modality_projector = DownloadManager().download(modality_projector)

    processor = model.get_model().initialize_vision_modules( # type: ignore
        patch_token_id=tokenizer.bos_token_id,
        modality_projector=modality_projector,
        vision_tower=getattr(model.config, "vision_tower", vision_tower), # type: ignore
        feature_layer=getattr(model.config, "feature_layer", -1), # type: ignore
        concat_patches=getattr(model.config, "concat_patches", 3) # type: ignore
    )

    return model, tokenizer, processor
