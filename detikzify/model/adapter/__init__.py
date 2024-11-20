from transformers import AutoTokenizer

from .modeling_adapter import CrossAttentionAdapterMixin
from .processing_adapter import AdapterProcessor

def has_adapter(model):
    return hasattr(model, "adapter")

def load(model, processor, adapter_name_or_path=None, **kwargs):
    embedding_model = "meta-llama/Llama-3.2-1B"
    model.load_cross_attn_adapter(embedding_model, adapter_name_or_path, **kwargs)
    processor = AdapterProcessor(
        processor=processor,
        tokenizer=AutoTokenizer.from_pretrained(
            embedding_model,
            pad_token="<|finetune_right_pad_id|>",
            model_max_length=512,
        ),
    )
    model.embedding_model.config.pad_token_id = processor.tokenizer.pad_token_id

    return model, processor
