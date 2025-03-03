#!/usr/bin/env -S torchrun --nproc_per_node gpu
from argparse import ArgumentParser
from datetime import timedelta
from os.path import basename, join

from datasets import load_dataset
from torch import distributed as dist
from transformers import AutoTokenizer, set_seed
from transformers.utils.logging import enable_explicit_format, set_verbosity_info

from detikzify.model import load
from detikzify.model.adapter import AdapterProcessor
from detikzify.train.adapter import CrossAttentionSiglipVisionModel, train


def load_adapter(base_model, embedding_model, adapter_model):
    model, processor = load(base_model)
    vision_model = CrossAttentionSiglipVisionModel.from_pretrained(
        pretrained_model_name_or_path=None,
        config=model.config.vision_config,
        state_dict=model.model.vision_model.state_dict(),
        torch_dtype="bfloat16",
    )
    del model

    vision_model.load_cross_attn_adapter(embedding_model, adapter_model)
    processor = AdapterProcessor(
        processor=processor.image_processor,
        tokenizer=AutoTokenizer.from_pretrained(
            embedding_model,
            pad_token="<|finetune_right_pad_id|>",
            model_max_length=512,
        )
    )
    vision_model.embedding_model.config.pad_token_id = processor.tokenizer.pad_token_id

    return vision_model, processor

def parse_args():
    argument_parser = ArgumentParser(
        description="Fine-tune a TikZero adapter end-to-end, optionally conditioned on captions."
    )
    argument_parser.add_argument("--base_model",
        required=True,
        help="The DeTikZify model checkpoint for weights initialization."
    )
    argument_parser.add_argument("--embedding_model",
        default="meta-llama/Llama-3.2-1B",
        help=(
            "The adapter embedding model checkpoint for weights initialization. "
            "Only LLaMA 3.1/3.2 models are officially supported."
        )
    )
    argument_parser.add_argument("--adapter_model",
        required=True,
        help= "The adapter model checkpoint obtained from the `pretrain.py` script."
    )
    argument_parser.add_argument("--datikz",
        default="nllg/datikz-v3",
        help="Path or name of the DaTikZ dataset.",
    )
    argument_parser.add_argument("--caption_condition",
        action="store_true",
        help="whether to also condition model on captions",
    )
    argument_parser.add_argument("--output",
        required=True,
        help="directory where to write the model files",
    )
    argument_parser.add_argument("--deepspeed",
        help="path to a DeepSpeed json config file",
    )
    argument_parser.add_argument("--gradient_checkpointing",
        action="store_true",
        help="use gradient checkpointing",
    )

    return argument_parser.parse_args()

if __name__ == "__main__":
    set_verbosity_info()
    enable_explicit_format()
    dist.init_process_group(timeout=timedelta(days=3))
    set_seed(0)

    args = parse_args()
    vision_model, processor = load_adapter(args.base_model, args.embedding_model, args.adapter_model)
    datikz = load_dataset(args.datikz, split="train")

    train(
        model=vision_model,
        processor=processor,
        dataset=datikz.filter(lambda ex: len(ex['caption']) > 0),
        caption_condition=args.caption_condition,
        output_dir=join(args.output, basename(model.config.name_or_path)), # type: ignore
        gradient_checkpointing=args.gradient_checkpointing,
        deepspeed=args.deepspeed,
    )
