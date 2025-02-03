#!/usr/bin/env -S torchrun --nproc_per_node gpu
from argparse import ArgumentParser
from datetime import timedelta
from os.path import basename, join

from accelerate import Accelerator
from datasets import Dataset
from datasets import load_dataset
from torch import distributed as dist
from transformers import AutoTokenizer, set_seed
from transformers.utils.logging import enable_explicit_format, set_verbosity_info

from detikzify.model import load
from detikzify.model.adapter import AdapterProcessor
from detikzify.train.adapter import CrossAttentionSiglipVisionModel, train
from detikzify.util import batchify, convert, expand

@batchify
def process_mlbcap(batch, size):
    for image, description in zip(batch['image'], batch['figure_description']):
        yield {
            "image": convert(expand(image, size, do_trim=True), "png"),
            "text": description
        }

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
        description="Fine-tune a DeTikZify adapter on MLBCAP descriptions."
    )
    argument_parser.add_argument("--base_model",
        required=True,
        help="The DeTikZify model checkpoint for weights initialization."
    )
    argument_parser.add_argument("--embedding_model",
        required=True,
        help=(
            "The adapter embedding model checkpoint for weights initialization. "
            "Only LLaMA 3.1/3.2 models are officially supported."
        )
    )
    argument_parser.add_argument("--adapter_model",
        required=True,
        help= "The adapter model checkpoint obtained from the `pretrain.py` script."
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
    argument_parser.add_argument("--multimodal",
        action="store_true",
        help="fine-tune an adapter for multimodal inputs",
    )

    return argument_parser.parse_args()

if __name__ == "__main__":
    set_verbosity_info()
    enable_explicit_format()
    dist.init_process_group(timeout=timedelta(days=3))
    set_seed(0)

    args = parse_args()
    vision_model, processor = load_adapter(args.base_model, args.embedding_model, args.adapter_model)

    with Accelerator().main_process_first():
        mlbcap: Dataset = load_dataset("TEAMREBOOTT-AI/SciCap-MLBCAP", split="train") # type: ignore
        mlbcap = mlbcap.map(
            process_mlbcap,
            remove_columns=mlbcap.column_names,
            batched=True,
            fn_kwargs=dict(size=vision_model.config.image_size),
        )

    train(
        model=vision_model,
        processor=processor,
        dataset=mlbcap,
        output_dir=join(args.output, basename(args.base_model)),
        gradient_checkpointing=args.gradient_checkpointing,
        deepspeed=args.deepspeed,
        multimodal=args.multimodal,
    )
