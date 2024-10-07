#!/usr/bin/env -S torchrun --nproc_per_node gpu
from argparse import ArgumentParser
from datetime import timedelta
from itertools import chain
from os import sched_getaffinity
from os.path import basename, join

from accelerate import Accelerator
from datasets import Dataset
from datasets import concatenate_datasets, load_dataset
from torch import distributed as dist
from transformers import AutoTokenizer, set_seed
from transformers.utils.logging import enable_explicit_format, set_verbosity_info

from detikzify.model import load
from detikzify.model.adapter import AdapterProcessor
from detikzify.train.adapter import CrossAttentionSiglipVisionModel, pretrain
from detikzify.util import batchify, convert, expand

@batchify
def process_arxivcap(batch, size):
    """Concatenate captions and OCR tokens."""
    for caption_images in chain.from_iterable(batch['caption_images']):
        caption = caption_images['caption']
        for cil_pair in caption_images['cil_pairs']:
            sub_caption = cil_pair['sub_caption']
            if text:=" ".join(filter(None, [caption, sub_caption])):
                yield dict(
                    text=text,
                    image=convert(expand(cil_pair['image'], size, do_trim=True), "png")
                )

def process_openmoji(ex, size):
    ex['image'] = convert(expand(ex['image'], size, do_trim=True), "png")
    return ex

def parse_args():
    argument_parser = ArgumentParser(
        description="Pre-train a DeTikZify adapter on ArxivCap."
    )
    argument_parser.add_argument("--base_model",
        required=True,
        help="The DeTikZify model checkpoint for weights initialization."
    )
    argument_parser.add_argument("--embedding_model",
        required=True,
        help="The adapter embedding model checkpoint for weights initialization."
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
    model, processor = load(args.base_model)

    vision_model = CrossAttentionSiglipVisionModel.from_pretrained(
        pretrained_model_name_or_path=None,
        config=model.config.vision_config,
        state_dict=model.model.vision_model.state_dict(),
        torch_dtype="bfloat16",
    )
    del model

    vision_model.init_cross_attn_adapter(args.embedding_model)
    processor = AdapterProcessor(
        processor=processor.image_processor,
        tokenizer=AutoTokenizer.from_pretrained(
            args.embedding_model,
            pad_token="<|finetune_right_pad_id|>",
            model_max_length=512,
        )
    )
    vision_model.embedding_model.config.pad_token_id = processor.tokenizer.pad_token_id

    with Accelerator().main_process_first():
        arxivcap: Dataset = load_dataset("MMInstruction/ArxivCap", split="train") # type: ignore
        openmoji: Dataset = load_dataset("soypablo/Emoji_Dataset-Openmoji", split="train") # type: ignore
        arxivcap = arxivcap.map(
            process_arxivcap,
            batched=True,
            remove_columns=arxivcap.column_names,
            batch_size=100,
            fn_kwargs=dict(size=vision_model.config.image_size),
            num_proc=len(sched_getaffinity(0))
        )
        openmoji = openmoji.map(
            process_openmoji,
            fn_kwargs=dict(size=vision_model.config.image_size),
        )

    pretrain(
        model=vision_model,
        processor=processor,
        dataset=concatenate_datasets([arxivcap, openmoji]),
        output_dir=join(args.output, basename(args.base_model)),
        gradient_checkpointing=args.gradient_checkpointing,
        deepspeed=args.deepspeed,
    )
