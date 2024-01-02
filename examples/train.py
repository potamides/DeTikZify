#!/usr/bin/env -S torchrun --nproc_per_node gpu
from argparse import ArgumentParser
from os.path import basename, join

from datasets import Dataset
import torch.distributed as dist
from transformers import set_seed
from transformers.utils.logging import enable_explicit_format, set_verbosity_info

from detikzify.dataset import load_dataset
from detikzify.model import load
from detikzify.train import train

def parse_args():
    argument_parser = ArgumentParser(
        description="Fine-tune DeTikZify on DaTikZ."
    )
    argument_parser.add_argument("--base_model",
        default="deepseek-ai/deepseek-coder-1.3b-base",
        help="which base model to use",
    )
    argument_parser.add_argument(
        "--projector",
        help="url or path to a pretrained projector for clip soft prompts for multimodal models"
    )
    argument_parser.add_argument("--datikz",
        required=True,
        help="path to the DaTikZ dataset (in parquet format)",
    )
    argument_parser.add_argument("--output",
        default="models/detikzify",
        help="directory where to write the model files",
    )
    argument_parser.add_argument("--gradient_checkpointing",
        action="store_true",
        help="use gradient checkpointing",
    )

    return argument_parser.parse_args()

if __name__ == "__main__":
    set_verbosity_info()
    enable_explicit_format()
    dist.init_process_group()
    set_seed(0)

    args = parse_args()
    model, tokenizer = load(args.base_model, pretrain_mm_mlp_adapter=args.projector)

    datikz: Dataset = load_dataset("parquet", data_files=args.datikz, split="train") # type: ignore
    datikz = datikz.select_columns(["image", "code"]).rename_column("code", "text")

    train(
        model=model,
        tokenizer=tokenizer,
        output_dir=join(args.output, basename(model.config.name_or_path)), # type: ignore
        dataset=datikz
    )
