#!/usr/bin/env -S torchrun --nproc_per_node gpu
from argparse import ArgumentParser
from os.path import basename, join

from datasets import Dataset
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
        required=True,
        help="The model checkpoint for weights initialization."
    )
    argument_parser.add_argument(
        "--projector",
        help="url or path to a pretrained modality projector"
    )
    argument_parser.add_argument("--datikz",
        required=True,
        help="path to the DaTikZ train split processed by the ./sketchify script (in parquet format)",
    )
    argument_parser.add_argument("--sketch_ratio",
        default=.5,
        help="ratio of synthetic sketches generated through the ./sketchify script or image transforms",
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
    set_seed(0)

    args = parse_args()
    model, processor = load(args.base_model, modality_projector=args.projector)

    datikz: Dataset = load_dataset("parquet", data_files=args.datikz, split="train") # type: ignore
    datikz = datikz.select_columns(["image", "code", "sketches"]).rename_column("code", "text")

    train(
        model=model,
        processor=processor,
        dataset=datikz,
        sketch_ratio=args.sketch_ratio,
        output_dir=join(args.output, basename(model.config.name_or_path)), # type: ignore
        gradient_checkpointing=args.gradient_checkpointing,
        deepspeed=args.deepspeed,
    )
