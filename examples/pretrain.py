#!/usr/bin/env -S torchrun --nproc_per_node gpu
from argparse import ArgumentParser
from itertools import chain
from os.path import basename, join

from datasets import concatenate_datasets, Dataset
from transformers import set_seed
from transformers.utils.logging import enable_explicit_format, set_verbosity_info

from detikzify.dataset import load_dataset
from detikzify.model import load
from detikzify.train import pretrain

def preprocess(ex):
    """Concatenate captions, paragraph mentions, and ocr tokens."""
    text = " ".join(chain(
        [ex["caption"]] if ex["caption"] else [],
        chain.from_iterable(ex["mention"] or [[]]),
        ex["ocr"] or []
    ))
    return dict(
        text=text.strip(),
        image=ex['image']
    )

def parse_args():
    argument_parser = ArgumentParser(
        description="Pretrain projection layer of DeTikZify."
    )
    argument_parser.add_argument("--base_model",
        default="deepseek-ai/deepseek-coder-1.3b-base",
        help="which base model to use",
    )
    argument_parser.add_argument("--datikz",
        required=True,
        help="path to the DaTikZ dataset (in parquet format)",
    )
    argument_parser.add_argument("--output",
        default="models/projector",
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
    set_seed(0)

    args = parse_args()
    model, tokenizer = load(args.base_model)

    paper2fig: Dataset = load_dataset("paper2fig", size=model.config.vision_config['input_size'][-1], split="train") # type: ignore
    scicap: Dataset = load_dataset("scicap", size=model.config.vision_config['input_size'][-1], split="train") # type: ignore
    datikz: Dataset = load_dataset("parquet", data_files=args.datikz, split="train") # type: ignore

    paper2fig = paper2fig.map(preprocess, remove_columns=paper2fig.column_names)
    scicap = scicap.map(preprocess, remove_columns=scicap.column_names)
    datikz = datikz.select_columns(["image", "caption"]).rename_column("caption", "text").filter(lambda ex: ex['text'])

    print(f"Paper2Fig100k: {len(paper2fig)}", f"SciCap: {len(scicap)}", f"DaTikZ: {len(datikz)}", sep="\n")

    pretrain(
        model=model,
        tokenizer=tokenizer,
        output_dir=join(args.output, basename(model.config.name_or_path)), # type: ignore
        dataset=concatenate_datasets([paper2fig, scicap, datikz]),
    )
