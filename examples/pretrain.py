#!/usr/bin/env -S torchrun --nproc_per_node gpu
from argparse import ArgumentParser
from functools import partial
from os.path import basename, join

from datasets import Dataset, IterableDataset
from transformers import set_seed
from transformers.utils.logging import enable_explicit_format, set_verbosity_info

from detikzify.dataset import load_dataset
from detikzify.model import load
from detikzify.train import pretrain
from detikzify.util import listify, expand, convert

@listify
def preprocess(exs, size):
    """Concatenate captions and OCR tokens."""
    for ex in exs:
        for caption_images in ex['caption_images']:
            caption = caption_images['caption']
            for cil_pair in caption_images['cil_pairs']:
                sub_caption = cil_pair['sub_caption']
                ocr = " ".join(cil_pair['image_ocr'])
                if text:=" ".join(filter(None, [caption, sub_caption, ocr])):
                    yield dict(
                        text=text,
                        image=convert(expand(cil_pair['image'], size, do_trim=True), "png")
                    )

def parse_args():
    argument_parser = ArgumentParser(
        description="Pretrain projection layer of DeTikZify."
    )
    argument_parser.add_argument("--base_model",
        help=(
            "The model checkpoint for weights initialization. "
            "Leave None if you want to train a model from scratch"
        )
    )
    argument_parser.add_argument("--size",
        default=1_000_000,
        type=int,
        help="The amount of figures to use for pretraining."
    )
    argument_parser.add_argument("--output",
        default="models/projector",
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
    model, processor = load(args.base_model)

    arxivcap: IterableDataset = load_dataset("MMInstruction/ArxivCap", split="train", streaming=True) # type: ignore
    arxivcap = arxivcap.shuffle().map(
        preprocess,
        batched=True,
        remove_columns=arxivcap.column_names,
        fn_kwargs=dict(size=model.config.vision_config.image_size),
    )

    pretrain(
        model=model,
        processor=processor,
        dataset=Dataset.from_generator(partial(iter, arxivcap.take(args.size)), features=arxivcap.features),
        output_dir=join(args.output, basename(model.config.name_or_path)),
        gradient_checkpointing=args.gradient_checkpointing,
        deepspeed=args.deepspeed,
    )
