#!/usr/bin/env -S torchrun --nproc_per_node gpu
from argparse import ArgumentParser
from collections import ChainMap
from datetime import timedelta
from functools import cached_property, partial
from operator import itemgetter
from os import environ
from os.path import basename
from random import choice, random
from sys import exit
from typing import Optional

from accelerate import Accelerator
from datasets import (
    Dataset,
    DatasetDict,
    Features,
    Image,
    Value,
    concatenate_datasets,
    load_dataset,
)
from torch.distributed import init_process_group
from torch.multiprocessing import Pool, set_start_method
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import is_flash_attn_2_available
from transformers.utils.logging import enable_explicit_format, set_verbosity_info

from detikzify.evaluate.imagesim import ImageSim
from detikzify.infer.tikz import TikzDocument
from detikzify.model import load
from detikzify.util import SketchAugment, batchify, expand
from sketchify import Sketchifier

try:
    from trl import GRPOConfig, GRPOTrainer
except ModuleNotFoundError:
    print(
        "You need to install trl with vision support to be able to use this script:",
        "git clone https://github.com/hellopahe/trl.git",
        "curl -L http://github.com/huggingface/trl/pull/3568.patch | git -C trl apply",
        "pip install trl",
        sep="\n\t"
    )
    exit(1)


WORLD_SIZE = int(environ.get("WORLD_SIZE", 1))


class RandSketchifier:
    def __init__(self, size, sketch_ratio):
        self.sketchifier = Sketchifier()
        self.deep_sketchifier = SketchAugment()
        self.size = size
        self.sketch_ratio = sketch_ratio

    @staticmethod
    def randbool():
        return choice([True, False])

    def sketchify(self, img):
        return self.sketchifier(img) if self.randbool() else self.deep_sketchifier(img)

    def randsketchify(self, img):
        return self.sketchify(img) if random() < self.sketch_ratio else img

    def __call__(self, img):
        return self.randsketchify(expand(img, self.size, do_trim=True))


class TrainDataset:
    def __init__(self, processor, datikz_name="nllg/datikz-v3", size=420, sketch_ratio=.5):
        self.processor = processor
        self.datikz: DatasetDict = load_dataset(datikz_name) # type: ignore
        self.sketchify = RandSketchifier(
            size=size,
            sketch_ratio=sketch_ratio
        )

    @staticmethod
    def get_fig_type(ftype):
        for type_ in ["table", "photograph", "plot", "schematic", "other"]:
            if type_ in ftype.lower():
                return type_
        return "N/A"

    @batchify
    def extract_figures(self, batch, meta_ds, filter_urls):
        for img in batch['image']:
            filename = basename(img['path'].split("::")[0])
            if filename in meta_ds and filename.rpartition("v")[0] not in filter_urls:
                meta = meta_ds[filename]
                ftype = self.get_fig_type(meta["figure_type"])
                # "other" are mostly text snippets
                if meta["content_type"] == "figure":
                    yield dict(image=img, type=ftype)

    def sample_spiqa_dataset(self, n_samples, split="train"):
        img_ds: Dataset = load_dataset( # type: ignore
            path="google/spiqa",
            data_files="train_val/SPIQA_train_val_Images.zip",
            split="train",
            features=Features({"image": Image(decode=False), "label": Value("string")})
        )

        meta_ds = load_dataset(path="google/spiqa", data_files=f"train_val/SPIQA_{split}.json", split="train")
        meta_ds = ChainMap(*map(itemgetter("all_figures"), meta_ds[0].values())) # type: ignore

        filter_urls = concatenate_datasets(list(self.datikz.values()))['uri']
        filter_urls = {basename(url) for url in filter_urls if url.startswith("https://arxiv.org")}

        img_ds = img_ds.map(
            self.extract_figures,
            batched=True,
            remove_columns=img_ds.column_names,
            fn_kwargs=dict(meta_ds=meta_ds, filter_urls=filter_urls)
        ).shuffle()

        schematics = img_ds.filter(lambda ex: ex['type'] == "schematic").select(range(round(.6 * n_samples)))
        plots = img_ds.filter(lambda ex: ex['type'] == "plot").select(range(round(.2 * n_samples)))
        other = img_ds.filter(lambda ex: ex['type'] not in ["plot", "schematic"]).select(range(n_samples - len(schematics) - len(plots)))

        for ex in concatenate_datasets([schematics, plots, other]).cast_column("image", Image()):
            yield {"prompt": "", "images": self.sketchify(ex['image'])} # type: ignore

    def sample_datikz_dataset(self, n_samples, split="train"):
        tokenizer, image_seq_len = self.processor.tokenizer, self.processor.image_seq_len

        datikz_filtered = self.datikz[split].filter(
            function=lambda ex: len(tokenizer.tokenize(ex['code'])) + image_seq_len > tokenizer.model_max_length,
        ).train_test_split(train_size=n_samples)

        for ex in datikz_filtered['train']:
            yield {"prompt": "", "images": self.sketchify(ex['image'])} # type: ignore

    def sample(self, n_samples):
        spiqa: Dataset = Dataset.from_generator( # type: ignore
            generator=self.sample_spiqa_dataset,
            gen_kwargs=dict(n_samples=round(.5 * n_samples))
        )
        datikz: Dataset = Dataset.from_generator( # type: ignore
            generator=self.sample_datikz_dataset,
            gen_kwargs=dict(n_samples=n_samples-len(spiqa))
        )

        return concatenate_datasets([spiqa, datikz])


class RewardFunc:
    __name__ = "SelfSim Reward"

    def __init__(self, model, processor, num_workers=1, strict=False):
        self.model = model
        self.processor = processor
        self.strict = strict
        self.pool = Pool(num_workers)

    @cached_property
    def reward_model(self):
        return ImageSim.from_detikzify(self.model, self.processor, sync_on_compute=False)

    @staticmethod
    def compile(code, size, strict):
        doc = TikzDocument(code)

        if doc.is_rasterizable and not (strict and doc.compiled_with_errors):
            return doc.rasterize(size=size)

    def __call__(self, images, completions, **_):
        rewards, compile = list(), partial(
            self.compile,
            size=self.model.config.vision_config.image_size,
            strict=self.strict
        )

        for doc, img in zip(self.pool.imap(compile, completions), images):
            if doc is not None:
                self.reward_model.update(doc, img)
                rewards.append(self.reward_model.compute())
            else:
                rewards.append(-1.)
            self.reward_model.reset()
        return rewards


def train(
    model,
    processor,
    dataset,
    output_dir: str,
    overwrite: bool = False,
    deepspeed: Optional[str] = None,
    num_compile_workers: int = 4,
    # training hyperparams
    strict: bool = False,
    freeze_encoder: bool = True,
    num_generations: int = 32,
    batch_size: int = 16,
    micro_batch_size: int = 1,
    num_train_steps: int = 500,
    learning_rate: float = 1e-5,
    gradient_checkpointing: bool = False,
):
    for _, param in model.model.vision_model.named_parameters():
        param.requires_grad = not freeze_encoder
    model.enable_input_require_grads()

    training_args = GRPOConfig(
        per_device_train_batch_size=micro_batch_size,
        gradient_accumulation_steps=num_generations * batch_size // micro_batch_size // WORLD_SIZE,
        num_generations=num_generations,
        gradient_checkpointing=gradient_checkpointing,
        # https://github.com/huggingface/transformers/issues/32576
        gradient_checkpointing_kwargs={'use_reentrant': False},
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        epsilon=0.4,
        temperature=max(1., model.generation_config.temperature),
        top_p=model.generation_config.top_p,
        top_k=model.generation_config.top_k,
        max_steps=num_train_steps,
        logging_steps=num_train_steps//100,
        save_steps=num_train_steps//10,
        save_strategy="steps",
        save_total_limit=1,
        learning_rate=learning_rate,
        torch_compile=True,
        bf16=True,
        tf32=True,
        max_completion_length=processor.tokenizer.model_max_length-processor.image_seq_len,
        max_prompt_length=None,
        optim="adamw_torch" if deepspeed else "adamw_torch_fused",
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        report_to="none",
        log_completions=True,
        num_completions_to_print=1,
        output_dir=output_dir,
        overwrite_output_dir=overwrite,
        deepspeed=deepspeed,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=processor,
        reward_funcs=[RewardFunc(model, processor, num_workers=num_compile_workers, strict=strict)],
        args=training_args,
        train_dataset=dataset,
    )
    trainer.generation_config.bad_words_ids = [[model.config.image_token_id]]
    # trainer.generation_config.begin_suppress_tokens = [model.config.text_config.eos_token_id]
    trainer.train(resume_from_checkpoint=None if overwrite else get_last_checkpoint(output_dir))

    if trainer.is_deepspeed_enabled:
        # https://huggingface.co/docs/accelerate/v0.11.0/en/deepspeed#saving-and-loading
        from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint # type: ignore
        last_checkpoint = get_last_checkpoint(output_dir)
        load_state_dict_from_zero_checkpoint(trainer.model.float(), last_checkpoint)

    trainer.save_model(output_dir)


def parse_args():
    argument_parser = ArgumentParser(
        description="Post-train DeTikZify with GRPO."
    )
    argument_parser.add_argument("--base_model",
        required=True,
        help="The model checkpoint for weights initialization."
    )
    argument_parser.add_argument("--datikz",
        default="nllg/datikz-v3",
        help="Path or name of the DaTikZ dataset.",
    )
    argument_parser.add_argument("--sketch_ratio",
        default=.5,
        type=float,
        help="ratio of synthetic sketches generated through UltraSketch or image transforms",
    )
    argument_parser.add_argument("--output",
        required=True,
        dest="output_dir",
        help="directory where to write the model files",
    )
    argument_parser.add_argument("--num_compile_workers",
        default=4,
        type=int,
        help="number of threads to compile TikZ code with",
    )
    argument_parser.add_argument("--deepspeed",
        help="path to a DeepSpeed json config file",
    )
    argument_parser.add_argument("--gradient_checkpointing",
        action="store_true",
        help="use gradient checkpointing",
    )
    argument_parser.add_argument("--strict",
        action="store_true",
        help="treat recoverable compilation errors as fatal errors",
    )
    argument_parser.add_argument("--batch_size",
        default=16,
        type=int,
        help="global batch size for training",
    )
    argument_parser.add_argument("--num_train_steps",
        default=500,
        type=int,
        help="number of training steps to run GRPO for",
    )
    return vars(argument_parser.parse_args())


if __name__ == "__main__":
    set_verbosity_info()
    enable_explicit_format()
    set_start_method('forkserver') # https://github.com/pytorch/pytorch/issues/17199#issuecomment-465313245
    init_process_group("nccl", timeout=timedelta(days=3))
    set_seed(0)

    args = parse_args()
    model, processor = load(
        model_name_or_path=args.pop("base_model"),
        torch_dtype="bfloat16",
        attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
    )

    with Accelerator().main_process_first():
        dataset = TrainDataset(
            processor=processor,
            datikz_name=args.pop('datikz'),
            sketch_ratio=args.pop("sketch_ratio"),
            size=model.config.vision_config.image_size,
        ).sample(args['batch_size'] * args['num_train_steps'])

    train(model=model, processor=processor, dataset=dataset, **args)
