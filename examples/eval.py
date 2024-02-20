#!/usr/bin/env python
from argparse import ArgumentParser
from collections import defaultdict
from itertools import cycle
from functools import partial
from json import dump, load as load_json
from os.path import isfile, join

from datasets import load_dataset
from torch import bfloat16, float16
from torch.cuda import is_available as is_cuda_available, is_bf16_supported
from tqdm import tqdm
from transformers import set_seed
from transformers.utils import is_flash_attn_2_available

from detikzify.evaluate import (
    CrystalBLEU,
    KernelInceptionDistance,
    PatchSim,
    TexEditDistance,
)
from detikzify.infer import DetikzifyPipeline, TikzDocument
from detikzify.model import load as load_model

def parse_args():
    argument_parser = ArgumentParser(
        description="Evaluate fine-tuned models."
    )
    argument_parser.add_argument(
        "--cache_dir",
        help="directory where model outputs should be saved to",
    )
    argument_parser.add_argument(
        "--trainset",
        required=True,
        help="path to the datikz train set (in parquet format)",
    )
    argument_parser.add_argument(
        "--testset",
        required=True,
        help="path to the datikz test set (in parquet format)",
    )
    argument_parser.add_argument(
        "--output",
        required=True,
        help="where to save the computed scores (as json)",
    )
    argument_parser.add_argument(
        "--path",
        nargs='+',
        metavar="MODEL=PATH",
        required=True,
        help="(multiple) key-value pairs of model names and paths/urls to models/adapters (local or hub) or json files",
    )
    return argument_parser.parse_args()

def generate_brute_force(pipe, image, strict=False, return_list=False, **tqdm_kwargs):
    """Repeat inference until the generated tikz code compiles."""
    tikzpics = list()
    for _ in tqdm(cycle([True]), desc="Try", **tqdm_kwargs):
        tikzpics.append(tikzpic:=pipe(image=image))
        if not tikzpic.compiled_with_errors if strict else tikzpic.has_content:
            return tikzpics if return_list else tikzpic

def load_metrics(trainset):
    bleu = CrystalBLEU(corpus=trainset)
    eed = TexEditDistance()
    wmdsim = PatchSim()
    poolsim = PatchSim(pool=True)
    kid = KernelInceptionDistance()

    def compile_sampling_rate(predictions):
        samples = list()
        for preds in predictions:
            samples.append(len(preds))
        return sum(samples) / len(predictions)

    def compute(references, predictions):
        ref_code, pred_code = [[ref['code']] for ref in references], [pred[-1].code for pred in predictions]
        ref_image, pred_image = [ref['image'] for ref in references], [pred[-1].rasterize() for pred in predictions]
        assert all(pred[-1].has_content for pred in predictions)

        scores = {
            "CompileSamplingRate": compile_sampling_rate(predictions=predictions)
        }
        metrics = {
            bleu: partial(bleu.update, list_of_references=ref_code, hypotheses=pred_code),
            eed: partial(eed.update, target=ref_code, preds=pred_code),
            wmdsim: lambda: [wmdsim.update(img1=img1, img2=img2) for img1, img2 in zip(ref_image, pred_image)],
            poolsim: lambda: [poolsim.update(img1=img1, img2=img2) for img1, img2 in zip(ref_image, pred_image)],
            kid: lambda: [(kid.update(img1, True), kid.update(img2, False)) for img1, img2 in zip(ref_image, pred_image)],
        }

        for metric, update in metrics.items():
            update()
            scores[str(metric)] = metric.compute() # type: ignore
            metric.reset()
        return scores

    return compute

if __name__ == "__main__":
    set_seed(0)
    args = parse_args()

    trainset = load_dataset("parquet", data_files=args.trainset, split="train")
    testset = load_dataset("parquet", data_files={"test": args.testset}, split="test").sort("caption") # type: ignore

    predictions = defaultdict(list)
    for model_name, path in map(lambda s: s.split('='), tqdm(args.path, desc="Predicting")):
        if path.endswith("json"):
            with open(path) as f:
                predictions[model_name] = [[TikzDocument(code) for code in sample] for sample in load_json(f)]
        else:
            model, tokenizer = load_model(
                base_model=path,
                device_map="auto",
                torch_dtype=bfloat16 if is_cuda_available() and is_bf16_supported() else float16,
                attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
            )
            pipe = DetikzifyPipeline(model=model, tokenizer=tokenizer)
            cache_file = join(args.cache_dir, f'{model_name}.json') if args.cache_dir else None

            if cache_file and isfile(cache_file):
                with open(cache_file) as f:
                    predictions[model_name] = [[TikzDocument(code) for code in sample] for sample in load_json(f)]
            try:
                for idx, item in enumerate(tqdm(testset, desc=model_name.title(), leave=False, position=0)):
                    if idx >= len(predictions[model_name]):
                        tikz = generate_brute_force(pipe, image=item['image'], return_list=True, position=1, leave=False) # type: ignore
                        predictions[model_name].append(tikz)
                del model, tokenizer, pipe
            finally:
                if cache_file:
                    with open(cache_file, 'w') as f:
                        dump([[p.code for p in ps] for ps in predictions[model_name]], f)

    scores = dict()
    metrics = load_metrics(trainset['code']) # type: ignore
    for model_name, prediction in tqdm(predictions.items(), desc="Computing metrics", total=len(predictions)):
        scores[model_name] = metrics(references=testset, predictions=prediction)
    with open(args.output, "w") as file:
        dump(scores, file)
