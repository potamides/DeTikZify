#!/usr/bin/env -S torchrun --nproc_per_node gpu
from argparse import ArgumentParser
from collections import defaultdict
from datetime import timedelta
from functools import partial
from itertools import count
from json import dump, load as load_json
from operator import itemgetter
from os import getenv
from os.path import isfile, join
from time import time

from datasets import load_dataset
from numpy import array
from scipy.stats.mstats import winsorize
from torch import bfloat16, distributed as dist, float16
from torch.cuda import is_available as is_cuda_available, is_bf16_supported
from tqdm import tqdm
from transformers import set_seed
from transformers.utils import is_flash_attn_2_available

from detikzify.evaluate import (
    ClipScore,
    CrystalBLEU,
    DreamSim,
    ImageSim,
    KernelInceptionDistance,
    TexEditDistance,
)
from detikzify.infer import DetikzifyPipeline, TikzDocument
from detikzify.model import adapter, load as load_model

WORLD_SIZE = int(getenv("WORLD_SIZE", 1))
RANK = int(getenv("RANK", 0))

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
        default="nllg/datikz-v3",
        help="path or name of the DaTikZ train set",
    )
    argument_parser.add_argument(
        "--testset",
        required=True,
        help="path to the DaTikZ test split processed by the ./sketchify script (in parquet format)",
    )
    argument_parser.add_argument(
        "--output",
        required=True,
        help="where to save the computed scores (as json)",
    )
    argument_parser.add_argument(
        "--timeout",
        type=int,
        help="minimum time to run MCTS in seconds",
    )
    argument_parser.add_argument(
        "--model_inputs",
        default="image",
        choices=["image", "sketch", "caption", "caption-image", "caption-sketch"],
        help="which inputs to condition the model on",
    )
    argument_parser.add_argument(
        "--path",
        nargs='+',
        metavar="MODEL=PATH[:ADAPTER] | MODEL=JSON",
        required=True,
        help="(multiple) key-value pairs of model names and paths/urls to models and optionally adapters or json files",
    )
    return argument_parser.parse_args()

# https://stackoverflow.com/a/54802737
def chunk(l, n):
    """Yield n number of striped chunks from l."""
    for i in range(0, n):
        yield l[i::n]

def interleave(chunks):
    """Interleave chunks until one is exhausted."""
    interleaved = list()
    for idx in count():
        try:
            interleaved.extend(chunk[idx] for chunk in chunks)
        except IndexError:
            break
    return interleaved

def generate(pipe, item, model_inputs, strict=False, timeout=None, **tqdm_kwargs):
    """Run MCTS until the generated tikz code compiles."""
    start, success, tikzpics = time(), False, set()
    inputs = {"text" if key == "caption" else "image":  item[key] for key in model_inputs.split("-")}

    for score, tikzpic in tqdm(pipe.simulate(**inputs), desc="Try", **tqdm_kwargs):
        tikzpics.add((score, tikzpic.code))
        if not tikzpic.compiled_with_errors if strict else tikzpic.is_rasterizable:
            success = True
        if success and (not timeout or time() - start >= timeout):
            break
    return [tikzpic for _, tikzpic in sorted(tikzpics, key=itemgetter(0))]

def predict(model_name, base_model, testset, model_inputs="image", adapter_model=None, cache_file=None, timeout=None):
    predictions, worker_preds = list(), list()
    model, processor = load_model(
        model_name_or_path=base_model,
        device_map=RANK,
        torch_dtype=bfloat16 if is_cuda_available() and is_bf16_supported() else float16,
        attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
    )
    if adapter_model is not None:
        model, processor = adapter.load(model, processor, adapter_model)
    # if we don't have a timeout (i.e., only run mcts until we obtain smth compileable), we can use fast metrics
    pipe = DetikzifyPipeline(model=model, processor=processor, metric="model" if timeout else "fast")

    if cache_file and isfile(cache_file):
        with open(cache_file) as f:
            predictions = load_json(f)
    try:
        worker_chunk = list(chunk(list(testset)[len(predictions):], WORLD_SIZE))[RANK]
        # FIXME: right now there only is a progress bar for Rank 0
        for item in tqdm(worker_chunk, desc=f"{model_name.title()} ({RANK})", disable=RANK!=0):
            tikz = generate(pipe, item, model_inputs, timeout=timeout, position=1, leave=False, disable=RANK!=0)
            worker_preds.append(tikz)
        del model, processor, pipe
    finally:
        dist.all_gather_object(gathered:=WORLD_SIZE * [None], worker_preds)
        predictions.extend(interleave(gathered))
        if cache_file and RANK == 0:
            with open(cache_file, 'w') as f:
                dump(predictions, f)
    return predictions

def load_metrics(trainset, measure_throughput=False, **kwargs):
    bleu = CrystalBLEU(corpus=trainset, **kwargs)
    eed = TexEditDistance(**kwargs)
    clip = ClipScore(**kwargs)
    imgsim = ImageSim(**kwargs)
    dreamsim = DreamSim(**kwargs)
    kid = KernelInceptionDistance(**kwargs)

    def mean_token_efficiency(predictions, limit=0.05):
        samples = list()
        for preds in predictions:
            samples.append(len(preds[-1].code)/sum(len(pred.code) for pred in preds))
        return winsorize(array(samples), limits=limit).mean().item()

    def mean_sampling_throughput(predictions, limit=0.05):
        return winsorize(array(list(map(len, predictions))), limits=limit).mean().item()

    def compute(references, predictions, compute_redacted=True, **redact_kwargs):
        ref_code, pred_code = [[ref['code']] for ref in references], [pred[-1].code for pred in predictions]
        ref_image, pred_image = [ref['image'] for ref in references], [pred[-1].rasterize() for pred in predictions]
        captions = [ref['caption'] for ref in references]
        assert all(pred[-1].is_rasterizable for pred in predictions)

        if measure_throughput:
            scores = {"MeanSamplingThroughput": mean_sampling_throughput(predictions=predictions)}
        else:
            scores = {"MeanTokenEfficiency": mean_token_efficiency(predictions=predictions)}

        redacted_metrics, standard_metrics = {}, {
            bleu: partial(bleu.update, list_of_references=ref_code, hypotheses=pred_code),
            eed: partial(eed.update, target=ref_code, preds=pred_code),
            clip: partial(clip.update, text=captions, images=pred_image),
            imgsim: lambda: [imgsim.update(img1=img1, img2=img2) for img1, img2 in zip(ref_image, pred_image)],
            dreamsim: lambda: [dreamsim.update(img1=img1, img2=img2) for img1, img2 in zip(ref_image, pred_image)],
            kid: lambda: [(kid.update(img1, True), kid.update(img2, False)) for img1, img2 in zip(ref_image, pred_image)],
        }

        if compute_redacted:
            pred_redacted = [pred[-1].rasterize(redact=True, **redact_kwargs) for pred in predictions]
            redacted_metrics.update({
                clip: partial(clip.update, text=captions, images=pred_redacted),
                imgsim: lambda: [imgsim.update(img1=img1, img2=img2) for img1, img2 in zip(ref_image, pred_redacted)],
                dreamsim: lambda: [dreamsim.update(img1=img1, img2=img2) for img1, img2 in zip(ref_image, pred_redacted)],
            })

        for metrics, redacted in [(standard_metrics, False), (redacted_metrics, True)]:
            for metric, update in metrics.items():
                update()
                if redacted:
                    scores[f"Redacted {str(metric)}"] = metric.compute()
                else:
                    scores[str(metric)] = metric.compute() # type: ignore
                metric.reset()

        return scores

    return compute

if __name__ == "__main__":
    set_seed(0)
    dist.init_process_group(timeout=timedelta(days=3))
    args = parse_args()

    trainset = load_dataset(args.trainset, split="train")
    testset = load_dataset("parquet", data_files={"test": args.testset}, split="test").sort("caption") # type: ignore

    predictions = defaultdict(list)
    for model_name, path in map(lambda s: s.split('='), tqdm(args.path, desc="Predicting")):
        if path.endswith("json"):
            with open(path) as f:
                predictions[model_name] = load_json(f)
        else:
            cache_file = join(args.cache_dir, f'{model_name}.json') if args.cache_dir else None
            predictions[model_name] = predict(
                model_name=model_name,
                base_model=path.partition(":")[0],
                adapter_model=path.partition(":")[2] or None,
                model_inputs=args.model_inputs,
                testset=testset,
                cache_file=cache_file,
                timeout=args.timeout,
            )

    if RANK == 0: # Scoring only on main process
        scores = dict()
        metrics = load_metrics(trainset['code'], measure_throughput=args.timeout is not None, sync_on_compute=False) # type: ignore
        for model_name, prediction in tqdm(predictions.items(), desc="Computing metrics", total=len(predictions)):
            scores[model_name] = metrics(
                references=testset,
                # use an unrealistically long timeout as we know that the (last) images compile
                predictions=[[TikzDocument(code, 600) for code in pred] for pred in prediction],
                rot_13=True
            )
        with open(args.output, "w") as file:
            dump(scores, file)
