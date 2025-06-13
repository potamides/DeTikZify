#!/usr/bin/env -S torchrun --nproc_per_node gpu

from abc import ABC, abstractmethod
from argparse import ArgumentParser
from dataclasses import dataclass
from datetime import timedelta
from itertools import count
from json import dump, load
from os import getenv
from os.path import isfile
import torch
import re
from typing import Callable

from datasets import load_dataset
from openai import OpenAI
from torch import distributed as dist
from torch.cuda import is_available as is_cuda_available, is_bf16_supported
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoModelForVision2Seq, AutoTokenizer, AutoProcessor
from transformers.utils import is_flash_attn_2_available

from detikzify.infer import TikzDocument

WORLD_SIZE = int(getenv("WORLD_SIZE", 1))
RANK = int(getenv("RANK", 0))

@dataclass
class TikzReply:
    reply: str
    code: TikzDocument

    @property
    def raw(self):
        return dict(reply=self.reply, code=self.code.code)

class AbstractTikZGenerator(ABC):
    hyper_params = dict(max_tokens=4096+512, temperature=0.8, top_p=0.95)
    client_class: Callable
    prompt = (
        "Please generate a scientific figure according to the following requirements: {caption}. "
        "Your output should be in TikZ code. Do not include any text other than the TikZ code."
    )

    def __init__(self, object_name="scientific figure", sketch_inputs=False, *args, **kwargs):
        self.client = self.client_class(*args, **kwargs)
        self.object_name = object_name
        self.input_type = "sketch" if sketch_inputs else "picture"

    def __call__(self, *args, **kwargs):
        return self.generate(*args, **kwargs)

    @staticmethod
    def extract_code(response):
        # regex search for LaTeX code block
        latex_pattern = r"```(?:latex|tex|tikz)\n([\s\S]*?)\n```"
        cls = "\\documentclass[tikz]{{standalone}}\n\n{code}"
        doc = "\\begin{{document}}\n\n{code}\n\n\\end{{document}}"
        code = response.strip()
        matches = re.search(latex_pattern, code)

        if matches: # If a match is found, extract LaTeX code
            code = matches.group(1).strip()
        if not r"\begin{document}" in code:
            code = doc.format(code=code)
        if not r"\documentclass" in code:
            code = cls.format(code=code)

        return TikzDocument(code)

    @abstractmethod
    def _generate(self, text):
        raise NotImplementedError

    def generate(self, caption):
        response = self._generate(
            text=self.prompt.format(caption=caption)
        )

        return TikzReply(
            reply=response,
            code=self.extract_code(response)
        )


class OpenAITikZGenerator(AbstractTikZGenerator):
    client_class = OpenAI

    def _generate(self, text):
        response = self.client.chat.completions.create(
            model="gpt-4o",
            **self.hyper_params,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                    ],
                },
            ],
        )
        return response.choices[0].message.content


class IdeficsTikZGenerator(AbstractTikZGenerator):
    hyper_params = dict(max_new_tokens=4096, temperature=0.8, top_p=0.95, do_sample=True)

    def __init__(self, *args, **kwargs):
        model_name = "HuggingFaceM4/Idefics3-8B-Llama3"
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            *args,
            **kwargs
        )

    def _generate(self, text):
        # Create inputs
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                ]
            }
        ]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Generate
        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, **self.hyper_params)
        generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

        return generated_texts[-1].partition("\nAssistant: ")[-1]


class QwenVisionTikZGenerator(AbstractTikZGenerator):
    hyper_params = dict(max_new_tokens=4096, temperature=0.8, top_p=0.95, do_sample=True)

    def __init__(self, *args, **kwargs):
        model_name = "Qwen/Qwen2-VL-72B-Instruct"
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            *args,
            **kwargs
        )

    def _generate(self, text):
        # Create inputs
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                ]
            }
        ]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[prompt], return_tensors="pt")
        inputs = inputs.to(self.model.device)

        # Generate
        generated_ids = self.model.generate(**inputs, **self.hyper_params)
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids) ]
        generated_texts = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        return generated_texts[-1]


class QwenCoderTikzGenerator(AbstractTikZGenerator):
    hyper_params = dict(max_new_tokens=4096, temperature=0.8, top_p=0.95, do_sample=True)

    def __init__(self, *args, **kwargs):
        model_name = "Qwen/Qwen2.5-Coder-32B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            *args,
            **kwargs
        )

    def _generate(self, text):
        # Create inputs
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": text}
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # Generate
        generated_ids = self.model.generate(**model_inputs, **self.hyper_params)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return response

class TextDetikzifyGenerator(AbstractTikZGenerator):
    hyper_params = dict(max_new_tokens=4096, temperature=0.8, top_p=0.95, do_sample=True)

    def __init__(self, *args, **kwargs):
        model_name = "potmaides/text-detikzify-scratch-8b"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            *args,
            **kwargs
        )

    def _generate(self, text):
        model_inputs = self.tokenizer([text + self.tokenizer.bos_token], return_tensors="pt").to(self.model.device)

        # Generate
        generated_ids = self.model.generate(**model_inputs, **self.hyper_params)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return response

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


def parse_args():
    argument_parser = ArgumentParser(
        description="Generate tikzpictures with GPT-4 and co"
    )
    argument_parser.add_argument(
        "--output",
        required=True,
        help="where to save generated text",
    )
    argument_parser.add_argument(
        "--testset",
        required=True,
        help="path to the datikz test set (in parquet format)",
    )
    return argument_parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dist.init_process_group(timeout=timedelta(days=3))
    model = QwenCoderTikzGenerator(
        device_map=RANK,
        torch_dtype="bfloat16" if is_cuda_available() and is_bf16_supported() else "float16",
        attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
    )

    if isfile(args.output):
        with open(args.output) as f:
            all_predictions = load(f)
    else:
        all_predictions = list()
    worker_predictions = list()
    dataset = load_dataset(args.testset, name="test", split="test").sort("caption") # type: ignore
    worker_chunk = list(chunk(dataset['caption'][len(all_predictions):], WORLD_SIZE))[RANK]
    try:
        for idx, caption in enumerate(tqdm(worker_chunk, disable=RANK!=0)): # type: ignore
            text, code, success = list(), list(), False
            while True:
                reply = model(caption)
                code.append(reply.code.code)
                text.append(reply.reply)
                if reply.code.is_rasterizable:
                    worker_predictions.append(dict(code=code, text=text))
                    break
    finally:
        dist.all_gather_object(gathered:=WORLD_SIZE * [None], worker_predictions)
        all_predictions.extend(interleave(gathered))
        with open(args.output, "w") as f:
            dump(all_predictions, f)
