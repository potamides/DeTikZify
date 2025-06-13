#!/usr/bin/env python

from abc import ABC, abstractmethod
from argparse import ArgumentParser
import base64
from io import BytesIO
from json import dump, load
from os.path import isfile
import re
from time import time

from PIL import ImageDraw
from anthropic import Anthropic
from datasets import load_dataset
from openai import OpenAI

from detikzify.infer import TikzDocument


class AbstractVisualSelfRefiner(ABC):
    system_role = "You are a helpful assistant who can write TikZ code."
    hyper_params = dict(max_tokens=4096, temperature=0.8, top_p=0.95)
    client_class: type

    def __init__(self, object_name="scientific figure", sketch_inputs=False, *args, **kwargs):
        self.client = self.client_class(*args, **kwargs)
        self.object_name = object_name
        self.input_type = "sketch" if sketch_inputs else "picture"
        self.state = list()

    def __call__(self, *args, **kwargs):
        return self.refine(*args, **kwargs)

    @property
    def initial_prompt(self):
        return (
            f"This is a {self.input_type} of a {self.object_name}. "
            f"Generate LaTeX code that draws this {self.object_name} using TikZ. "
            "Ensure that the LaTeX code is self-contained and does not require any packages except TikZ-related imports. "
            "Don't forget to include \\usepackage{tikz}! "
            "I understand that this is a challenging task, so do your best. "
            "Return your result in a ```latex code block."
        )

    @property
    def refinement_prompt(self):
        return (
            "```latex\n{code}\n```\n"
            f"This is the TikZ/LaTeX code for the {self.object_name} shown in the picture labeled \"Input\". "
            f"Can you improve it to better resemble the provided reference {self.input_type}? "
            "First, analyze the \"Input\" picture to understand its components and layout. "
            f"Then, consider how the {self.object_name} can be enhanced to more closely match the reference {self.input_type}. "
            "Finally, rewrite the TikZ code to implement these improvements, making the image more similar to the reference. "
            "Ensure that the LaTeX code is self-contained and does not require any packages except TikZ-related imports. "
            "Don't forget to include \\usepackage{{tikz}}! "
            "Return your result in a ```latex code block."
        )
    @property
    def repair_prompt(self):
        return (
            "Given the error message:\n{error}\n"
            "And the problematic code:\n```latex\n{code}\n```\n"
            "First, identify the issue based on the error message. "
            "Then, determine the cause of the error in the code. "
            "Finally, propose and implement a solution. "
            "Return the fixed code in a ```latex code block."
        )

    @staticmethod
    def b64encode(img):
        buffered = BytesIO()
        img.save(buffered, format="png")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    @staticmethod
    def extract_code(response):
        # regex search for LaTeX code block
        latex_pattern = r"```(?:latex|tex)\n([\s\S]*?)\n```"
        matches = re.search(latex_pattern, response['text'])

        # If a match is found, return the LaTeX code
        if matches:
            return TikzDocument(matches.group(1))
        else:  # No LaTeX code block found
            return TikzDocument(response['text'])

    @abstractmethod
    def create(self, text, input_img=None, ref_img=None, input_label=None, ref_label=None):
        raise NotImplementedError

    def request(self, *args, **kwargs):
        response = self.create(*args, **kwargs)
        self.state.append(response)
        return response

    def generate(self, tikzdoc=None, *args, **kwargs):
        return self.extract_code(self.request(
            text=self.refinement_prompt.format(code=tikzdoc.code) if tikzdoc else self.initial_prompt,
            *args, **kwargs
        ))

    def repair(self, tikzdoc):
        ln, msg = sorted(tikzdoc.errors.items())[0]
        error = ("Error on line {errorln}: " if ln else "Error: ") + msg
        return self.extract_code(
            self.request(text=self.repair_prompt.format(error=error, code=tikzdoc.code))
        )

    def refine(self, ref_img, initial=None, iterations=None):
        current = initial

        try:
            # use iterations+1 because the initial generation is not a refinement
            while iterations is None or (iterations:=iterations-1)+1 >= 0:
                if current is None:
                    current = self.generate(ref_img=ref_img)
                elif current.is_rasterizable:
                    current = self.generate(
                        input_img=current.rasterize(),
                        input_label="input",
                        ref_img=ref_img,
                        ref_label="reference" if self.input_type == "picture" else self.input_type,
                        tikzdoc=current
                    )
                else:
                    current = self.repair(current)
                yield current
        finally:
            self.state = list()


class OpenAIVisualSelfRefiner(AbstractVisualSelfRefiner):
    client_class = OpenAI

    def encode(self, img, label=None):
        if img:
            if label:
                ImageDraw.Draw(img:=img.copy()).text((1, 1), label.title(), (0, 0, 0))
            return ({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{self.b64encode(img)}"}},)
        return ()

    def create(self, text, input_img=None, ref_img=None, input_label=None, ref_label=None):
        start, response = time(), self.client.chat.completions.create(
            model="gpt-4-vision-preview" if input_img or ref_img else "gpt-4",
            **self.hyper_params,
            messages=[
                {"role": "system", "content": self.system_role},
                {
                    "role": "user",
                    "content": [
                        *self.encode(input_img, input_label),
                        *self.encode(ref_img, ref_label),
                        {"type": "text", "text": text},
                    ],
                },
            ],
        )
        return dict(
            text=response.choices[0].message.content,
            usage=response.usage.completion_tokens,
            time=time()-start
        )


class AnthropicVisualSelfRefiner(AbstractVisualSelfRefiner):
    client_class = Anthropic

    def encode(self, img, label=None):
        heading = tuple()
        if label:
            heading = ({"type": "text", "text": f"{label.title()}:"},)
        if img:
            return heading + (
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": self.b64encode(img),
                    },
                },
            )
        return ()

    def create(self, text, input_img=None, ref_img=None, input_label=None, ref_label=None):
        start, message = time(), self.client.messages.create(
            model="claude-3-opus-20240229",
            **self.hyper_params,
            messages=[
                {
                    "role": "user",
                    "content": [
                        *self.encode(input_img, input_label),
                        *self.encode(ref_img, ref_label),
                        {
                            "type": "text",
                            "text": text
                        }
                    ],
                }
            ],
        )

        return dict(
            text=message.content[0].text,
            usage=message.usage.output_tokens,
            time=time()-start
        )


def parse_args():
    argument_parser = ArgumentParser(
        description="Generate tikzpictures with GPT-4 and Claude-2 (needs openai and anthropic packages)"
    )
    argument_parser.add_argument(
        "--model",
        choices=["gpt-4-vision", "claude-3-opus"],
        help="specify which llm to use",
    )
    argument_parser.add_argument(
        "--output",
        help="where to save generated text",
    )
    argument_parser.add_argument(
        "--testset",
        required=True,
        help="path to the datikz test set (in parquet format)",
    )
    argument_parser.add_argument(
        "--use_sketches",
        action="store_true",
        help="condition model on sketches instead of images",
    )
    argument_parser.add_argument(
        "--timeout",
        type=int,
        help="minimum time to run self-refine in seconds",
    )
    argument_parser.add_argument(
        "--iterations",
        default=4,
        type=int,
        help="maximum amount of self-refine iterations",
    )
    argument_parser.add_argument(
        "--early_stopping",
        action="store_true",
        help="if TikZ code compiles stop before all iterations are exhausted"
    )
    return argument_parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    refiner = AnthropicVisualSelfRefiner() if args.model == "claude-3-opus" else OpenAIVisualSelfRefiner()
    if isfile(args.output):
        with open(args.output) as f:
            all_predictions = load(f)
    else:
        all_predictions = list()
    try:
        dataset = load_dataset("parquet", data_files=args.testset, split="train").sort("caption") # type: ignore
        for idx, img in enumerate(tqdm(dataset['sketch' if args.use_sketches else 'image'])): # type: ignore
            if idx >= len(all_predictions):
                state, code, success = list(), list(), False
                while True:
                    state.append(refiner.state)
                    for tikz in refiner(img, iterations=args.iterations):
                        code.append(tikz.code)
                        success = success or tikz.is_rasterizable
                        runtime = sum(output['time'] for state in state for output in state)
                        if success and (not args.timeout or runtime >= args.timeout):
                            all_predictions.append(dict(code=code, state=state))
                            if args.early_stopping or args.timeout:
                                break
                    else:
                        continue
                    break
    finally:
        with open(args.output, "w") as f:
            dump([[p.code for p in ps] for ps in all_predictions], f)
