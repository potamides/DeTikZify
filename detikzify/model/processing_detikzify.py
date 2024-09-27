# Copyright 2024 the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adapted from
# https://github.com/huggingface/transformers/commit/e1b150862e66e16acf951edfa13206ffcd1032be

from typing import List, Optional, TYPE_CHECKING, Union, Unpack

from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput, make_list_of_images
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin
from transformers.tokenization_utils_base import (
    BatchEncoding,
    PreTokenizedInput,
    TextInput,
)
from transformers.utils import logging


if TYPE_CHECKING:
    from transformers.tokenization_utils_base import PreTokenizedInput

logger = logging.get_logger(__name__)


class DetikzifyProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "add_special_tokens": True,
            "padding": False,
            "is_split_into_words": False,
        },
    }


class DetikzifyProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "SiglipImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, image_processor, tokenizer=None, image_seq_len: int = 300, image_token: str = "<|reserved_special_token_2|>", **kwargs):
        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")
        if image_token not in tokenizer.vocab:
            raise ValueError(f"{image_token} needs to be added to the `tokenizer` vocabulary.")

        self.image_token = image_token
        self.image_seq_len = image_seq_len

        super().__init__(image_processor, tokenizer, **kwargs)

    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        images: ImageInput = None,
        image_seq_len: Optional[int] = None,
        add_bos_token: bool = None,
        add_eos_token: bool = None,
        **kwargs: Unpack[DetikzifyProcessorKwargs],
    ) -> BatchEncoding:
        output_kwargs = self._merge_kwargs(
            DetikzifyProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        # Temporary fix for "padding_side" in init_kwargs
        output_kwargs["text_kwargs"].pop("padding_side", None)

        if images is None:
            raise ValueError("`images` are expected as arguments to a `DetikzifyProcessor` instance.")
        else:
            images = make_list_of_images(images)
        if text is None:
            text = len(images) * [""]
        elif isinstance(text, str):
            text = [text]
        if len(images) != len(text):
            raise ValueError(
                f"Received {len(images)} images for {len(text)} prompts. Each prompt should be associated with an image."
            )

        prompt_strings = []
        for prompt in text:
            assert self.image_token not in prompt, "Image tokens are added by the processor!"
            if add_bos_token:
                prompt = self.tokenizer.bos_token + prompt
            if add_eos_token:
                prompt += self.tokenizer.eos_token
            image_seq_len = image_seq_len if image_seq_len is not None else self.image_seq_len
            prompt_strings.append((self.image_token * image_seq_len) + prompt)

        image_inputs = self.image_processor(images=images, **output_kwargs["images_kwargs"])
        text_inputs = self.tokenizer(text=prompt_strings, **output_kwargs["text_kwargs"])

        return BatchFeature(data={**image_inputs, **text_inputs})

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))
