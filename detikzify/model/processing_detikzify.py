# Copyright 2024 The HuggingFace Inc. team.
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
# https://github.com/andimarafioti/transformers/commit/9b09c481c4c39a172156b7ee44dc642160d0e809

from typing import TYPE_CHECKING, List, Optional, Union, Unpack

from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput, is_valid_image, load_image
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin
from transformers.tokenization_utils_base import AddedToken, BatchEncoding, TextInput
from transformers.utils import logging, is_remote_url


if TYPE_CHECKING:
    from transformers.tokenization_utils_base import PreTokenizedInput

logger = logging.get_logger(__name__)


def is_image_or_image_url(elem):
    return is_remote_url(elem) or is_valid_image(elem)


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
    image_processor_class = "DetikzifyImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, image_processor, tokenizer=None, image_seq_len: int = 300, image_token: str = "<|reserved_special_token_2|>", **kwargs):
        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")

        self.image_token = AddedToken(image_token, normalized=False, special=True)
        self.image_seq_len = image_seq_len
        tokenizer.add_special_tokens(dict(
            additional_special_tokens=[
                self.image_token,
            ]
        ))

        super().__init__(image_processor, tokenizer, **kwargs)

    def __call__(
        self,
        images: Union[ImageInput, List[ImageInput], List[List[ImageInput]]] = None,
        text: Union[TextInput, "PreTokenizedInput", List[TextInput], List["PreTokenizedInput"]] = None,
        audio=None,
        videos=None,
        image_seq_len: Optional[int] = None,
        **kwargs: Unpack[DetikzifyProcessorKwargs],
    ) -> BatchEncoding:
        if text is None and images is None:
            raise ValueError("You must provide either `text` or `images`.")

        output_kwargs = self._merge_kwargs(
            DetikzifyProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        # Temporary fix for "padding_side" in init_kwargs
        output_kwargs["text_kwargs"].pop("padding_side", None)

        image_seq_len = image_seq_len if image_seq_len is not None else self.image_seq_len

        n_images_in_text = []
        n_images_in_images = []
        inputs = BatchFeature()

        if images is not None:
            if is_image_or_image_url(images):
                images = [[images]]
            elif isinstance(images, list) and is_image_or_image_url(images[0]):
                images = [images]
            elif (
                not isinstance(images, list)
                and not isinstance(images[0], list)
                and not is_image_or_image_url(images[0][0])
            ):
                raise ValueError(
                    "Invalid input images. Please provide a single image or a list of images or a list of list of images."
                )
            n_images_in_images = [len(sample) for sample in images]

            # Load images if they are URLs
            images = [[load_image(im) if is_remote_url(im) else im for im in sample] for sample in images]

            image_inputs = self.image_processor(images, **output_kwargs["images_kwargs"])
            inputs.update(image_inputs)

        if text is not None:
            if isinstance(text, str):
                text = [text]
            elif not isinstance(text, list) and not isinstance(text[0], str):
                raise ValueError("Invalid input text. Please provide a string, or a list of strings")

            image_token = self.image_token.content

            prompt_strings = []
            for sample in text:
                n_images_in_text.append(sample.count(image_token))

                split_sample = sample.split(image_token)
                if len(split_sample) == 1:
                    raise ValueError("The image token should be present in the text.")

                # Expand image token to length `image_seq_len`
                prompt_strings.append((image_token * image_seq_len).join(split_sample))

            text_inputs = self.tokenizer(text=prompt_strings, **output_kwargs["text_kwargs"])
            inputs.update(text_inputs)

            if n_images_in_images != n_images_in_text:
                raise ValueError(
                    f"The number of images in the text {n_images_in_text} and images  {n_images_in_images} should be the same."
                )

        return inputs

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))