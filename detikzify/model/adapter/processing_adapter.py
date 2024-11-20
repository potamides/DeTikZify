from typing import List, TYPE_CHECKING, Union, Unpack

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


class AdapterProcessor(ProcessorMixin):
    attributes = ["processor", "tokenizer"]
    processor_class = ("ProcessorMixin", "ImageProcessingMixin")
    tokenizer_class = "AutoTokenizer"

    def __init__(self, processor, tokenizer=None, **kwargs):
        if processor is None:
            raise ValueError("You need to specify a `processor`.")
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")
        super().__init__(processor, tokenizer, **kwargs)

    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        images: ImageInput = None,
        **kwargs: Unpack[ProcessingKwargs],
    ) -> BatchEncoding:
        if images is None:
            raise ValueError("`images` are expected as arguments to a `AdapterProcessor` instance.")
        else:
            images = make_list_of_images(images)
        if text is not None:
            if isinstance(text, str):
                text = [text]
            if len(images) != len(text):
                raise ValueError(
                    f"Received {len(images)} images for {len(text)} prompts. Each prompt should be associated with an image."
                )
            text_kwargs = kwargs.pop("text_kwargs", {})
            text_inputs = {f"adapter_{key}": value for key, value in self.tokenizer(text=text, **kwargs, **text_kwargs).items()}
        else:
            text_inputs = dict()

        images_kwargs = kwargs.pop("images_kwargs", {})
        image_inputs = self.processor(images=images, **kwargs, **images_kwargs)
        return BatchFeature(data={**image_inputs, **text_inputs})

    def batch_decode(self, *args, **kwargs):
        return self.processor.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.processor.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        processor_input_names = self.processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + processor_input_names))
