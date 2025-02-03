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

from ...util import DUMMY_IMAGE

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
        text: Optional[Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]] = None,
        images: Optional[ImageInput] = None,
        **kwargs: Unpack[ProcessingKwargs],
    ) -> BatchEncoding:
        if images is None and text is None:
            raise ValueError("Either `images` or `text` (or both) are expected as arguments to an `AdapterProcessor` instance.")

        text_kwargs, images_kwargs = kwargs.pop("text_kwargs", {}), kwargs.pop("images_kwargs", {})

        if text is None:
            text_inputs = dict()
        else:
            text = [text] if isinstance(text, str) else text
            text_inputs = {f"adapter_{key}": value for key, value in self.tokenizer(text=text, **kwargs, **text_kwargs).items()}
            if getattr(self.processor, "model_expects_text", False):
                images_kwargs.update(text=text, add_bos_token=True)
        if images is None:
            image_inputs = self.processor(images=len(text) * [DUMMY_IMAGE], **kwargs, **images_kwargs)
            image_inputs = dict((k, image_inputs[k]) for k in ["input_ids", "attention_mask"] if k in image_inputs)
        else:
            images = make_list_of_images(images)
            image_inputs = self.processor(images=images, **kwargs, **images_kwargs)

        if text is not None and images is not None and len(images) != len(text):
            raise ValueError(
                f"Received {len(images)} images for {len(text)} prompts. Each prompt should be associated with an image."
            )

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
