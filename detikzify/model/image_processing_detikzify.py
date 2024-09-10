# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature
from transformers.image_transforms import (
    PaddingMode,
    pad,
    rescale,
    to_channel_dimension_format,
)
from transformers.image_utils import (
    ChannelDimension,
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ImageInput,
    PILImageResampling,
    get_image_size,
    infer_channel_dimension_format,
    is_jax_tensor,
    is_scaled_image,
    is_tf_tensor,
    is_torch_tensor,
    is_valid_image,
    to_numpy_array,
    valid_images,
    validate_preprocess_arguments,
)
from transformers.utils import TensorType, is_vision_available, logging
from transformers.utils.import_utils import (
    is_flax_available,
    is_tf_available,
    is_torch_available,
)


if is_torch_available():
    import torch

if is_tf_available():
    import tensorflow as tf

if is_flax_available():
    import jax.numpy as jnp

logger = logging.get_logger(__name__)


if is_vision_available():
    import PIL
    from PIL import Image, ImageChops


def get_resize_output_image_size(
    height: int, width: int, min_len: Optional[int] = 1, max_len: Optional[int] = None
) -> Tuple[int, int]:
    max_len = max(height, width) if max_len is None else max_len
    aspect_ratio = width / height

    if width >= height:
        width = max_len
        height = int(width / aspect_ratio)
        if height % 2 != 0:
            height += 1
    elif height > width:
        height = max_len
        width = int(height * aspect_ratio)
        if width % 2 != 0:
            width += 1

    # Avoid resizing to a size smaller than min_len
    height = max(height, min_len)
    width = max(width, min_len)
    return height, width


# Copied from transformers.models.idefics2.image_processing_idefics2.make_list_of_images
def make_list_of_images(images: ImageInput) -> List[List[np.ndarray]]:
    # If it's a single image, convert it to a list of lists
    if is_valid_image(images):
        images = [[images]]
    # If it's a list of images, it's a single batch, so convert it to a list of lists
    elif isinstance(images, (list, tuple)) and len(images) > 0 and is_valid_image(images[0]):
        images = [images]
    # If it's a list of batches, it's already in the right format
    elif (
        isinstance(images, (list, tuple))
        and len(images) > 0
        and isinstance(images[0], (list, tuple))
        and is_valid_image(images[0][0])
    ):
        pass
    else:
        raise ValueError(
            "Invalid input type. Must be a single image, a list of images, or a list of batches of images."
        )
    return images


def get_max_height_width(
    images_list: List[List[np.ndarray]], input_data_format: Optional[Union[str, ChannelDimension]] = None
) -> List[int]:
    if input_data_format is None:
        input_data_format = infer_channel_dimension_format(images_list[0][0])

    max_height = max_width = float("-inf")
    for images in images_list:
        for image in images:
            height, width = get_image_size(image, channel_dim=input_data_format)
            max_height = max(height, max_height)
            max_width = max(width, max_width)
    return 2 * (max(max_height, max_width),)


# Copied from transformers.models.detr.image_processing_detr.make_pixel_mask
def make_pixel_mask(
    image: np.ndarray, output_size: Tuple[int, int], input_data_format: Optional[Union[str, ChannelDimension]] = None
) -> np.ndarray:
    input_height, input_width = get_image_size(image, channel_dim=input_data_format)
    mask = np.zeros(output_size, dtype=np.int64)
    mask[:input_height, :input_width] = 1
    return mask


# Custom to_pil_image function to support image_mode
def to_pil_image(
    image: Union[np.ndarray, "PIL.Image.Image", "torch.Tensor", "tf.Tensor", "jnp.ndarray"],
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
    image_mode: Optional[str] = None,
) -> "PIL.Image.Image":
    if isinstance(image, PIL.Image.Image):
        return image
    # Convert all tensors to numpy arrays before converting to PIL image
    if is_torch_tensor(image) or is_tf_tensor(image):
        image = image.numpy()
    elif is_jax_tensor(image):
        image = np.array(image)
    elif not isinstance(image, np.ndarray):
        raise ValueError("Input image type not supported: {}".format(type(image)))

    # If the channel has been moved to first dim, we put it back at the end.
    image = to_channel_dimension_format(image, ChannelDimension.LAST, input_data_format)

    # If there is a single channel, we squeeze it, as otherwise PIL can't handle it.
    image = np.squeeze(image, axis=-1) if image.shape[-1] == 1 else image
    image = image.astype(np.uint8)
    return PIL.Image.fromarray(image, mode=image_mode)


def convert_to_rgb(
    image: ImageInput,
    palette: Optional[PIL.ImagePalette.ImagePalette] = None,
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
) -> ImageInput:
    if not isinstance(image, PIL.Image.Image):
        mode = "P" if palette is not None else None
        # Custom to_pil_image function to support image_mode
        image = to_pil_image(image, image_mode=mode, input_data_format=input_data_format)
        if image.mode == "P" and palette is not None:
            image.putpalette(palette)

    image_rgba = image.convert("RGBA")
    background = Image.new("RGBA", image_rgba.size, (255, 255, 255))
    alpha_composite = Image.alpha_composite(background, image_rgba)
    alpha_composite = alpha_composite.convert("RGB")
    return np.array(alpha_composite)


class DetikzifyImageProcessor(BaseImageProcessor):
    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_convert_rgb: bool = True,
        do_trim: bool = True,
        do_resize: bool = True,
        size: Dict[str, int] = None,
        resample: PILImageResampling = PILImageResampling.LANCZOS,
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_pad: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.do_convert_rgb = do_convert_rgb
        self.do_trim = do_trim
        self.do_resize = do_resize
        self.size = size if size is not None else {"longest_edge": 420}
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else IMAGENET_STANDARD_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_STANDARD_STD
        self.do_pad = do_pad

    def trim(
        self,
        image: np.ndarray,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> np.ndarray:
        image_mode = "P" if image.ndim == 2 or image.shape[-1] == 1 else None
        image = to_pil_image(image, input_data_format=input_data_format, image_mode=image_mode)

        bg = Image.new(image.mode, image.size, (255, 255, 255))
        diff = ImageChops.difference(image, bg)
        trimmed_image = image.crop(bbox) if (bbox:=diff.getbbox()) else image

        trimmed_array = np.array(trimmed_image)
        if trimmed_array.ndim == 2:
            trimmed_array = np.expand_dims(trimmed_array, axis=-1)
        return trimmed_array

    def resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        resample: PILImageResampling = PILImageResampling.LANCZOS,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> np.ndarray:
        if "longest_edge" in size:
            size = get_resize_output_image_size(
                *get_image_size(image, channel_dim=input_data_format),
                max_len=size["longest_edge"]
            )
        elif "height" in size and "width" in size:
            size = (size["height"], size["width"])
        else:
            raise ValueError("size must be a dictionary with key 'longest_edge' or 'height' and 'width'.")

        image_mode = "P" if image.ndim == 2 or image.shape[-1] == 1 else None
        image = to_pil_image(image, input_data_format=input_data_format, image_mode=image_mode)
        resized_image = image.resize((size[1], size[0]), resample=resample)

        resized_array = np.array(resized_image)
        if resized_array.ndim == 2:
            resized_array = np.expand_dims(resized_array, axis=-1)
        return resized_array

    def pad(
        self,
        images: List[np.ndarray],
        constant_values: Union[float, Iterable[float]] = 1,
        data_format: Optional[ChannelDimension] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> BatchFeature:
        pad_size = get_max_height_width(images, input_data_format=input_data_format)
        batch_size = len(images)
        max_num_images = max(len(images_) for images_ in images)
        input_data_format = (
            infer_channel_dimension_format(images[0][0]) if input_data_format is None else input_data_format
        )

        if input_data_format == ChannelDimension.FIRST:
            n_channels = images[0][0].shape[0]
            empty_image = lambda size: np.zeros((n_channels, *size), dtype=np.uint8)
        elif input_data_format == ChannelDimension.LAST:
            n_channels = images[0][0].shape[-1]
            empty_image = lambda size: np.zeros((*size, n_channels), dtype=np.uint8)

        else:
            raise ValueError("Invalid channel dimension format.")

        padded_images_list = [[empty_image(pad_size) for _ in range(max_num_images)] for _ in range(batch_size)]
        for batch_idx in range(batch_size):
            for sample_idx, image in enumerate(images[batch_idx]):
                input_height, input_width = get_image_size(image, channel_dim=input_data_format)
                pad_height = pad_size[0] - input_height
                pad_width = pad_size[1] - input_width
                padding = (
                    (rounded:=round(pad_height/2), pad_height-rounded),
                    (rounded:=round(pad_width/2), pad_width-rounded)
                )
                padded_images_list[batch_idx][sample_idx] = pad(
                    image,
                    padding,
                    mode=PaddingMode.CONSTANT,
                    constant_values=constant_values,
                    data_format=input_data_format if data_format is None else data_format,
                    input_data_format=input_data_format,
                )

        return padded_images_list

    def preprocess(
        self,
        images: ImageInput,
        do_convert_rgb: Optional[bool] = None,
        do_trim: Optional[bool] = None,
        do_resize: Optional[bool] = None,
        size: Optional[Dict[str, int]] = None,
        resample: PILImageResampling = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_pad: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        input_data_format: Optional[ChannelDimension] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        crop_size: Optional[Dict[str, int]] = None,
    ):
        if crop_size is not None:
            logger.warning("crop_size is not used in DetikzifyImageProcessor.preprocess.")

        do_trim = do_trim if do_trim is not None else self.do_trim
        do_resize = do_resize if do_resize is not None else self.do_resize
        size = size if size is not None else self.size
        resample = resample if resample is not None else self.resample
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb
        do_pad = do_pad if do_pad is not None else self.do_pad

        images_list = make_list_of_images(images)

        if not valid_images(images_list[0]):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        # save the palettes for conversion to RGB
        palettes_list = [
            [im.getpalette() if isinstance(im, Image.Image) and im.mode == "P" else None for im in images]
            for images in images_list
        ]

        # All transformations expect numpy arrays.
        images_list = [[to_numpy_array(image) for image in images] for images in images_list]

        if is_scaled_image(images_list[0][0]) and do_rescale:
            logger.warning_once(
                "It looks like you are trying to rescale already rescaled images. If the input"
                " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
            )

        new_images_list = []
        for images in images_list:
            new_images = []
            for img in images:
                if img.ndim == 2:
                    img = np.expand_dims(img, axis=-1)
                new_images.append(img)
            new_images_list.append(new_images)
        images_list = new_images_list
        del new_images_list

        if input_data_format is None:
            # We assume that all images have the same channel dimension format.
            input_data_format = infer_channel_dimension_format(images_list[0][0], num_channels=(1, 3, 4))

        validate_preprocess_arguments(
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_resize=do_resize,
            size=size,
            resample=resample,
        )

        def _trim_resize(image):
            if do_trim:
                image = self.trim(image=image, input_data_format=input_data_format)
            if do_resize:
                image = self.resize(image=image, size=size, resample=resample, input_data_format=input_data_format)
            return image

        images_list = [[_trim_resize(image=image) for image in images] for images in images_list]
        # Trim and resize might already change the channel dimension, so we will recompute it
        input_data_format = infer_channel_dimension_format(images_list[0][0], num_channels=(1, 3, 4))

        if do_convert_rgb:
            images_list = [
                [convert_to_rgb(img, plt, input_data_format=input_data_format) for img, plt in zip(images, palettes)]
                for images, palettes in zip(images_list, palettes_list)
            ]

        if do_rescale:
            rescaled_images_array = []
            for image in images_list:
                rescaled_images_array.append([rescale(img, rescale_factor) for img in image])
            images_list = rescaled_images_array

        if do_normalize:
            images_list = [
                [
                    self.normalize(image=image, mean=image_mean, std=image_std, input_data_format=input_data_format)
                    for image in images
                ]
                for images in images_list
            ]

        if do_pad:
            images_list = self.pad(images_list, input_data_format=input_data_format)

        if data_format is not None:
            images_list = [
                [
                    to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)
                    for image in images
                ]
                for images in images_list
            ]

        data = {"pixel_values": np.array(images_list) if do_pad and return_tensors is not None else images_list }
        return BatchFeature(data=data, tensor_type=return_tensors)
