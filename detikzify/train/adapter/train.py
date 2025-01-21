import os

from transformers.utils import logging

from .pretrain import train as pretrain

logger = logging.get_logger("transformers")

WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))

def train(*args, learning_rate: float = 5e-5, **kwargs):
    return pretrain(*args, learning_rate=learning_rate, **kwargs)
