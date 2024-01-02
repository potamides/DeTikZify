"""
Images from the Paper2Fig100k dataset.
"""
from itertools import chain
from json import load
from os.path import basename
import tarfile

from datasets import Features, Image, Sequence, Value, builder
from datasets.info import DatasetInfo
from datasets.splits import Split, SplitGenerator

from detikzify.util import expand

class Paper2FigConfig(builder.BuilderConfig):
    """BuilderConfig for Paper2Fig."""

    def __init__(self, size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = size
        self.archive = "https://zenodo.org/records/7299423/files/Paper2Fig100k.tar.gz"

class Paper2Fig(builder.GeneratorBasedBuilder):
    """The Paper2Fig100k dataset in the format DeTikZify expects (everything is training data)."""

    BUILDER_CONFIG_CLASS = Paper2FigConfig

    def _info(self):
        features = {
            "caption": Value("string"),
            "mention": Sequence(Sequence(Value("string"))),
            "ocr": Sequence(Value("string")),
            "image": Image(),
        }
        return DatasetInfo(
            description=str(__doc__),
            features=Features(features),
        )

    def _split_generators(self, dl_manager):
        archive = dl_manager.download(self.config.archive) # type: ignore
        return [SplitGenerator(name=str(Split.TRAIN), gen_kwargs=dict(archive=archive))]

    def _generate_examples(self, archive):
        with tarfile.open(archive) as tf:
            metadata = dict()
            for figdata in chain.from_iterable(load(tf.extractfile(f)) for f in tf if f.name.endswith(".json")): # type: ignore
                metadata[figdata.pop("figure_id")] = figdata
            for idx, member in enumerate(tf):
                if member.name.endswith(".png"):
                    figure_id = basename(member.name).removesuffix(".png")
                    figdata = metadata[figure_id]
                    yield idx, dict(
                        caption=figdata["captions"][0],
                        mention=[figdata["captions"][1:]],
                        ocr=[result['text'] for result in figdata['ocr_result']['ocr_result']],
                        image=expand(tf.extractfile(member), self.config.size), # type: ignore
                    )
