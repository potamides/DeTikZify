"""
The SciCap dataset, unified in a single train split.
"""

from json import load
from os import symlink
from os.path import basename, join
from subprocess import run
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from datasets import Features, Image, Sequence, Value, builder
from datasets.info import DatasetInfo
from datasets.splits import Split, SplitGenerator
from datasets.utils.hub import hf_hub_url

from detikzify.util import convert, expand

class SciCapConfig(builder.BuilderConfig):
    """BuilderConfig for SciCap."""

    def __init__(self, size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.repo_id = "CrowdAILab/scicap"
        self.size = size
        self.files = {
            "img": {
                (public:="img-split"): 10,
                (hidden:="img-hide_test"): 0
            },
            "text": {
                "train": public,
                "train-acl": public,
                "val": public,
                "public-test": public,
                "hide_test": hidden,
            }
        }


class SciCap(builder.GeneratorBasedBuilder):
    """The SciCap dataset in the format DeTikZify expects (everything is training data)."""

    BUILDER_CONFIG_CLASS = SciCapConfig

    def _info(self):
        features = {
            "caption": Value("string"),
            "mention": Sequence(Sequence(Value("string"))),
            "paragraph": Sequence(Value("string")),
            "ocr": Sequence(Value("string")),
            "image": Image(),
        }
        return DatasetInfo(
            description=str(__doc__),
            features=Features(features),
        )
    def _split_generators(self, dl_manager):
        with TemporaryDirectory(ignore_cleanup_errors=True) as tmpdirname:
            def dl(path):
                return dl_manager.download(hf_hub_url(self.config.repo_id, path)) # type: ignore

            def zip_dl(path, num_splits=0):
                paths = [f"{path}.zip"] + list(f"{path}.z{{:02d}}".format(i+1) for i in range(num_splits))
                downloaded = [dl(path) for path in paths]
                if num_splits:
                    output = join(tmpdirname, f"{path}-joined.zip")
                    for src, dst in zip(downloaded, paths):
                        symlink(src, join(tmpdirname, dst)) # type: ignore
                    run(["zip", "-FF", join(tmpdirname, paths[0]), "--out", output], check=True, capture_output=True)
                    return output
                else:
                    return downloaded[0]

            files_to_download = self.config.files # type: ignore
            img = {file:zip_dl(file, num_splits) for file, num_splits in files_to_download['img'].items()}
            text = {dl(f"{file}.json"):img[img_file] for file, img_file in files_to_download['text'].items()}

            yield SplitGenerator(name=str(Split.TRAIN), gen_kwargs={"shards": text})

    def _generate_examples(self, shards):
        idx = 0
        for path, image_zip in shards.items():
            with ZipFile(file=image_zip, mode='r') as zf:
                imagemap = {basename(name):name for name in zf.namelist()}
                with open(path) as f:
                    images, annotations = load(f).values()
                    for annotation, image in zip(annotations, images):
                        assert image["id"] == annotation['image_id']
                        with zf.open(imagemap[image['file_name']]) as img:
                            yield idx, dict(
                                caption=annotation.get("caption_no_index"),
                                mention=annotation.get("mention"),
                                paragraph=annotation.get("paragraph"),
                                ocr=image.get("ocr"),
                                image=convert(expand(img, self.config.size), "png")
                            )
                            idx += 1
