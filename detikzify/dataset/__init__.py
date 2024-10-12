from datasets.load import load_dataset as _load_dataset
from os.path import dirname, isdir, join

def load_dataset(path, *args, **kwargs):
    if isdir(local := join(dirname(__file__), path)):
        return _load_dataset(local, *args, trust_remote_code=True, **kwargs)
    return _load_dataset(path, *args, **kwargs)
