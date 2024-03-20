from importlib import import_module
from typing import Any

from .patchsim import * # this metric is used by MCTS, so it is not optional

__all__ = ["PatchSim", "CrystalBLEU", "KernelInceptionDistance", "TexEditDistance", "DreamSim"] # type: ignore

# lazy import optional metrics (https://peps.python.org/pep-0562/)
def __getattr__(name) -> Any:
    def load(metric):
        return getattr(import_module("." + metric, __name__), name)
    try:
        match name:
            case "CrystalBLEU":
                return load("crystalbleu")
            case "KernelInceptionDistance":
                return load("kid")
            case "TexEditDistance":
                return load("eed")
            case "DreamSim":
                return load("dreamsim")
    except ImportError:
        raise ValueError(
            "Missing dependencies: "
            "Install this project with the [evaluate] feature name!"
        )
    return import_module("." + name, __name__)
