try:
    from .patchsim import *
    from .eed import *
    from .kid import *
    from .crystalbleu import *
except ImportError:
    raise ValueError(
        "Missing dependencies: "
        "Install this project with the [evaluate] feature name!"
    )
