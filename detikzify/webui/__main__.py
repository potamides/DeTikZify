from .webui import build_ui, MODELS
from argparse import ArgumentParser

def parse_args():
    argument_parser = ArgumentParser(
        description="Web UI for DeTikZify."
    )
    argument_parser.add_argument(
        "--model",
        default=list(MODELS)[0],
        help="Initially selected model.",
    )
    argument_parser.add_argument(
        "--lock",
        action="store_true",
        help="Whether to allow users to change the model or not.",
    )
    argument_parser.add_argument(
        "--lock_reason",
        default="Duplicate this space to be able to change this value.",
        help="Additional information why model selection is locked.",
    )
    argument_parser.add_argument(
        "--share",
        action="store_true",
        help="Whether to create a publicly shareable link for the interface.",
    )
    argument_parser.add_argument(
        "--rasterize",
        action="store_true",
        help= "Whether to rasterize the generated image before displaying it."
    )
    argument_parser.add_argument(
        "--force_light",
        action="store_true",
        help= "Whether to enforce light theme (useful for vector graphics with dark text)."
    )
    argument_parser.add_argument(
        "--timeout",
        default=120,
        type=int,
        help="Allowed timeframe for compilation.",
    )
    return vars(argument_parser.parse_args())

if __name__ == "__main__":
    args = parse_args()
    share = args.pop("share")
    build_ui(**args).queue().launch(share=share)
