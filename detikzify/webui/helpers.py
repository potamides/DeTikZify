from functools import cache, lru_cache
from inspect import signature
from operator import itemgetter
from os import fdopen
from tempfile import mkstemp

import gradio as gr

from ..infer import TikzDocument
from ..model import load

def to_svg(
    tikzdoc: TikzDocument,
    build_dir: str
):
    if not tikzdoc.is_rasterizable:
        if tikzdoc.compiled_with_errors:
            raise gr.Error("TikZ code did not compile!")
        else:
            gr.Warning("TikZ code compiled to an empty image!")
    elif tikzdoc.compiled_with_errors:
        gr.Warning("TikZ code compiled with errors!")

    fd, path = mkstemp(dir=build_dir, suffix=".svg")
    with fdopen(fd, "w") as f:
        if pdf:=tikzdoc.pdf:
            f.write(pdf[0].get_svg_image())
    return path if pdf else None

# https://stackoverflow.com/a/50992575
def make_ordinal(n):
    n = int(n)
    if 11 <= (n % 100) <= 13:
        suffix = 'th'
    else:
        suffix = ['th', 'st', 'nd', 'rd', 'th'][min(n % 10, 4)]
    return str(n) + suffix

class MctsOutputs(set):
    def __init__(self, build_dir, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.build_dir, self.svgmap, self.fails = build_dir, dict(), 0

    def add(self, score, tikzdoc): # type: ignore
        if (score, tikzdoc) not in self:
            try:
                 svg = to_svg(tikzdoc, build_dir=self.build_dir)
                 super().add((score, tikzdoc))
                 self.svgmap[tikzdoc] = svg
            except gr.Error:
                gr.Warning("TikZ code did not compile, discarding output!")
                if len(self): self.fails += 1
        elif len(self): self.fails += 1

    @property
    def programs(self):
        return [tikzdoc.code for _, tikzdoc in sorted(self, key=itemgetter(0), reverse=True)]

    @property
    def images(self):
        return [
            (self.svgmap[tikzdoc], make_ordinal(idx))
            for idx, (_, tikzdoc) in enumerate(sorted(self, key=itemgetter(0), reverse=True), 1)
        ]

    @property
    def first_success(self):
        return len(self) == 1 and not self.fails

def make_light(stylable):
    """
    Patch gradio to only contain light mode colors.
    """
    if isinstance(stylable, gr.themes.Base): # remove dark variants from the entire theme
        params = signature(stylable.set).parameters
        colors = {color: getattr(stylable, color.removesuffix("_dark")) for color in dir(stylable) if color in params}
        return stylable.set(**colors)
    elif isinstance(stylable, gr.Blocks): # also handle components which do not use the theme (e.g. modals)
        stylable.load(
            fn=None,
            js="() => document.querySelectorAll('.dark').forEach(el => el.classList.remove('dark'))"
        )
        return stylable
    else:
        raise ValueError

@lru_cache(maxsize=1)
def cached_load(*args, **kwargs):
    gr.Info("Instantiating model. This could take a while...")
    return load(*args, **kwargs)

@cache
def info_once(message):
    gr.Info(message)

class GeneratorLock:
    """
    Ensure that only one instance of a given generator is active.
    Useful when a previous invocation was canceled.
    """
    def __init__(self, gen_func):
        self.gen_func = gen_func
        self.generator = None

    def generate(self, *args, **kwargs):
        if self.generator:
            if self.generator.gi_running:
                return # somehow we can end up here
            self.generator.close()
        self.generator = self.gen_func(*args, **kwargs)
        yield from self.generator

    def __call__(self, *args, **kwargs):
        yield from self.generate(*args, **kwargs)
