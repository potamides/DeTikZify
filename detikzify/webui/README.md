# Web UI
The web UI of DeTi*k*Zify requires [TeX Live
2023](https://www.tug.org/texlive), [ghostscript](https://www.ghostscript.com),
and [poppler](https://poppler.freedesktop.org). You can launch it by running
`python -m detikzify.webui`. It comes with a command line interface. With the
`--share` flag, for example, you can create a shareable link. Checkout `--help`
for a full list of supported options. As scientific figures usually use black
fonts on a white background, it is best to use the web UI in light mode. This
can be enforced by using the `--light` flag. If [FlashAttention-2](
https://huggingface.co/docs/transformers/en/perf_infer_gpu_one?install=NVIDIA#flashattention-2)
is installed, it is picked up automatically and should boost inference speeds.

## Usage Tips
**Visual Prompting** Creating sketches for DeTi*k*Zify (or providing any input
images) shares many similarities with the process of [prompting large language
models](https://en.wikipedia.org/wiki/Prompt_engineering). If DeTi*k*Zify
struggles to comprehend your intent, consider "rephrasing" your input. This
could mean simplifying your sketches or focusing more on the key issue at hand.
In
[this](https://github.com/potamides/DeTikZify/assets/53401822/2819ebca-81f6-4173-8809-0b4255d3e976)
particular instance, for example, we attempted to "prompt" DeTikZify to align
characters diagonally around an equal sign, but it was unsuccessful even after
many simulations. However, upon adjusting the input (by reducing the stroke
width and using more easily recognizable characters) we achieved the [intended
output](https://github.com/potamides/DeTikZify/assets/53401822/c8ecfbff-d22e-41d5-8f73-e0cfafe88690)
after only one simulation.

**Image Editor** You can draw sketches in the integrated image editor, but its
feature set is quite limited. If you are not satisfied with the synthesized
Ti*k*Z programs, try drawing more elaborate sketches in an editor of your
choice (perhaps with graphics primitives) and upload them into the UI.
Alternatively, experimenting with line thickness and/or colors in the
integrated editor might also help.

**Input Postprocessing** Please note that all input images are cropped to the
smallest square around their content and then resized to the resolution
DeTi*k*Zify expects. If you leave large margins this means that DeTi*k*Zify
might perceive your input differently from how you intended (e.g., by drawing
thicker axes). As a rule of thumb, always try to fill as much of the canvas as
possible.

**Input Complexity** If you provide very complex sketches (or figures) and are
not satisfied with the results, you can also try segmenting (or simplifying)
your input and letting DeTi*k*Zify synthesize the individual pieces
independently. This has the advantage that the results will probably be better,
and the disadvantage that you will have to modify and assemble the pieces
yourself.

**Source Code Artifacts** Due to the way we preprocess our
[arXiv.org](https://arxiv.org) data, the preambles of the extracted Ti*k*Z
programs sometimes include packages that are not used inside the `tikzpicture`
environments, and the DeTi*k*Zify models pick up on this behavior. While this
does not hinder compilation in any way, we still recommend everyone to check
the generated preambles and clean them up, if necessary.

**Accuracy-Efficiency Trade-Off** We noticed that lower values for temperatures
and top-p (nucleus) values force DeTi*k*Zify to generate Ti*k*Z programs that
follow the input images more closely, at the expense of generating more
compile-time errors. We pick sensible defaults that aim to balance these two
aspects, but you might want to try to tune these parameters yourself.

**External Graphics** In DaTi*k*Z<sub>v2</sub>, we replace any externally
included graphics in the `tikzpicture`  environments with the [example
image](https://mirrors.ctan.org/macros/latex/contrib/mwe/example-image.pdf)
placeholder from the [mwe](http://www.ctan.org/pkg/mwe) package. So if you want
to generate code with placeholders for your own external graphics, just draw
that example image.
