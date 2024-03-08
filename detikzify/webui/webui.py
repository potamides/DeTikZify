from functools import lru_cache
from inspect import signature
from multiprocessing.pool import ThreadPool
from os.path import basename
from sys import float_info
from tempfile import NamedTemporaryFile
from textwrap import dedent
from typing import Dict

from PIL import Image
import gradio as gr
from torch import bfloat16, float16
from torch.cuda import is_available as is_cuda_available, is_bf16_supported
from transformers import TextIteratorStreamer
from transformers.utils import is_flash_attn_2_available

from detikzify.infer import DetikzifyPipeline, TikzDocument
from detikzify.model import load

MODELS = {
    basename(model): model
    for model in [
        "nllg/detikzify-ds-1.3b",
        "nllg/detikzify-ds-5.7b",
        "nllg/detikzify-ds-6.7b",
        "nllg/detikzify-tl-1.1b",
        "nllg/detikzify-cl-7b",
    ]
}

# https://github.com/gradio-app/gradio/issues/3202#issuecomment-1741571240
# https://github.com/gradio-app/gradio/issues/2666#issuecomment-1651127149
# https://stackoverflow.com/a/64033350
CSS = """
    #input-image {
        flex-grow: 1;
    }
    #output-code {
        flex-grow: 1;
        height: 50vh;
        overflow-y: clip !important;
        scrollbar-width: thin !important;
    }
    #output-code .cm-scroller {
        flex-grow: 1;
    }
    #output-code .cm-gutters {
        position: relative !important;
    }
    #output-image {
        flex-grow: 1;
        height: 50vh;
        overflow-y: auto !important;
        scrollbar-width: thin !important;
    }
    #output-image .image-container {
       width: 100%;
       height: 100%;
    }
    #outputs .tabs {
        display: flex;
        flex-direction: column;
        flex-grow: 1;
    }
    #outputs .tabitem[style="display: block;"] {
        flex-grow: 1;
        display: flex !important;
    }
    #outputs .gap {
        flex-grow: 1;
    }
    #outputs .form {
        flex-grow: 1 !important;
    }
    #outputs .form > :last-child{
        flex-grow: 1;
    }
"""

@lru_cache(maxsize=1)
def cached_load(*args, **kwargs):
    gr.Info("Instantiating model. This could take a while...") # type: ignore
    return load(*args, **kwargs)

def inference(
    model_name: str,
    image: Dict[str, Image.Image],
    temperature: float,
    top_p: float,
    top_k: int,
    preprocess: bool,
):
    model, tokenizer = cached_load(
        base_model=model_name,
        device_map="auto",
        torch_dtype=bfloat16 if is_cuda_available() and is_bf16_supported() else float16,
        attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
    )
    streamer = TextIteratorStreamer(
        tokenizer=tokenizer.text,
        skip_prompt=True,
        skip_special_tokens=True
    )
    generate = DetikzifyPipeline(
        model=model,
        tokenizer=tokenizer,
        streamer=streamer,
        temperature=temperature + float_info.epsilon,
        top_p=top_p,
        top_k=top_k,
    )

    thread = ThreadPool(processes=1)
    async_result = thread.apply_async(generate, kwds=dict(image=image['composite'], preprocess=preprocess))

    generated_text = ""
    for new_text in streamer:
        generated_text += new_text
        yield generated_text, None, False
    yield async_result.get().code, None, True

def tex_compile(
    code: str,
    timeout: int,
    rasterize: bool
):
    tikzdoc = TikzDocument(code, timeout=timeout)
    if not tikzdoc.is_rasterizable:
        if tikzdoc.compiled_with_errors:
            raise gr.Error("TikZ code did not compile!") # type: ignore
        else:
            gr.Warning("TikZ code compiled to an empty image!") # type: ignore
    elif tikzdoc.compiled_with_errors:
        gr.Warning("TikZ code compiled with errors!") # type: ignore

    if rasterize:
        yield tikzdoc.rasterize()
    else:
        with NamedTemporaryFile(suffix=".svg", buffering=0) as tmpfile:
            if pdf:=tikzdoc.pdf:
                tmpfile.write(pdf[0].get_svg_image().encode())
            yield tmpfile.name if pdf else None

def check_inputs(image: Dict[str, Image.Image]):
    if image['composite'].getcolors(1) is not None:
        raise gr.Error("Image has no content!")

def get_banner():
    return dedent('''\
    # DeTi*k*Zify: Sketch-Guided Synthesis of Scientific Vector Graphics with Ti*k*Z

    <p>
      <a style="display:inline-block" href="https://github.com/potamides/DeTikZify">
        <img src="https://img.shields.io/badge/View%20on%20GitHub-green?logo=github&labelColor=gray" alt="View on GitHub">
      </a>
      <a style="display:inline-block" href="https://arxiv.org/abs/2310.00367">
        <img src="https://img.shields.io/badge/View%20on%20arXiv-B31B1B?logo=arxiv&labelColor=gray" alt="View on arXiv">
      </a>
      <a style="display:inline-block" href="https://colab.research.google.com/drive/14S22x_8VohMr9pbnlkB4FqtF4n81khIh">
        <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
      </a>
      <a style="display:inline-block" href="https://huggingface.co/spaces/nllg/DeTikZify">
        <img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg" alt="Open in HF Spaces">
      </a>
    </p>
    ''')

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

def build_ui(
    model=list(MODELS)[0],
    lock=False,
    rasterize=False,
    force_light=False,
    lock_reason="Duplicate this space to be able to change this value.",
    timeout=60,
):
    theme = make_light(gr.themes.Soft()) if force_light else gr.themes.Soft()
    with gr.Blocks(css=CSS, theme=theme, title="DeTikZify") as demo: # type: ignore
        if force_light: make_light(demo)
        gr.Markdown(get_banner())
        with gr.Row(variant="panel"):
            with gr.Column():
                sketchpad = gr.ImageEditor(
                    sources=["upload", "clipboard"],
                    elem_id="input-image",
                    type="pil",
                    label="Sketchpad",
                    show_label=False,
                    crop_size="1:1",
                    brush=gr.Brush(
                        colors=["black", "gray", "red", "green", "blue"],
                        default_size=1
                    ),
                )
                with gr.Row():
                    run_btn = gr.Button("Run", variant="primary")
                    stop_btn = gr.Button("Stop", variant="stop")
                    clear_btn = gr.ClearButton(sketchpad, variant="stop")
            with gr.Column(elem_id="outputs"):
                with gr.Tabs() as tabs:
                    with gr.TabItem(label:="TikZ Code", id=0):
                        tikz_code = gr.Code(label=label, show_label=False, interactive=False, elem_id="output-code")
                    with gr.TabItem(label:="Compiled Image", id=1):
                        result_image = gr.Image(label=label, show_label=False, show_share_button=rasterize, elem_id="output-image")
                    clear_btn.add([tikz_code, result_image])
        with gr.Accordion(label="Model Settings", open=False):
            base_model = gr.Dropdown(
                label="Base Model",
                allow_custom_value=True,
                info=lock_reason if lock else None,
                choices=list(({model: model} | MODELS).items()),
                value=MODELS.get(model, model),
                interactive=not lock,
            )
            with gr.Accordion(label="Advanced"):
                temperature = gr.Slider(minimum=0, maximum=2, step=0.1, value=0.7, label="Temperature")
                top_p = gr.Slider(minimum=0, maximum=1, step=0.05, value=0.9, label="Top-P")
                top_k = gr.Slider(minimum=0, maximum=100, step=5, value=0, label="Top-K")
                preprocess = gr.Checkbox(value=True, label="Preprocess image", info="Trim whitespace and pad image to square.")

        events = list()
        finished = gr.Textbox(visible=False) # hack to cancel compile on canceled inference
        for listener in [run_btn.click]:
            generate_event = listener(
                check_inputs,
                inputs=[sketchpad],
                queue=False
            ).success(
                lambda: gr.Tabs(selected=0),
                outputs=tabs, # type: ignore
                queue=False
            ).then(
                inference,
                inputs=[base_model, sketchpad, temperature, top_p, top_k, preprocess],
                outputs=[tikz_code, result_image, finished]
            )

            def tex_compile_if_finished(finished, *args):
                yield from (tex_compile(*args, timeout=timeout, rasterize=rasterize) if finished == "True" else [])

            compile_event = generate_event.then(
                lambda finished: gr.Tabs(selected=1) if finished == "True" else gr.Tabs(),
                inputs=finished,
                outputs=tabs, # type: ignore
                queue=False
            ).then(
                tex_compile_if_finished,
                inputs=[finished, tikz_code],
                outputs=result_image
            )
            events.extend([generate_event, compile_event])

        # scroll with output
        tikz_code.change(
            fn=None,
            js="() => document.getElementById('output-code').querySelector('.cm-gutters').scrollIntoView(false)"
        )
        for btn in [clear_btn, stop_btn]:
            btn.click(fn=None, cancels=events, queue=False)
        return demo
