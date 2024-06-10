from os.path import basename

BANNER = '''\
<h1>DeTi<i>k</i>Zify: Synthesizing Graphics Programs for Scientific Figures and Sketches with Ti<i>k</i>Z</h1>

<p>
  <a style="display:inline-block" href="https://arxiv.org/abs/2405.15306">
    <img src="https://img.shields.io/badge/View%20on%20arXiv-B31B1B?logo=arxiv&labelColor=gray" alt="View on arXiv">
  </a>
  <a style="display:inline-block" href="https://github.com/potamides/DeTikZify">
    <img src="https://img.shields.io/badge/View%20on%20GitHub-green?logo=github&labelColor=gray" alt="View on GitHub">
  </a>
  <a style="display:inline-block" href="https://huggingface.co/collections/nllg/detikzify-664460c521aa7c2880095a8b">
    <img src="https://img.shields.io/badge/View%20on%20Hugging%20Face-blue?labelColor=gray&logo=data:image/png;base64,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" alt="View on Hugging Face">
  </a>
  <a style="display:inline-block" href="https://colab.research.google.com/drive/1hPWqucbPGTavNlYvOBvSNBAwdcPZKe8F">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
  </a>
</p>
'''

MODELS = {
    basename(model): model
    for model in [
        "nllg/detikzify-ds-7b",
        "nllg/detikzify-cl-7b",
        "nllg/detikzify-ds-1.3b",
        "nllg/detikzify-tl-1.1b",
    ]
}

ALGORITHMS = {
    "mcts": "MCTS",
    "sampling": "Sampling"
}

# https://github.com/gradio-app/gradio/issues/3202#issuecomment-1741571240
# https://github.com/gradio-app/gradio/issues/2666#issuecomment-1651127149
# https://stackoverflow.com/a/64033350
CSS = """
    .input-image {
        flex-grow: 1;
    }
    .output-code {
        flex-grow: 1;
        height: 0vh;
        min-height: 250px;
        scrollbar-width: thin !important;
    }
    .output-code .hide {
        display: none;
    }
    .output-code .cm-scroller {
        flex-grow: 1;
    }
    .output-code .cm-gutters {
        position: relative !important;
    }
    .output-image {
        flex-grow: 1;
        height: 0vh;
        min-height: 250px;
        overflow-y: auto !important;
        scrollbar-width: thin !important;
    }
    .output-image .image-container, .output-image .grid-container {
       width: 100%;
       height: 100%;
    }
    .output-image .thumbnail-item img {
        object-fit: contain;
    }
    .output-image .grid-wrap.fixed-height {
        max-height: 100% !important;
    }
    .outputs .tabs {
        display: flex;
        flex-direction: column;
        flex-grow: 1;
    }
    .outputs .tabitem[style="display: block;"] {
        flex-grow: 1;
        display: flex !important;
    }
    .outputs .gap {
        flex-grow: 1;
    }
    .outputs .form {
        flex-grow: 1 !important;
    }
    .outputs .form > :last-child{
        flex-grow: 1;
    }
"""

# (Ab)use an invisible fake button with id preview-close to propagate the
# actual press of the button that closes a preview
# https://github.com/gradio-app/gradio/issues/6697
GALLERY_DESELECT_HACK = """
<script>
    const observerOptions = {
      childList: true,
      subtree: true
    };

    const observer = new MutationObserver((mutationsList, observer) => {
      for (let mutation of mutationsList) {
        if (mutation.type === 'childList') {
          for (let node of mutation.removedNodes) {
            if (node.nodeName == "BUTTON" && node.classList.contains('preview')) {
              document.getElementById('preview-close').click();
            }
          }
        }
      }
    });

    observer.observe(document.body, observerOptions);
</script>
"""
