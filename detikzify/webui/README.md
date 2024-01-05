# Web UI
The web UI of DeTi*k*Zify requires [TeX Live](https://www.tug.org/texlive),
[ghostscript](https://www.ghostscript.com), and
[poppler](https://poppler.freedesktop.org). You can launch it by running
`python -m detikzify.webui`. It comes with a command line interface. With the
`--share` flag, for example, you can create a shareable link. Checkout `--help`
for a full list of supported flags. As scientific figures usually use black
fonts on a white background, it is best to use the web UI in light mode. This
can be enforced by using the `--force_light` flag.
