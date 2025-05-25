## fix listen in 0.0.0.0

```
vim /usr/local/lib/python3.11/dist-packages/detikzify/webui/__main__.py
build_ui(**args).queue().launch(share=share, server_name="0.0.0.0", server_port=7860)
```