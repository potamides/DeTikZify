from transformers import LlamaConfig

class DetikzifyConfig(LlamaConfig):
    model_type = "detikzify"

    # compatibility with new inference code
    @property
    def image_token_id(self):
        return self.patch_token_id

    @property
    def pooling_mode(self):
        return "cos"
