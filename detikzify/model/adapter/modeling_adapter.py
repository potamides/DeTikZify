# coding=utf-8
# Copyright 2024 the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adapted from modelling_mllama.py and modelling_siglip.py
# https://github.com/huggingface/transformers/commit/2e24ee4dfa39cc0bc264b89edbccc373c8337086
from functools import partial
from os.path import basename
from typing import Optional, Tuple

import torch
from torch import nn
from transformers import AutoModel, PreTrainedModel, PretrainedConfig
from transformers.activations import ACT2FN
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
from transformers.utils import (
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
)

if is_flash_attn_2_available():
    from transformers.modeling_flash_attention_utils import _flash_attention_forward

logger = logging.get_logger(__name__)



class CrossAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config: Optional[PretrainedConfig] = None,
    ):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = self.config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

        self.q_norm = nn.LayerNorm(self.head_dim, eps=config.layer_norm_eps)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cross_attention_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Input shape: Batch x Time x Channel"""

        batch_size, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(cross_attention_states)
        value_states = self.v_proj(cross_attention_states)

        query_states = query_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        k_v_seq_len = key_states.shape[-2]
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale

        if attn_weights.size() != (batch_size, self.num_heads, q_len, k_v_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size, self.num_heads, q_len, k_v_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (batch_size, 1, q_len, k_v_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(batch_size, 1, q_len, k_v_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (batch_size, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, q_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


class CrossSdpaAttention(CrossAttention):
    """
    Attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from `CrossAttention`
    as the weights of the module stays untouched. The only changes are on the forward pass to adapt to SDPA API.
    """

    # Adapted from CrossAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        cross_attention_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Input shape: Batch x Time x Channel"""
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "Using CrossSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                cross_attention_states=cross_attention_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
            )

        batch_size, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(cross_attention_states)
        value_states = self.v_proj(cross_attention_states)

        query_states = query_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, q_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, None


class CrossFlashAttention2(CrossAttention):
    """
    CrossAttention flash attention module. This module inherits from `CrossAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2.__init__
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    # Adapted from transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        cross_attention_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        output_attentions = False

        batch_size, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(cross_attention_states)
        value_states = self.v_proj(cross_attention_states)

        query_states = query_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32.

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        if attention_mask is not None and attention_mask.all():
            # FIXME: figure out why all 1 attention mask leads to different results
            attention_mask = None

        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=dropout_rate,
            is_causal=False,
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
        )

        attn_output = attn_output.reshape(batch_size, q_len, self.embed_dim).contiguous()
        attn_output = self.out_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights



CROSS_ATTENTION_CLASSES = {
    "eager": CrossAttention,
    "sdpa": CrossSdpaAttention,
    "flash_attention_2": CrossFlashAttention2,
}


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class CrossAttentionLayer(torch.nn.Module):
    """Cross-attention transformer block with tanh-gated attention and feedforward."""

    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__()
        self.embed_dim = config.hidden_size
        self.cross_attn = CROSS_ATTENTION_CLASSES[config._attn_implementation](config=config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = MLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

        self.cross_attn_attn_gate = torch.nn.Parameter(torch.zeros(1))
        self.cross_attn_mlp_gate = torch.nn.Parameter(torch.zeros(1))

    def forward(
        self,
        hidden_states: torch.Tensor,
        cross_attention_states: torch.Tensor,
        cross_attention_mask: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.cross_attn(
            hidden_states=hidden_states,
            attention_mask=cross_attention_mask,
            cross_attention_states=cross_attention_states,
            output_attentions=output_attentions,
        )
        hidden_states = residual + self.cross_attn_attn_gate.tanh() * hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.cross_attn_mlp_gate.tanh() * hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class CrossAttentionAdapter(PreTrainedModel):
    base_model_prefix = "model"
    no_split_modules = ["CrossAttentionLayer"]
    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    def __init__(self, config, input_hidden_size, cross_attn_every_n_layers=1, use_dummy=True):
        super().__init__(config)
        self.num_patches = (config.image_size // config.patch_size) ** 2
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self.layers = nn.ModuleList([ # type: ignore
            CrossAttentionLayer(config)
            if (layer_idx + 1) % cross_attn_every_n_layers == 0
            else None
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.connector = nn.Linear(
            input_hidden_size,
            config.hidden_size,
            bias=True
        )

        if use_dummy:
            self.dummy_input = nn.Parameter(
                torch.ones(
                    config.num_channels,
                    config.image_size,
                    config.image_size
                )
            )

        self.post_init()

    def connect(self, inputs):
        return self.connector(inputs)

    def prepare_4d_attention_mask(self, attention_mask, dtype):
        if attention_mask is not None and not self._use_flash_attention_2:
            # [batch_size, seq_len] -> [batch_size, 1, tgt_seq_len, src_seq_len]
            return _prepare_4d_attention_mask(attention_mask, dtype, self.num_patches)
        return attention_mask


# pyright: reportAttributeAccessIssue=false
class CrossAttentionAdapterMixin:
    def init_cross_attn_adapter(
        self,
        model_or_model_name_or_path,
        cross_attn_every_n_layers: Optional[int] = 1,
        use_dummy: Optional[bool] = True,
        **adapter_kwargs,
    ):
        self.embedding_model = self.load_embedding_model(
            model_or_model_name_or_path,
            **adapter_kwargs
        ).to(self.device)
        self.adapter = CrossAttentionAdapter._from_config(
            input_hidden_size=self.embedding_model.config.hidden_size,
            cross_attn_every_n_layers=cross_attn_every_n_layers,
            use_dummy=use_dummy,
            config=getattr(self.config, "vision_config", self.config),
            torch_dtype=self.dtype,
            **adapter_kwargs
        ).to(self.device, self.dtype)
        self.handles = self.add_hooks()

    def load_cross_attn_adapter(
        self,
        model_or_model_name_or_path,
        adapter_name_or_path: Optional[str] = None,
        cross_attn_every_n_layers: Optional[int] = 1,
        use_dummy: Optional[bool] = True,
        **adapter_kwargs,
    ):
        self.embedding_model = self.load_embedding_model(
            model_or_model_name_or_path,
            **adapter_kwargs
        )
        try:
            self.adapter = CrossAttentionAdapter.from_pretrained(
                pretrained_model_name_or_path=adapter_name_or_path,
                input_hidden_size=self.embedding_model.config.hidden_size,
                cross_attn_every_n_layers=cross_attn_every_n_layers,
                use_dummy=use_dummy,
                config=getattr(self.config, "vision_config", self.config),
                torch_dtype=self.dtype,
                **adapter_kwargs
            ).to(self.dtype)
        except OSError:
            name_or_path = adapter_name_or_path or basename(self.embedding_model.name_or_path)
            self.adapter = CrossAttentionAdapter.from_pretrained(
                pretrained_model_name_or_path=self.config.name_or_path,
                input_hidden_size=self.embedding_model.config.hidden_size,
                cross_attn_every_n_layers=cross_attn_every_n_layers,
                config=getattr(self.config, "vision_config", self.config),
                subfolder=f"adapters/{name_or_path}",
                torch_dtype=self.dtype,
                **adapter_kwargs
            ).to(self.dtype)
        if "device_map" not in adapter_kwargs:
            self.embedding_model = self.embedding_model.to(self.device)
            self.adapter = self.adapter.to(self.device)
        self.handles = self.add_hooks()

    def load_embedding_model(self, model_or_model_name_or_path, **model_kwargs):
        if isinstance(model_or_model_name_or_path, str):
            model = AutoModel.from_pretrained(
                pretrained_model_name_or_path=model_or_model_name_or_path,
                torch_dtype=self.dtype,
                **model_kwargs
            )
        else:
            model = model_or_model_name_or_path
        return model.to(self.dtype)

    def add_hooks(self):
        handles, adapter_inputs, cross_attention = list(), dict(), dict()

        for name, module in self.named_modules():
            if "vision_model" in name and type(module) == nn.ModuleList:
                vision_layers = module
                break
        else:
            raise ValueError("Couldn't locate vision encoder layers!")

        # HACK: convert args to kwargs
        def args_to_kwargs(module, args):
            return dict(zip(type(module).forward.__code__.co_varnames[1:], args))

        def forward_hook(layer, args, kwargs):
            if (adapter_input_ids:=kwargs.pop("adapter_input_ids", None)) is not None:
                if not hasattr(self, "adapter"):
                    raise ValueError("Got `adapter_input_ids` but no adapter is loaded!")
                adapter_inputs.update(
                    input_ids=adapter_input_ids,
                    attention_mask=kwargs.pop("adapter_attention_mask", None)
                )
                if "pixel_values" not in kwargs | args_to_kwargs(layer, args):
                    if hasattr(self.adapter, "dummy_input"):
                        dummy_input = self.adapter.dummy_input.clamp(-1, 1)
                    else:
                        config = getattr(self.config, "vision_config", self.config)
                        dummy_input = torch.ones(config.num_channels, config.image_size, config.image_size)
                    kwargs['pixel_values'] = dummy_input.repeat(len(adapter_input_ids), 1, 1, 1)

            return args, kwargs

        for layer, cross_layer in zip(vision_layers, self.adapter.layers):
            if cross_layer is not None:
                def layer_hook(cross_layer, layer, args, kwargs):
                    if adapter_inputs:
                        embeddings = self.embedding_model(**adapter_inputs).last_hidden_state
                        cross_attention.update(
                            cross_attention_states=self.adapter.connect(embeddings),
                            cross_attention_mask=self.adapter.prepare_4d_attention_mask(
                                adapter_inputs["attention_mask"],
                                embeddings.dtype
                        ))
                        adapter_inputs.clear()
                    if cross_attention:
                        kwargs |= args_to_kwargs(layer, args)
                        kwargs['hidden_states'] = cross_layer(**cross_attention, **kwargs)[0]
                        return [], kwargs

                handles.append(layer.register_forward_pre_hook(partial(layer_hook, cross_layer), with_kwargs=True))
        handles.append(self.register_forward_pre_hook(forward_hook, with_kwargs=True))
        handles.append(self.register_forward_hook(lambda *_: cross_attention.clear()))

        return handles

    def unload_cross_attn_adapter(self):
        for handle in self.handles:
            handle.remove()
        del self.adapter, self.embedding_model, self.handles

    def save_cross_attn_adapter(self, *args, **kwargs):
        return self.adapter.save_pretrained(*args, **kwargs)

    def has_adapter(self):
        return hasattr(self, "adapter")
