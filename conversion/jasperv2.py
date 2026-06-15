from __future__ import annotations

from typing import Any, Callable, Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf


@ModelBase.register("JasperV2Encoder")
class JasperV2EncoderModel(TextModel):
    model_arch = gguf.MODEL_ARCH.JASPER_V2_ENCODER

    def set_gguf_parameters(self):
        self.gguf_writer.add_block_count(self.hparams.get("num_hidden_layers", 28))
        self.gguf_writer.add_embedding_length(self.hparams.get("hidden_size", 1024))
        self.gguf_writer.add_embedding_length_out(2048)
        self.gguf_writer.add_feed_forward_length(self.hparams.get("intermediate_size", 2816))
        self.gguf_writer.add_head_count(self.hparams.get("num_attention_heads", 16))
        self.gguf_writer.add_head_count_kv(self.hparams.get("num_key_value_heads", 16))
        self.gguf_writer.add_layer_norm_rms_eps(self.hparams.get("rms_norm_eps", 1e-6))
        self.gguf_writer.add_rope_freq_base(float(self.hparams.get("rope_theta", 1000000.0)))
        rope_scaling = self.hparams.get("rope_scaling", None)
        if isinstance(rope_scaling, dict):
            self.gguf_writer.add_rope_scaling_type(rope_scaling.get("type"))
            self.gguf_writer.add_rope_scaling_factor(float(rope_scaling.get("factor", 1.0)))
        self.gguf_writer.add_context_length(self.hparams.get("max_position_embeddings", 32768))
        self.gguf_writer.add_causal_attention(False)
        self.gguf_writer.add_dense_features_dims("dense_2", self.hparams.get("hidden_size", 1024), 2048)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # jasper_mlp pre-backbone FFN
        if name == "jasper_mlp.gate_proj.weight":
            yield "jasper_mlp_gate.weight", data_torch
            return
        if name == "jasper_mlp.up_proj.weight":
            yield "jasper_mlp_up.weight", data_torch
            return
        if name == "jasper_mlp.down_proj.weight":
            yield "jasper_mlp_down.weight", data_torch
            return

        # output projection (linear_1 -> dense.2)
        if name == "linear_1.weight":
            yield "dense_2.weight", data_torch
            return
        if name == "linear_1.bias":
            yield "dense_2.bias", data_torch
            return

        # backbone weights: use standard name mapping
        new_name = self.map_tensor_name(name)
        if new_name is not None:
            yield new_name, data_torch
