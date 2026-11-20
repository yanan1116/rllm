"""Run Qwen3.5's Gated DeltaNet layers per sequence when verl packs several rows into one.

With use_remove_padding, verl concatenates a micro-batch into one (1, total_nnz) row. The
full-attention layers stay per sequence (flash varlen from position_ids), but HF's
Qwen3_5GatedDeltaNet has no notion of sequence boundaries: its causal conv1d and its recurrent
gated-delta state run straight across them, so every row that is not first in its micro-batch is
computed from the previous row's state. Measured on Qwen3.5-4B: the assistant tokens of the second
row move by 0.08 nats on average (the bf16 noise floor is 0.007), and the whole row by 1.5 nats.

This module splits each linear-attention call at the points where the text position ids restart
and runs the unchanged layer once per sequence, which reproduces an unpacked forward exactly. Any
call without a restart (one row, padded batches, generation with a cache) takes the original path.

Loaded inside every verl worker through actor_rollout_ref.model.external_lib=appworld_gdn_varlen
(train_appworld.sh --gdn-varlen); importing it applies the patch.
"""

from __future__ import annotations

import torch
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen3_5

_layer_forward = qwen3_5.Qwen3_5DecoderLayer.forward
_gdn_forward = qwen3_5.Qwen3_5GatedDeltaNet.forward


def sequence_bounds(position_ids: torch.Tensor | None, length: int) -> list[int] | None:
    """[0, s1, ..., length] where the text positions restart, or None for a single sequence."""
    if position_ids is None:
        return None
    positions = position_ids
    while positions.dim() > 2:  # (4, 1, L) multimodal rope ids: row 0 is the text position
        positions = positions[0]
    if positions.shape[0] != 1 or positions.shape[-1] != length:
        return None
    restarts = (positions[0, 1:] != positions[0, :-1] + 1).nonzero().flatten() + 1
    if restarts.numel() == 0:
        return None
    return [0, *restarts.tolist(), length]


def layer_forward(self, hidden_states, position_embeddings, attention_mask=None, position_ids=None,
                  past_key_values=None, **kwargs):
    if self.layer_type == "linear_attention":
        packed = past_key_values is None and attention_mask is None
        self.linear_attn._sequence_bounds = (
            sequence_bounds(position_ids, hidden_states.shape[1]) if packed else None)
    return _layer_forward(self, hidden_states, position_embeddings, attention_mask=attention_mask,
                          position_ids=position_ids, past_key_values=past_key_values, **kwargs)


def gdn_forward(self, hidden_states, cache_params=None, attention_mask=None):
    bounds = getattr(self, "_sequence_bounds", None)
    if bounds is None or cache_params is not None:
        return _gdn_forward(self, hidden_states, cache_params=cache_params, attention_mask=attention_mask)
    if hidden_states.shape[1] != bounds[-1]:
        raise RuntimeError(f"appworld_gdn_varlen: hidden length {hidden_states.shape[1]} != packed length "
                           f"{bounds[-1]}; the sequence bounds belong to another call")
    return torch.cat([_gdn_forward(self, hidden_states[:, start:end])
                      for start, end in zip(bounds[:-1], bounds[1:])], dim=1)


qwen3_5.Qwen3_5DecoderLayer.forward = layer_forward
qwen3_5.Qwen3_5GatedDeltaNet.forward = gdn_forward
print("[appworld_gdn_varlen] Qwen3.5 Gated DeltaNet runs per packed sequence", flush=True)
