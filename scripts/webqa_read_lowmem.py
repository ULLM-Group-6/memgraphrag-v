#!/usr/bin/env python3
"""`webqa_run.py read` with block-wise Qwen2.5-VL vision attention.

transformers 4.49 SDPA vision attention builds one dense seq x seq mask over all
image patches in a prompt, which OOMs a 40 GB A100 for multi-image questions.
Attention is block-diagonal (per image / per window), so running the original
forward on groups of whole segments gives the same result with bounded memory.
webqa_run.py is executed unmodified, so its runner_sha256 contract still holds.
"""
import runpy
import sys
from pathlib import Path

import torch
from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl as qwen

CHUNK = 4096
original = qwen.Qwen2_5_VLVisionSdpaAttention.forward


def groups(bounds):
    out, start, last = [], bounds[0], bounds[0]
    for b in bounds[1:]:
        if b - start > CHUNK and last > start:
            out.append((start, last))
            start = last
        last = b
    out.append((start, last))
    return out


def forward(self, hidden_states, cu_seqlens, rotary_pos_emb=None, position_embeddings=None):
    if hidden_states.shape[0] <= CHUNK:
        return original(self, hidden_states, cu_seqlens, rotary_pos_emb, position_embeddings)
    bounds = cu_seqlens.tolist()
    parts = []
    for s, e in groups(bounds):
        local = torch.tensor([b - s for b in bounds if s <= b <= e],
                             dtype=cu_seqlens.dtype, device=cu_seqlens.device)
        rotary = None if rotary_pos_emb is None else rotary_pos_emb[s:e]
        position = None if position_embeddings is None else tuple(t[s:e] for t in position_embeddings)
        parts.append(original(self, hidden_states[s:e], local, rotary, position))
    return torch.cat(parts)


qwen.Qwen2_5_VLVisionSdpaAttention.forward = forward

if __name__ == "__main__":
    runner = str(Path(__file__).resolve().parent / "webqa_run.py")
    sys.argv = [runner, *sys.argv[1:]]
    runpy.run_path(runner, run_name="__main__")
