"""Shared helpers: a tiny deterministic tokenizer and a tiny random ModernBERT — no downloads, runs on CPU in seconds."""

from __future__ import annotations

import re
import zlib

import pytest

VOCAB = 500


class FakeTokenizer:
    """Word-level tokenizer with offsets, shaped like a HF fast tokenizer (the parts Leeth2Encoder uses)."""

    cls_token_id = 2
    bos_token_id = 2
    sep_token_id = 1
    eos_token_id = 1
    pad_token_id = 0

    def _split(self, text: str):
        return [(m.group(0), m.start(), m.end()) for m in re.finditer(r"\s*\S+", text)]

    def _id(self, tok: str) -> int:
        return 10 + zlib.crc32(tok.strip().encode("utf-8")) % (VOCAB - 10)

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False, **kw):
        if isinstance(text, (list, tuple)):
            return {"input_ids": [[self._id(t) for t, _, _ in self._split(x)] for x in text]}
        parts = self._split(text)
        out = {"input_ids": [self._id(t) for t, _, _ in parts]}
        if return_offsets_mapping:
            out["offset_mapping"] = [(s, e) for _, s, e in parts]
        return out


@pytest.fixture
def tok():
    return FakeTokenizer()


@pytest.fixture
def tiny_cfg():
    return dict(
        vocab_size=VOCAB,
        hidden_size=32,
        num_hidden_layers=3,
        num_attention_heads=4,
        intermediate_size=48,
        max_position_embeddings=512,
        global_attn_every_n_layers=3,
        local_attention=16,
        pad_token_id=0,
        bos_token_id=2,
        eos_token_id=1,
        cls_token_id=2,
        sep_token_id=1,
    )


@pytest.fixture
def tiny_model(tiny_cfg):
    import torch
    from transformers import ModernBertConfig, ModernBertModel

    from src.leeth2.model import Leeth2Config, Leeth2Model

    torch.manual_seed(0)
    bc = ModernBertConfig(**tiny_cfg, attn_implementation="eager")
    cfg = Leeth2Config(base_model="tiny", backbone=bc.to_dict(), vocab_size=VOCAB, hidden_size=32, head_dropout=0.0)
    return Leeth2Model(cfg, ModernBertModel(bc))


# CI installs a light dependency set without torch/transformers: the tests that need them are skipped there, while the
# pure-Python ones (schema, dataset adapters, synthetic domains, builder, report) still run.
import importlib.util as _ilu

_TORCH_TESTS = [
    "test_model_losses.py", "test_evaluate_calibrate.py", "test_train_smoke.py", "test_runtime.py", "test_export.py", "test_soft_targets.py",
    "test_backend_switch.py", "test_datafile.py", "test_encoding.py", "test_noul_content.py",
]
collect_ignore = _TORCH_TESTS if any(_ilu.find_spec(m) is None for m in ("torch", "transformers", "scipy", "safetensors")) else []
