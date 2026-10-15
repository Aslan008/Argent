"""The content expert of the yes/no read-out (Leeth2Config.noul_content): off = no extra modules; on and shut = the original head (to float rounding);
on and open = the content expert's logit."""

import dataclasses

import pytest
import torch

from src.leeth2.encoding import Leeth2Encoder, collate
from src.leeth2.model import Leeth2Config, Leeth2Model
from src.leeth2.schema import Sample

from .conftest import VOCAB


def _batch(tok):
    e = Leeth2Encoder(tok, VOCAB, max_len=128)
    items = [Sample("t", "f", "en", "noul", "it is raining outside today", "is it wet outside", None, 1.0),
             Sample("t", "f", "en", "noul", "the cat sat on the mat", "is this a complaint", None, 0.0)]
    return collate([e.encode(s, rng=None) for s in items])


def _with_content(base: Leeth2Model, tiny_cfg) -> Leeth2Model:
    from transformers import ModernBertConfig, ModernBertModel

    bc = ModernBertConfig(**tiny_cfg, attn_implementation="eager")
    cfg = Leeth2Config(**{**dataclasses.asdict(base.cfg), "noul_content": True})
    m = Leeth2Model(cfg, ModernBertModel(bc))
    missing, unexpected = m.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected and all(k.startswith("content_") for k in missing)
    return m.eval()


def test_off_by_default_adds_nothing(tiny_model):
    assert not tiny_model.cfg.noul_content
    assert not [k for k in tiny_model.state_dict() if k.startswith("content_")]


def test_fresh_expert_is_the_original_head(tok, tiny_model, tiny_cfg):
    b = _batch(tok)
    tiny_model.eval()
    m = _with_content(tiny_model, tiny_cfg)
    with torch.no_grad():
        want = tiny_model(**b)["noul_logit"]
        got = m(**b)["noul_logit"]
    assert torch.allclose(want, got, atol=1e-6)  # (1 - g) z + g z: equal up to float rounding


def test_a_shut_gate_is_the_original_and_an_open_gate_is_the_expert(tok, tiny_model, tiny_cfg):
    b = _batch(tok)
    tiny_model.eval()
    m = _with_content(tiny_model, tiny_cfg)
    with torch.no_grad():
        m.content_delta[2].weight.normal_(0, 0.5)
        m.content_delta[2].bias.fill_(0.7)
        m.content_scale.fill_(1.5)
        m.content_bias.fill_(-0.25)
        base = tiny_model(**b)["noul_logit"]
        m.content_gate[2].bias.fill_(-30.0)
        shut = m(**b)["noul_logit"]
        m.content_gate[2].bias.fill_(30.0)
        opened = m(**b)["noul_logit"]
        # what the expert must give: scale * original + bias + delta(q), with q the read-out the heads see
        hidden = m.encode_states(b["input_ids"], b["attention_mask"])
        B = hidden.size(0)
        h_rd = hidden[torch.arange(B), b["rd_pos"]]
        mask = b["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        q = torch.cat([h_rd, (hidden * mask).sum(1) / mask.sum(1).clamp(min=1.0)], dim=-1)
        expert = 1.5 * base - 0.25 + m.content_delta(q).squeeze(-1)
    assert torch.allclose(shut, base, atol=1e-5)
    assert torch.allclose(opened, expert, atol=1e-4)
    assert not torch.allclose(opened, base, atol=1e-3)


def test_head_parameters_include_the_expert(tiny_model, tiny_cfg):
    m = _with_content(tiny_model, tiny_cfg)
    ids = {id(p) for p in m.head_parameters()}
    for name, p in m.named_parameters():
        if name.startswith("content_"):
            assert id(p) in ids, name


def test_save_and_load_keeps_the_expert(tok, tiny_model, tiny_cfg, tmp_path):
    b = _batch(tok)
    m = _with_content(tiny_model, tiny_cfg)
    with torch.no_grad():
        m.content_delta[2].weight.normal_(0, 0.5)
        m.content_gate[2].bias.fill_(2.0)
        want = m(**b)["noul_logit"]
    m.save_pretrained(tmp_path)
    loaded = Leeth2Model.from_pretrained(tmp_path, device="cpu", dtype=torch.float32).eval()
    assert loaded.cfg.noul_content
    with torch.no_grad():
        got = loaded(**b)["noul_logit"]
    assert torch.allclose(want, got, atol=1e-5)
