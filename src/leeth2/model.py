"""Leeth 2 model: one pretrained bidirectional encoder + four light heads.

    tokens ──► embeddings (frozen table + trainable marker table)
           ──► ModernBERT / mmBERT layers
           ──► heads:
                 choice   per-option vector (mean of the option's tokens) x question read-out -> logit
                 noul     read-out + mean-pool -> logit
                 score    read-out + mean-pool -> ordinal logits (masked to the asked number of levels)
                 extract  per-token start/end logits over the state, "no answer" = the [RD] token

Only the transformer layers (~110M) and ~7M head parameters are trained; the 197M-parameter
embedding matrix stays frozen on purpose: updating only the tokens seen in fine-tuning
would drift them away from the ones that were not (hurting cross-lingual transfer) and costs
~2.4 GB of optimiser state for nothing.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoding import MARKER_NAMES, N_MARKERS
from .schema import MAX_LEVELS

NEG = -1e9


@dataclass
class Leeth2Config:
    base_model: str = "jhu-clsp/mmBERT-base"
    backbone: Optional[Dict[str, Any]] = None  # ModernBertConfig.to_dict(); lets us rebuild without the base repo
    vocab_size: int = 256000
    hidden_size: int = 768
    n_markers: int = N_MARKERS
    max_levels: int = MAX_LEVELS
    head_dropout: float = 0.1
    freeze_embeddings: bool = True
    # post-hoc calibration (fitted by calibrate.py on held-out data, never trained through)
    temp_choice: float = 1.0
    temp_noul: float = 1.0
    temp_score: float = 1.0
    # a long option list (> big_k options) is calibrated with its own temperature when one was fitted (0 = the same as temp_choice):
    # the logits of a 255-way choice do not have the spread of a 5-way one, and one scalar cannot be right for both
    temp_choice_big: float = 0.0
    big_k: int = 64
    # a choice with more options than this is answered in two rounds (evaluate.predict): parts of at most split_k options, then their winners.
    # 0 = off.  The network is reliable up to ~128 near-duplicate options and loses accuracy beyond ~160
    split_k: int = 0
    # the yes/no read-out is a mixture of two experts: the original noul head and a content head, blended by a gate that learnt to tell
    # "a question about the content of a text" ("is this post about X", "is this a complaint") from everything else.  Where the gate is
    # shut the answer is the original head's, to float rounding (commands, routing, safety); where it is open the content expert answers: the
    # original logit, rescaled and corrected by a small learnt term, so what the original head knows about unseen topics is kept and
    # only its calibration for such questions changes.  False = no extra modules (every earlier checkpoint)
    noul_content: bool = False
    version: str = "leeth2"
    # set by leeth2.grow: which backbone layers are new (inserted, started as identities) and how the network was widened
    grown: Optional[Dict[str, Any]] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=1)


class ChoiceHead(nn.Module):
    """logit_i = MLP(o_i, q) + <o_i, q>/sqrt(d); an option sees the question read-out, never other options directly."""

    def __init__(self, h: int, p: float):
        super().__init__()
        self.q = nn.Linear(2 * h, h)
        self.o = nn.Linear(h, h)
        self.mix = nn.Sequential(nn.GELU(), nn.Dropout(p), nn.Linear(h, h), nn.GELU(), nn.Linear(h, 1))
        self.scale = math.sqrt(h)

    def forward(self, opt: torch.Tensor, q: torch.Tensor) -> torch.Tensor:  # [B,K,H], [B,2H] -> [B,K]
        qv = self.q(q).unsqueeze(1)
        ov = self.o(opt)
        return self.mix(ov + qv).squeeze(-1) + (ov * qv).sum(-1) / self.scale


def _mlp(i: int, h: int, o: int, p: float) -> nn.Sequential:
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Dropout(p), nn.Linear(h, o))


class Leeth2Model(nn.Module):
    def __init__(self, cfg: Leeth2Config, backbone: Optional[nn.Module] = None):
        super().__init__()
        self.cfg = cfg
        if backbone is None:
            from transformers import ModernBertConfig, ModernBertModel

            if not cfg.backbone:
                raise ValueError("config.backbone is required to build the model without pretrained weights")
            backbone = ModernBertModel(ModernBertConfig(**cfg.backbone))
        self.backbone = backbone
        h = cfg.hidden_size
        p = cfg.head_dropout
        self.marker_embed = nn.Embedding(cfg.n_markers, h)
        nn.init.normal_(self.marker_embed.weight, std=0.02)
        self.choice_head = ChoiceHead(h, p)
        self.noul_head = _mlp(2 * h, h, 1, p)
        self.content_delta: Optional[nn.Module] = None
        self.content_gate: Optional[nn.Module] = None
        if cfg.noul_content:
            self.content_delta = nn.Sequential(nn.Linear(2 * h, 256), nn.GELU(), nn.Linear(256, 1))
            nn.init.zeros_(self.content_delta[2].weight)
            nn.init.zeros_(self.content_delta[2].bias)
            self.content_scale = nn.Parameter(torch.ones(()))  # content logit = scale * original logit + bias + delta(q)
            self.content_bias = nn.Parameter(torch.zeros(()))
            self.content_gate = nn.Sequential(nn.Linear(2 * h, 64), nn.GELU(), nn.Linear(64, 1))
            nn.init.constant_(self.content_gate[2].bias, -8.0)  # shut until trained: the mixture starts as the original head
        self.score_head = _mlp(2 * h, h, cfg.max_levels, p)
        self.span_head = nn.Linear(h, 2)
        if cfg.freeze_embeddings:
            self.freeze_embeddings()

    # -- construction ---------------------------------------------------------------
    @classmethod
    def from_base(cls, base: str, tokenizer: Any = None, attn_implementation: Optional[str] = None, **cfg_over: Any) -> "Leeth2Model":
        """Start from a pretrained ModernBERT-family encoder (path or hub id)."""
        from transformers import ModernBertModel

        kw: Dict[str, Any] = {}
        if attn_implementation:
            kw["attn_implementation"] = attn_implementation
        backbone = ModernBertModel.from_pretrained(base, **kw)
        bc = backbone.config
        cfg = Leeth2Config(base_model=str(base), backbone=bc.to_dict(), vocab_size=bc.vocab_size, hidden_size=bc.hidden_size, **cfg_over)
        model = cls(cfg, backbone)
        if tokenizer is not None:
            model.init_markers(tokenizer)
        return model

    @classmethod
    def from_arch(cls, path: str | Path, attn_implementation: Optional[str] = None) -> "Leeth2Model":
        """The architecture of a saved Leeth 2 (possibly grown: ``leeth2.grow``) with fresh weights; ``train.load_init`` fills them."""
        from transformers import ModernBertConfig, ModernBertModel

        p = Path(path)
        for cand in (p, p / "final"):
            if (cand / "leeth2_config.json").exists():
                p = cand
                break
        cfg = Leeth2Config(**json.loads((p / "leeth2_config.json").read_text(encoding="utf-8")))
        bc = ModernBertConfig(**cfg.backbone)
        if attn_implementation:
            bc._attn_implementation = attn_implementation
        return cls(cfg, ModernBertModel(bc))

    def init_markers(self, tokenizer: Any) -> None:
        """Start every marker at the embedding of a meaningful word instead of noise."""
        words = {"OPT": "option", "RD": "answer", "T_CHOICE": "choose", "T_NOUL": "true", "T_SCORE": "score", "T_EXTRACT": "extract"}
        emb = self.backbone.embeddings.tok_embeddings.weight.detach()
        with torch.no_grad():
            for i, name in enumerate(MARKER_NAMES):
                if name in words:
                    w = words[name]
                elif name.startswith("L"):
                    w = name[1:]
                else:
                    w = "x"
                ids = tokenizer(w, add_special_tokens=False)["input_ids"]
                if ids:
                    self.marker_embed.weight[i] = emb[torch.tensor(ids)].mean(0).to(self.marker_embed.weight.dtype)

    def freeze_embeddings(self) -> None:
        self.backbone.embeddings.tok_embeddings.weight.requires_grad_(False)

    # -- embeddings with virtual markers -------------------------------------------
    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        V = self.cfg.vocab_size
        tok = self.backbone.embeddings.tok_embeddings
        is_marker = input_ids >= V
        e = tok(input_ids.masked_fill(is_marker, 0))
        m = self.marker_embed((input_ids - V).clamp(min=0, max=self.cfg.n_markers - 1)).to(e.dtype)
        return torch.where(is_marker.unsqueeze(-1), m, e)

    # -- forward --------------------------------------------------------------------
    def encode_states(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.backbone(inputs_embeds=self.embed(input_ids), attention_mask=attention_mask)
        return out.last_hidden_state

    @staticmethod
    def pool_options(hidden: torch.Tensor, opt_index: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
        """Mean of each option's positions -> [B, K, H]. ``slots`` = arange(1, K+1); passing it as a tensor (instead of an
        int) keeps K a dynamic axis when the graph is exported to ONNX."""
        ks = slots.to(opt_index.dtype).view(1, -1, 1)
        onehot = (opt_index.unsqueeze(1) == ks).to(hidden.dtype)  # [B,K,L]
        summed = torch.bmm(onehot, hidden)
        count = onehot.sum(-1, keepdim=True).clamp(min=1.0)
        return summed / count

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        opt_index: torch.Tensor,
        rd_pos: torch.Tensor,
        state_mask: torch.Tensor,
        n_opts: torch.Tensor,
        k_max: int = 1,
        slots: Optional[torch.Tensor] = None,
        **_: Any,
    ) -> Dict[str, torch.Tensor]:
        if slots is None:
            slots = torch.arange(1, k_max + 1, device=input_ids.device)
        hidden = self.encode_states(input_ids, attention_mask)
        B = hidden.size(0)
        h_rd = hidden[torch.arange(B, device=hidden.device), rd_pos]
        m = attention_mask.unsqueeze(-1).to(hidden.dtype)
        mean = (hidden * m).sum(1) / m.sum(1).clamp(min=1.0)
        q = torch.cat([h_rd, mean], dim=-1)

        opt = self.pool_options(hidden, opt_index, slots)
        choice = self.choice_head(opt, q).float()
        valid = slots.unsqueeze(0) <= n_opts.unsqueeze(1)
        choice = choice.masked_fill(~valid, NEG)

        noul = self.noul_head(q).squeeze(-1).float()
        if self.content_gate is not None:
            gate = torch.sigmoid(self.content_gate(q).squeeze(-1).float())
            content = self.content_scale.float() * noul + self.content_bias.float() + self.content_delta(q).squeeze(-1).float()
            noul = (1.0 - gate) * noul + gate * content
        score = self.score_head(q).float()
        span = self.span_head(hidden).float()  # [B,L,2]
        start = span[..., 0].masked_fill(~state_mask, NEG)
        end = span[..., 1].masked_fill(~state_mask, NEG)
        return {"choice_logits": choice, "noul_logit": noul, "score_logits": score, "start_logits": start, "end_logits": end}

    # -- parameters -----------------------------------------------------------------
    def head_parameters(self) -> List[nn.Parameter]:
        mods = [self.marker_embed, self.choice_head, self.noul_head, self.score_head, self.span_head]
        mods += [m for m in (self.content_delta, self.content_gate) if m is not None]
        extra = [self.content_scale, self.content_bias] if self.content_gate is not None else []
        return [p for m in mods for p in m.parameters()] + extra

    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}

    # -- persistence ----------------------------------------------------------------
    def save_pretrained(self, out_dir: str | Path, half: bool = False) -> Path:
        from safetensors.torch import save_file

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "leeth2_config.json").write_text(self.cfg.to_json(), encoding="utf-8")
        sd = {}
        for k, v in self.state_dict().items():
            v = v.detach().cpu().contiguous()
            sd[k] = v.half() if (half and v.is_floating_point()) else v
        save_file(sd, str(out / "model.safetensors"))
        return out

    @classmethod
    def from_pretrained(cls, path: str | Path, device: str = "cpu", dtype: Optional[torch.dtype] = None, strict: bool = True) -> "Leeth2Model":
        from safetensors.torch import load_file

        p = Path(path)
        cfg = Leeth2Config(**json.loads((p / "leeth2_config.json").read_text(encoding="utf-8")))
        model = cls(cfg)
        sd = load_file(str(p / "model.safetensors"))
        want = model.state_dict()
        sd = {k: v.to(want[k].dtype) if k in want else v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if strict and (missing or unexpected):
            raise RuntimeError(f"checkpoint mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
        model.to(device)
        if dtype is not None:
            model.to(dtype)
        return model


COMPRESS_MODES = ("none", "emb16")


def compress_for_inference(model: "Leeth2Model", mode: str = "emb16") -> "Leeth2Model":
    """Shrink a CPU inference copy of the model without changing one output bit.

    ``emb16``: the token table is 63 % of the parameters (196M of 312M), it is frozen, and the weights on disk are already
    half precision. Keeping the table in fp16 and widening each looked-up row to fp32 gives exactly the numbers the fp32
    copy gives (measured: max probability difference 0.0 over 450 queries) for ~400 MB less memory.

    There is deliberately no int8 mode: dynamic int8 quantisation of the transformer layers was measured and dropped the
    routing exams from 96.3 % to 77.9 % and flipped 47 of 125 safety-gate decisions.
    """
    if mode not in COMPRESS_MODES:
        raise ValueError(f"unknown compression {mode!r}; choose from {COMPRESS_MODES}")
    if mode == "emb16":
        tok = model.backbone.embeddings.tok_embeddings
        if tok.weight.dtype != torch.float16:
            tok.weight.data = tok.weight.data.half()
            tok.register_forward_hook(lambda module, args, out: out.float())
    return model
