"""
src/leeth/model.py — Leeth Non-Autoregressive Decision Transformer Architecture.

Implements the multi-head System One decision engine:
- Backbone: 8-layer bidirectional Transformer Encoder with Rotary Position Embeddings (RoPE),
  RMSNorm, and PyTorch Scaled Dot-Product Attention (SDPA).
- Context Window: 1,024 tokens (LeethConfig.max_position_embeddings; extending it
  requires retraining — RoPE positions beyond the trained range degrade).
- Heads:
  1. ChoiceHead: Softmax classification over candidate options.
  2. NoulHead: Calibrated boolean judgment (sigmoid probability).
  3. ScoreHead: Ordinal rubric regression (0..100).
  4. ExtractHead: Pointer span logits (start_idx, end_idx) for parameter extraction.
"""

import math
from dataclasses import dataclass, fields
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint


@dataclass
class LeethConfig:
    vocab_size: int = 4096
    hidden_size: int = 512
    num_layers: int = 8
    num_heads: int = 8
    intermediate_size: int = 1536
    max_position_embeddings: int = 8192
    dropout: float = 0.1
    num_choices: int = 32          # Maximum candidate choice classes
    max_score_levels: int = 10     # Ordinal score head: logits over rubric levels
    temperature: float = 1.0
    opt_proj_dim: int = 256        # DynamicOptionHead projection dimension
    rope_base: float = 100000.0    # Scaled base frequency for long contexts up to 8192 tokens


#: Checkpoints saved before `rope_base` existed were trained with the rotary base the code hard-wired then.
LEGACY_ROPE_BASE = 10000.0


def config_from_checkpoint(raw) -> "LeethConfig":
    """The LeethConfig a checkpoint was TRAINED with.

    A pickled config only carries the attributes that existed when it was saved; reading a missing one
    falls through to the class default, which is what the CURRENT code wants for a NEW model. For an
    old checkpoint that silently changes the model: `rope_base` went from a hard-wired 10000 to 100000,
    the rotary angles differ, and torch no longer reproduces the exported ONNX (hidden states off by ~3).
    So a field the checkpoint does not carry takes the value in force when it was trained.
    """
    data = dict(vars(raw)) if isinstance(raw, LeethConfig) else dict(raw)
    data.setdefault("rope_base", LEGACY_ROPE_BASE)
    known = {f.name for f in fields(LeethConfig)}
    return LeethConfig(**{k: v for k, v in data.items() if k in known})


class DynamicOptionHead(nn.Module):
    """Single-Pass Dynamic Option Scoring (bi-encoder dot-product).

    Scores an ARBITRARY number of candidate options in one forward pass:
        logit_i = (state_proj(h_state) . opt_proj(h_opt_i)) / sqrt(proj_dim)
    - h_state: pooled state+question representation [B, H]
    - h_opt_i: hidden state at the i-th [OPT] marker position [B, K, H]
    Latency is independent of the option vocabulary — any string list works
    zero-shot (games, ad-hoc tool sets, new languages).
    """

    def __init__(self, hidden_size: int, proj_dim: int = 256):
        super().__init__()
        self.state_proj = nn.Linear(hidden_size, proj_dim, bias=False)
        self.opt_proj = nn.Linear(hidden_size, proj_dim, bias=False)
        self.scale = math.sqrt(proj_dim)

    def forward(self, state_repr: torch.Tensor, opt_reprs: torch.Tensor) -> torch.Tensor:
        # state_repr: [B, H] -> [B, 1, P]; opt_reprs: [B, K, H] -> [B, K, P]
        s = self.state_proj(state_repr).unsqueeze(1)
        o = self.opt_proj(opt_reprs)
        return torch.sum(s * o, dim=-1) / self.scale


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x_norm = x * torch.rsqrt(variance + self.eps)
        return self.weight * x_norm


class RotaryEmbedding(nn.Module):
    """Rotary Positional Embeddings (RoPE) with dynamic context extension."""

    def __init__(self, dim: int, max_position_embeddings: int = 8192, base: float = 100000.0):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len_cached = max(1024, max_position_embeddings)

        t = torch.arange(self.max_seq_len_cached, dtype=torch.float)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def _update_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        self.max_seq_len_cached = max(seq_len, self.max_seq_len_cached * 2)
        inv_freq = self.inv_freq.to(device)
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.float)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos().to(dtype)
        self.sin_cached = emb.sin().to(dtype)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len_cached or self.cos_cached.device != x.device:
            self._update_cache(seq_len, device=x.device, dtype=x.dtype)
        return self.cos_cached[:seq_len, :], self.sin_cached[:seq_len, :]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Reshape cos and sin for broadcasting: [1, 1, seq_len, head_dim]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LeethAttention(nn.Module):
    """Multi-Head Self-Attention with RoPE and SDPA."""

    def __init__(self, config: LeethConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        assert self.head_dim * self.num_heads == self.hidden_size, "hidden_size must be divisible by num_heads"

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.dropout = config.dropout

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # PyTorch 2.0+ Scaled Dot-Product Attention (Hardware-accelerated via FlashAttention/Cutlass)
        attn_mask = attention_mask
        if attn_mask is not None and attn_mask.dtype != torch.bool:
            # SDPA requires the additive mask dtype to match query dtype (bf16/fp16 weights)
            attn_mask = attn_mask.to(query_states.dtype)
        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,  # Bidirectional encoder
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        return self.out_proj(attn_output)


class LeethMLP(nn.Module):
    """Feedforward layer with GeLU activation."""

    def __init__(self, config: LeethConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))


class LeethLayer(nn.Module):
    """Transformer Encoder block with pre-normalization."""

    def __init__(self, config: LeethConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.hidden_size)
        self.attn = LeethAttention(config)
        self.norm2 = RMSNorm(config.hidden_size)
        self.mlp = LeethMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        normed = self.norm1(hidden_states)
        attn_out = self.attn(normed, cos, sin, attention_mask=attention_mask)
        hidden_states = hidden_states + attn_out

        normed2 = self.norm2(hidden_states)
        mlp_out = self.mlp(normed2)
        hidden_states = hidden_states + mlp_out

        return hidden_states


class LeethModel(nn.Module):
    """Complete Leeth Non-Autoregressive Decision Model with Multi-Heads."""

    def __init__(self, config: LeethConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.rotary_emb = RotaryEmbedding(
            config.hidden_size // config.num_heads,
            config.max_position_embeddings,
            base=getattr(config, "rope_base", 50000.0),
        )
        self.layers = nn.ModuleList([LeethLayer(config) for _ in range(config.num_layers)])
        self.final_norm = RMSNorm(config.hidden_size)

        # Decision Heads
        # 1. Choice Head (Categorical selection across candidate classes)
        self.choice_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.num_choices),
        )

        # 2. Noul Head (Calibrated boolean probability)
        self.noul_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, 1),
        )

        # 3. Score Head (Ordinal rubric: logits over rubric levels, softmax-restricted per question)
        self.score_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, getattr(config, "max_score_levels", 10)),
        )

        # 4. Extract Head (Pointer Span Logits: start and end position classifiers)
        self.extract_start_head = nn.Linear(config.hidden_size, 1)
        self.extract_end_head = nn.Linear(config.hidden_size, 1)

        # 5. MLM Head (Masked Language Modeling for self-supervised pretraining)
        self.lm_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            RMSNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.vocab_size, bias=False),
        )

        # 6. Dynamic Option Head (zero-shot scoring over arbitrary [OPT] lists)
        self.dynamic_opt_head = DynamicOptionHead(config.hidden_size, config.opt_proj_dim)

        # Calibrated temperature scaling parameter (learned or fixed)
        self.temperature = nn.Parameter(torch.tensor([config.temperature]), requires_grad=True)

        # Gradient checkpointing: trade compute for VRAM during large-model training
        self.gradient_checkpointing = False

        self._init_weights()

    def forward_mlm(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for Self-Supervised Masked Language Modeling pretraining."""
        bsz, seq_len = input_ids.size()
        hidden_states = self.embed_tokens(input_ids)
        cos, sin = self.rotary_emb(hidden_states, seq_len)

        attn_mask = None
        if attention_mask is not None:
            attn_mask = (1.0 - attention_mask[:, None, None, :].float()) * -10000.0

        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    layer, hidden_states, cos, sin, attn_mask, use_reentrant=False
                )
            else:
                hidden_states = layer(hidden_states, cos, sin, attention_mask=attn_mask)

        hidden_states = self.final_norm(hidden_states)
        return self.lm_head(hidden_states)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        opt_positions: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass computing representations and all decision head outputs in a single parallel pass."""
        bsz, seq_len = input_ids.size()

        # Embedding lookup
        hidden_states = self.embed_tokens(input_ids)

        # RoPE frequencies
        cos, sin = self.rotary_emb(hidden_states, seq_len)

        # Expand attention mask if provided
        attn_mask = None
        if attention_mask is not None:
            # [bsz, 1, 1, seq_len] with 0 for valid, -inf for masked
            attn_mask = (1.0 - attention_mask[:, None, None, :].float()) * -10000.0

        # Transformer encoder layers
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    layer, hidden_states, cos, sin, attn_mask, use_reentrant=False
                )
            else:
                hidden_states = layer(hidden_states, cos, sin, attention_mask=attn_mask)

        hidden_states = self.final_norm(hidden_states)

        # Sequence pooling: representation of [CLS] token (index 0)
        cls_repr = hidden_states[:, 0, :]

        # Mean pooling for question representation (ignoring padding)
        if attention_mask is not None:
            # dtype-preserving: keeps pooled_repr in the model's dtype (fp32/bf16/fp16)
            mask_expanded = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
            pooled_repr = (hidden_states * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1.0)
        else:
            pooled_repr = hidden_states.mean(dim=1)

        # 1. Choice logits (scaled by temperature)
        choice_logits = self.choice_head(pooled_repr) / self.temperature.clamp(min=0.1)

        # 2. Noul logits (scalar logit for calibrated boolean judgment)
        noul_logits = self.noul_head(pooled_repr).squeeze(-1) / self.temperature.clamp(min=0.1)
        noul_probs = torch.sigmoid(noul_logits)

        # 3. Score logits over rubric levels (softmax restricted per question at decode time)
        score_logits = self.score_head(pooled_repr)

        # 4. Extract Span logits for each token in sequence
        start_logits = self.extract_start_head(hidden_states).squeeze(-1)
        end_logits = self.extract_end_head(hidden_states).squeeze(-1)

        if attention_mask is not None:
            start_logits = start_logits + (1.0 - attention_mask.float()) * -10000.0
            end_logits = end_logits + (1.0 - attention_mask.float()) * -10000.0

        out = {
            "hidden_states": hidden_states,
            "pooled_repr": pooled_repr,
            "choice_logits": choice_logits,
            "noul_logits": noul_logits,
            "noul_probs": noul_probs,
            "score_logits": score_logits,
            "start_logits": start_logits,
            "end_logits": end_logits,
            "mlm_logits": self.lm_head(hidden_states),
        }

        # 5. Dynamic option scoring: gather [OPT]-marker hidden states and score
        # them against the pooled state representation in the same pass.
        if opt_positions is not None:
            # opt_positions: [B, K] long tensor of [OPT] token indices (-1 = absent)
            K = opt_positions.size(1)
            idx = opt_positions.clamp(min=0).unsqueeze(-1).expand(-1, -1, hidden_states.size(-1))
            opt_reprs = hidden_states.gather(1, idx)  # [B, K, H]
            valid = (opt_positions >= 0).unsqueeze(-1).to(hidden_states.dtype)
            opt_reprs = opt_reprs * valid  # zero out absent slots
            out["dynamic_opt_logits"] = self.dynamic_opt_head(pooled_repr, opt_reprs)

        return out
