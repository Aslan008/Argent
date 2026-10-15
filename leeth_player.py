"""
leeth_player.py — Inference wrapper for the fine-tuned Leeth model (DynamicOptionHead on mmBERT).

Loads:
- Base: .argent/models/mmbert-base (ModernBertForMaskedLM weights via pytorch_model.bin)
- Tokenizer: .argent/models/mmbert-argent
- Checkpoint: .argent/models/mmbert_dynamic_opt_v1.pt
  - Applies unfrozen layers (e.g. layer_20, layer_21) to the base model.
  - Instantiates DynamicOptHead(768, proj_dim=512) and loads weights.
  - Uses marker_ids = [892, 31637, 235307] for [OPT] options.

Strictly follows the prompt's inference algorithm:
1. Assembles prompt: [STATE] ... [QUESTION] Какой ход оптимален? [OPT] ...
2. Tokenizes (padding max_length=256), forward through base.base_model
3. Locates marker tokens [OPT]
4. Mean pools [0..first_marker) for state_repr
5. Extracts hidden vectors at marker positions for opt_reprs
6. Computes logits via dot(state_mlp(state_repr), opt_proj(opt_repr)) / sqrt(512)
7. Returns softmax probabilities and argmax prediction
"""

import math
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForMaskedLM, AutoTokenizer


class DynamicOptHead(nn.Module):
    """Zero-shot Dynamic Option Scoring Head."""

    def __init__(self, hidden_size: int = 768, proj_dim: int = 512):
        super().__init__()
        self.state_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, proj_dim),
        )
        self.opt_proj = nn.Linear(hidden_size, proj_dim, bias=False)
        self.proj_dim = proj_dim

    def forward(self, state_repr: torch.Tensor, opt_reprs: torch.Tensor) -> torch.Tensor:
        """Args:

        state_repr: [B, H] pooled state representation.
        opt_reprs: [B, K, H] hidden vectors at [OPT] markers.

        Returns:
            [B, K] scalar logits for each option.
        """
        s_proj = self.state_mlp(state_repr)  # [B, P]
        o_proj = self.opt_proj(opt_reprs)    # [B, K, P]
        # dot product per option scaled by 1/sqrt(P)
        logits = (s_proj.unsqueeze(1) * o_proj).sum(dim=-1) / math.sqrt(self.proj_dim)
        return logits


class LeethSnakePlayer:
    """Singleton/reusable inference engine for Leeth in Snake."""

    def __init__(
        self,
        base_dir: str = ".argent/models/mmbert-argent",
        tok_dir: str = ".argent/models/mmbert-argent",
        ckpt_path: str = ".argent/models/mmbert_dynamic_opt_v1.pt",
        device: Optional[str] = None,
    ):
        self.project_root = Path(__file__).resolve().parent
        self.base_dir = self.project_root / base_dir
        self.tok_dir = self.project_root / tok_dir
        self.ckpt_path = self.project_root / ckpt_path

        if device is None:
            self.device = "cpu"
        else:
            self.device = device

        self.model = None
        self.tokenizer = None
        self.head = None
        self.marker_ids: List[int] = [892, 31637, 235307]
        self._load_model()

    def _load_model(self) -> None:
        """Loads base transformer and patches with checkpoint weights."""
        # 1. Base model (loads pytorch_model.bin from .argent/models/mmbert-argent)
        load_path = str(self.base_dir if self.base_dir.exists() else self.tok_dir)
        self.base = AutoModelForMaskedLM.from_pretrained(load_path, use_safetensors=False, dtype=torch.float32).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.tok_dir))

        # 2. Checkpoint weights
        ckpt = torch.load(str(self.ckpt_path), map_location="cpu")
        layers = self.base.base_model.layers

        unfrozen = ckpt.get("unfrozen_layers", {})
        for name, sd in unfrozen.items():
            layer_idx = int(name.split("_")[1])
            layers[layer_idx].load_state_dict(sd)

        # 3. Dynamic option head
        proj_dim = ckpt.get("proj_dim", 512)
        hidden_size = ckpt.get("hidden_size", 768)
        self.head = DynamicOptHead(hidden_size, proj_dim).eval()
        self.head.load_state_dict(ckpt["opt_head"])

        self.marker_ids = ckpt.get("marker_ids", [892, 31637, 235307])

        if self.device != "cpu":
            self.base = self.base.to(self.device)
            self.head = self.head.to(self.device)

    def predict(
        self,
        state_text: str,
        options: Optional[List[str]] = None,
        shuffle: bool = True,
        rng: Optional[random.Random] = None,
    ) -> Dict[str, Any]:
        """Runs single-step inference for snake state.

        Args:
            state_text: The ASCII grid string (e.g. 'змейка 6x6, H=голова, F=еда, B=тело\n...')
            options: List of directions (default: ["вверх", "вниз", "влево", "вправо"])
            shuffle: If True, shuffles option order in prompt (as in training)
            rng: Random generator for shuffle

        Returns:
            dict containing:
                - 'best_move': str (highest probability option)
                - 'probabilities': Dict[str, float] (option -> prob)
                - 'logits': Dict[str, float]
                - 'confidence': float (max probability)
                - 'prompt': str
        """
        if options is None:
            options = ["вверх", "вниз", "влево", "вправо"]

        eval_options = list(options)
        if shuffle:
            r = rng or random.Random()
            r.shuffle(eval_options)

        # 1. Assemble prompt
        parts = [f"[STATE] {state_text}", "[QUESTION] Какой ход оптимален?"]
        for opt in eval_options:
            parts.append(f"[OPT] {opt}")
        prompt = " ".join(parts)

        # 2. Tokenize with padding up to 256
        enc = self.tokenizer(
            prompt,
            max_length=256,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        if self.device != "cpu":
            enc = {k: v.to(self.device) for k, v in enc.items()}

        ids = enc["input_ids"][0].tolist()
        m_len = len(self.marker_ids)
        marker_positions = [
            i for i in range(len(ids) - m_len + 1)
            if ids[i : i + m_len] == self.marker_ids
        ]

        if len(marker_positions) != len(eval_options):
            # Fallback if truncation or tokenization split markers
            raise ValueError(
                f"Expected {len(eval_options)} markers, found {len(marker_positions)} in token sequence."
            )

        # 3. Forward pass through base encoder
        with torch.no_grad():
            hs = self.base.base_model(**enc).last_hidden_state  # [1, L, 768]
            # State repr: mean pool of tokens before first marker
            state_repr = hs[:, : marker_positions[0], :].mean(dim=1)  # [1, 768]
            # Opt reprs: hidden states at exact marker positions
            opt_reprs = hs[:, marker_positions, :]  # [1, K, 768]

            logits_t = self.head(state_repr, opt_reprs)[0]  # [K]
            probs_t = torch.softmax(logits_t, dim=-1)

        logits_list = logits_t.cpu().tolist()
        probs_list = probs_t.cpu().tolist()

        prob_dict = {opt: float(p) for opt, p in zip(eval_options, probs_list)}
        logit_dict = {opt: float(l) for opt, l in zip(eval_options, logits_list)}

        best_move = max(prob_dict, key=prob_dict.get)
        confidence = prob_dict[best_move]

        return {
            "best_move": best_move,
            "probabilities": prob_dict,
            "logits": logit_dict,
            "confidence": confidence,
            "prompt": prompt,
        }
