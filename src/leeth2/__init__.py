"""Leeth 2 — a single calibrated System-One decision model (Choice / Noul / Score / Extract).

One bidirectional encoder (mmBERT-base, multilingual, 8k context) and four light heads.
Everything here runs both locally (tiny test models, CPU) and inside Kaggle kernels
(2x T4), so the training stack is unit-testable before any GPU quota is spent.

Layout
    schema     one record format for every dataset (``Sample``)
    encoding   Sample -> token ids with virtual marker tokens (``Leeth2Encoder``)
    model      backbone + heads (``Leeth2Model``)
    losses     per-primitive losses
    metrics    accuracy / ECE / Brier / F1 helpers
    calibrate  post-hoc temperature scaling
    evaluate   batched inference + per-task report
    data       dataset adapters, corpus builder
    train      DDP trainer for Kaggle
"""

__all__ = ["schema", "encoding", "model", "losses", "metrics"]
