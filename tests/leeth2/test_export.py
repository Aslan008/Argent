"""ONNX export keeps the exact function of the PyTorch model, including variable batch, length and option count."""

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from src.leeth2 import export as X


def test_export_is_numerically_identical_and_shapes_are_dynamic(tiny_model, tmp_path):
    p = X.export(tiny_model, tmp_path / "m.onnx")
    assert p.exists() and p.stat().st_size > 10_000
    worst = X.parity(tiny_model, p)
    assert all(v < 5e-4 for v in worst.values()), worst
    # a shape never used during tracing still works (more options, longer sequence)
    sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
    ex = X.example_inputs(tiny_model.cfg.vocab_size, tiny_model.cfg.n_markers, B=2, L=57, K=9)
    out = sess.run(X.OUTPUTS, {k: ex[k].numpy() for k in X.INPUTS})
    assert out[0].shape == (2, 9) and out[3].shape == (2, 57)
    # masked option slots stay masked
    ex["n_opts"] = ex["n_opts"] * 0 + 3
    out = sess.run(X.OUTPUTS, {k: ex[k].numpy() for k in X.INPUTS})
    assert (out[0][:, 3:] <= -1e8).all() and (out[0][:, :3] > -1e8).all()
