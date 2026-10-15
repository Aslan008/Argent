"""Инспекция чекпоинта leeth_best.pt: ключи головы + config."""
import sys

if "model_leeth" not in sys.modules:
    from src.leeth import model as _leeth_model

    sys.modules["model_leeth"] = _leeth_model

import torch

ck = torch.load("src/leeth/data_400m/leeth_best.pt", map_location="cpu", weights_only=False)
print("top-level keys:", list(ck.keys()))
sd = ck["state_dict"]
head_keys = [k for k in sd if "opt" in k or "head" in k]
print("head/opt keys in state_dict:", head_keys[:20])
print("config type:", type(ck.get("config")))
cfg = ck.get("config")
if isinstance(cfg, dict):
    print("config keys:", list(cfg.keys()))
else:
    print("config:", cfg)
print("n_state_dict_keys:", len(sd))