"""The whole closed-loop evaluation, new games and sites included, runs end to end with a (tiny, untrained) model -- before it is trusted on Kaggle."""

import shutil
from pathlib import Path

import pytest

TOK_DIR = Path(".argent/models/mmbert-argent")
pytestmark = pytest.mark.skipif(not (TOK_DIR / "tokenizer.json").exists(), reason="mmBERT tokenizer not on disk")


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    import torch
    from transformers import ModernBertConfig

    from src.leeth2.model import Leeth2Config, Leeth2Model

    d = tmp_path_factory.mktemp("tiny_player")
    bb = ModernBertConfig(vocab_size=256000, hidden_size=32, num_hidden_layers=3, num_attention_heads=4, intermediate_size=48, max_position_embeddings=1024,
                          global_attn_every_n_layers=3, local_attention=16, pad_token_id=0, bos_token_id=2, eos_token_id=1, cls_token_id=2, sep_token_id=1).to_dict()
    torch.manual_seed(0)
    Leeth2Model(Leeth2Config(backbone=bb, vocab_size=256000, hidden_size=32)).save_pretrained(d)
    for name in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copy(TOK_DIR / name, d / name)
    return d


def test_evaluate_model_covers_every_game_and_site(tiny_model):
    from src.leeth2 import gameeval
    from src.leeth2.envs2 import ENVS_MORE, ENVS_UNSEEN
    from src.leeth2.webenv2 import WEB2_ENVS, WEB2_UNSEEN

    rep = gameeval.evaluate_model(tiny_model, games_per_env=3, step_n=24, device="cpu")
    games = rep["games"]
    for group in (ENVS_MORE, ENVS_UNSEEN, WEB2_ENVS, WEB2_UNSEEN):
        for name in group:
            assert name in games, name
            assert games[name]["decisions"] > 0
    assert "step_sokoban" in rep["steps"] and "step_web2_checkout" in rep["steps"]
    base = gameeval.baselines(3)
    text = gameeval.table({"tiny": rep}, base)
    assert "web2_checkout" in text and "[UNSEEN]" in text and "HARM" in text


def test_baselines_know_the_expert_and_random_for_the_new_envs():
    from src.leeth2 import gameeval

    base = gameeval.baselines(6)
    assert base["expert"]["sokoban"]["solved"] == 1.0
    assert base["expert"]["web2_login"]["success"] == 1.0 and base["random"]["web2_checkout"]["harm"] > 0.2
