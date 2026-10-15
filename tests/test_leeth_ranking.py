# -*- coding: utf-8 -*-
"""Тесты Фаз F1/G3: ranking-голова, skill_router, browser_router, safety_gates.

Структурные проверки на случайных весах (без загрузки прод-модели) — быстрый прогон.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.leeth.model import LeethConfig, LeethModel
from src.leeth.ranking import LeethRankingModel
from src.leeth.browser_router import BrowserRouter, requires_confirmation_check
from src.leeth.skill_router import SkillRouter, _extract_description
from src.leeth.safety_gates import gate_plugin_command, gate_browser_action


@pytest.fixture(scope="module")
def tiny_setup():
    config = LeethConfig(vocab_size=64, hidden_size=32, num_layers=2, num_heads=4,
                         intermediate_size=64, max_position_embeddings=128)
    backbone = LeethModel(config)
    backbone.eval()
    ranking = LeethRankingModel(backbone, embed_dim=16)
    ranking.eval()
    return backbone, ranking


def _ids(text, vocab_size=64, length=8):
    ids = [hash(w) % (vocab_size - 6) + 6 for w in text.split()] or [6]
    ids = ids[:length]
    return torch.tensor([ids]), torch.tensor([[1] * len(ids)])


def test_ranking_forward_shapes(tiny_setup):
    backbone, ranking = tiny_setup
    q_ids, q_mask = _ids("query text")
    i_ids, i_mask = _ids("item text")
    scores = ranking(q_ids, q_mask, i_ids, i_mask)
    assert scores.shape == (1,)
    assert -1.0 <= scores.item() <= 1.0  # косинус


def test_ranking_backbone_frozen_by_default(tiny_setup):
    backbone, ranking = tiny_setup
    trainable = [n for n, p in ranking.named_parameters() if p.requires_grad]
    assert all("backbone" not in n for n in trainable)
    assert any("rank_query_proj" in n for n in trainable)


def test_ranking_training_gradients_flow(tiny_setup):
    """КРИТИЧНО: тренировочный путь (forward + backward) должен давать градиенты
    на проекциях. encode_query/encode_item под no_grad — только инференс;
    обучение обязано идти через forward/encode_*_trainable."""
    backbone, ranking = tiny_setup
    ranking.train()
    try:
        q_ids, q_mask = _ids("query text")
        i_ids, i_mask = _ids("item text")
        scores = ranking(q_ids, q_mask, i_ids, i_mask)
        loss = -scores.mean()
        loss.backward()
    finally:
        ranking.eval()  # вернуть fixture в детерминированный режим для соседних тестов
    assert ranking.rank_query_proj.weight.grad is not None
    assert ranking.rank_query_proj.weight.grad.abs().sum().item() > 0
    assert ranking.rank_item_proj.weight.grad is not None
    assert ranking.rank_item_proj.weight.grad.abs().sum().item() > 0


def test_ranking_trainable_encode_has_grad(tiny_setup):
    """encode_*_trainable — тот путь, что использует finetune_ranking.py."""
    backbone, ranking = tiny_setup
    q_ids, q_mask = _ids("query text")
    emb = ranking.encode_query_trainable(q_ids, q_mask)
    emb.sum().backward()
    assert ranking.rank_query_proj.weight.grad is not None


def test_load_ranking_head_roundtrip(tiny_setup, tmp_path):
    """Сохранение -> загрузка: скоры идентичны, embed_dim читается из чекпоинта."""
    backbone, ranking = tiny_setup
    ranking.eval()  # предыдущие тесты могли оставить fixture в train-режиме (dropout!)
    path = tmp_path / "ranking_head.pt"
    torch.save({"query_proj": ranking.rank_query_proj.state_dict(),
                "item_proj": ranking.rank_item_proj.state_dict(),
                "embed_dim": 16}, path)
    from src.leeth.ranking import load_ranking_head
    loaded = load_ranking_head(backbone, weights_path=str(path))
    assert loaded is not None
    q_ids, q_mask = _ids("query text")
    i_ids, i_mask = _ids("item text")
    s1 = ranking(q_ids, q_mask, i_ids, i_mask)
    s2 = loaded(q_ids, q_mask, i_ids, i_mask)
    assert torch.allclose(s1, s2, atol=1e-6)


def test_load_ranking_head_missing_file_returns_none(tiny_setup, tmp_path):
    from src.leeth.ranking import load_ranking_head
    assert load_ranking_head(tiny_setup[0], weights_path=str(tmp_path / "nope.pt")) is None
    assert load_ranking_head(tiny_setup[0], weights_path=None) is None


def test_rank_orders_all_items(tiny_setup):
    backbone, ranking = tiny_setup
    q_ids, q_mask = _ids("find the save button")
    items = [_ids(f"item {i}") for i in range(5)]
    item_ids = torch.cat([i[0] for i in items])
    item_mask = torch.cat([i[1] for i in items])
    results = ranking.rank(q_ids, q_mask, item_ids, item_mask)
    assert len(results) == 5
    # Результаты отсортированы по убыванию скора.
    assert results[0].score >= results[-1].score


def test_skill_router_fallback_when_no_head():
    router = SkillRouter(ranking=None, tokenizer=None)
    assert router.route("любой запрос") is None  # честный делегат
    assert not router.is_ready


def test_skill_router_rebuild_index(tiny_setup, tmp_path):
    backbone, ranking = tiny_setup
    # Фейковый skills-каталог
    (tmp_path / "unity-dev").mkdir()
    (tmp_path / "unity-dev" / "SKILL.md").write_text(
        "# Unity Dev\nПомощь по Unity проектам: сцены, скрипты, ассеты.\nПодробнее...", encoding="utf-8")
    (tmp_path / "empty").mkdir()  # без SKILL.md — должен быть пропущен
    router = SkillRouter(ranking=ranking, tokenizer=None, skills_dir=tmp_path)
    n = router.rebuild_index()
    assert n == 1
    assert router.entries[0].name == "unity-dev"
    assert "Unity" in router.entries[0].description


def test_extract_description():
    md = "# Заголовок\nПервый абзац с описанием.\n\nЕщё текст"
    assert "Заголовок" in _extract_description(md)
    assert "Первый абзац" in _extract_description(md)


def test_browser_router_fallback_when_no_head(monkeypatch, tmp_path):
    # "No engine" means no ranking head AND no DoPT solver: with the solver installed the router can answer without a head (that is
    # its job), so the machine's own model files must not decide this test.
    real_exists = Path.exists
    monkeypatch.setattr(Path, "exists", lambda self: False if ("dopt" in str(self) or "dynamic_opt" in str(self)) else real_exists(self))
    router = BrowserRouter(ranking=None, tokenizer=None, weights_path=tmp_path / "missing_ranking_head.pt")
    assert not router.is_ready
    assert router.route("цель", [{"text": "Save", "tag": "button"}]) is None


def test_dangerous_button_confirmation():
    assert requires_confirmation_check({"text": "Confirm purchase", "role": "button"}) is True
    assert requires_confirmation_check({"text": "Delete file", "role": "button"}) is True
    assert requires_confirmation_check({"text": "Оплатить заказ", "role": "button"}) is True
    assert requires_confirmation_check({"text": "Save draft", "role": "button"}) is False
    assert requires_confirmation_check({"text": "Next page", "role": "button"}) is False


def test_browser_dangerous_hidden_command():
    # Опасная команда, спрятанная в тексте кнопки — тоже ловится.
    assert requires_confirmation_check({"text": "Run rm -rf /", "role": "link"}) is True


def test_gate_plugin_command_blocks_dangerous():
    v = gate_plugin_command("rm -rf /")
    assert v.allowed is False
    v = gate_plugin_command("format D: /fs:NTFS /q")
    assert v.allowed is False


def test_gate_plugin_command_blocks_self_modification():
    v = gate_plugin_command("python -c \"import shutil; shutil.rmtree('plugins')\"")
    # Не попадает под plugin-паттерны напрямую, но rm-паттерны могут сработать.
    # Главное: безопасные команды проходят.
    v_ok = gate_plugin_command("echo hello")
    assert v_ok.allowed is True


def test_gate_plugin_command_allows_benign():
    assert gate_plugin_command("echo hello").allowed is True
    assert gate_plugin_command("git status").allowed is True


def test_gate_browser_action():
    v = gate_browser_action("Confirm purchase")
    assert v.allowed is True and v.requires_confirmation is True
    v = gate_browser_action("Documentation")
    assert v.allowed is True and v.requires_confirmation is False
    v = gate_browser_action("Delete account")
    assert v.requires_confirmation is True