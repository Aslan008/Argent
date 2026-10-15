# -*- coding: utf-8 -*-
"""tests/test_lmtrust_blindspots.py — LMTrust Mode A: blind spot регрессии.

Каждый тест закрывает слепую зону, найденную аудитом (Mode B) или
сгенерированную по матрице слоёв. Маркеры: @pytest.mark.lmtrust(layer, direction).

Правило LMTrust: тесты ДОЛЖНЫ уметь падать — здесь каждый тест привязан к
конкретному контракту, а не тавтология (антипаттерн AP4).
"""

import pytest
import torch
import torch.nn as nn

from src.leeth.contrastive import (
    choice_contrastive_loss,
    combined_loss,
    info_nce_loss,
    margin_loss,
    noul_contrastive_loss,
)
from src.leeth.ranking import LeethRankingModel, load_ranking_head
from src.leeth.safety_gates import gate_browser_action, gate_plugin_command

pytestmark = pytest.mark.timeout(120)


# ═══════════════════════════════════════════════════════════════════════════
# contrastive.py — чистые функции: L0, L1, L2, L4 (D1, D4, D10)
# ═══════════════════════════════════════════════════════════════════════════

class TestContrastiveAdversarial:
    """L4 × D1/D4: нарушение предположений (пустые тензоры, нулевая температура)."""

    @pytest.mark.lmtrust(layer="L4", direction="D1")
    def test_margin_loss_empty_tensors_returns_zero_not_nan(self):
        """Регрессия: F.relu(...).mean() по пустому тензору давал NaN."""
        out = margin_loss(torch.empty(0), torch.empty(0))
        assert torch.isfinite(out), "пустой вход дал не-число"
        assert out.item() == 0.0

    @pytest.mark.lmtrust(layer="L4", direction="D1")
    def test_noul_contrastive_empty_batch_finite(self):
        out = noul_contrastive_loss(torch.empty(0), torch.empty(0))
        assert torch.isfinite(out)

    @pytest.mark.lmtrust(layer="L4", direction="D1")
    def test_info_nce_rejects_zero_temperature(self):
        """Регрессия: temperature=0 давал inf/NaN логиты молча."""
        q = torch.randn(2, 4)
        with pytest.raises(ValueError, match="temperature"):
            info_nce_loss(q, torch.randn(2, 4), [torch.randn(2, 4)], temperature=0.0)

    @pytest.mark.lmtrust(layer="L4", direction="D1")
    def test_info_nce_rejects_negative_temperature(self):
        """Отрицательная температура переворачивает softmax — запрещена."""
        q = torch.randn(2, 4)
        with pytest.raises(ValueError, match="temperature"):
            info_nce_loss(q, torch.randn(2, 4), [torch.randn(2, 4)], temperature=-0.5)

    @pytest.mark.lmtrust(layer="L4", direction="D4")
    def test_info_nce_empty_batch_returns_zero_not_nan(self):
        """Регрессия: cross_entropy по 0 элементов давал NaN."""
        out = info_nce_loss(torch.empty(0, 4), torch.empty(0, 4), [torch.empty(0, 4)])
        assert torch.isfinite(out) and out.item() == 0.0

    @pytest.mark.lmtrust(layer="L4b", direction="D4")
    def test_info_nce_rejects_empty_negatives(self):
        """Регрессия: без negatives лосс молча возвращал 0 (нет сигнала)."""
        q = torch.randn(2, 4)
        with pytest.raises(ValueError, match="negative"):
            info_nce_loss(q, torch.randn(2, 4), [])

    @pytest.mark.lmtrust(layer="L4b", direction="D4")
    def test_info_nce_rejects_zero_width_neg_stack(self):
        """Регрессия: стек [bsz, 0, dim] проходил и давал нулевой лосс."""
        q = torch.randn(2, 4)
        with pytest.raises(ValueError, match="neg_embs"):
            info_nce_loss(q, torch.randn(2, 4), torch.empty(2, 0, 4))

    @pytest.mark.lmtrust(layer="L1", direction="D7")
    def test_info_nce_accepts_prestacked_tensor(self):
        """API: готовый стек [bsz, n_neg, dim] эквивалентен списку."""
        torch.manual_seed(3)
        q = torch.randn(2, 4)
        pos = torch.randn(2, 4)
        negs = [torch.randn(2, 4), torch.randn(2, 4)]
        as_list = info_nce_loss(q, pos, negs)
        as_stack = info_nce_loss(q, pos, torch.stack(negs, dim=1))
        assert torch.isclose(as_list, as_stack)

    @pytest.mark.lmtrust(layer="L4", direction="D4")
    def test_choice_contrastive_rejects_out_of_bounds_targets(self):
        """Регрессия: gather по out-of-bounds падал без контекста."""
        logits = torch.randn(2, 3)
        with pytest.raises(ValueError, match="pos_targets"):
            choice_contrastive_loss(logits, torch.tensor([5, 0]), logits, torch.tensor([0, 1]))

    @pytest.mark.lmtrust(layer="L4", direction="D4")
    def test_choice_contrastive_rejects_negative_targets(self):
        """Отрицательный индекс молча собирал чужой класс — теперь ValueError."""
        logits = torch.randn(2, 3)
        with pytest.raises(ValueError, match="neg_targets"):
            choice_contrastive_loss(logits, torch.tensor([0, 1]), logits, torch.tensor([-1, 1]))

    @pytest.mark.lmtrust(layer="L4", direction="D10")
    def test_combined_loss_rejects_negative_weight(self):
        """Отрицательный вес инвертировал контрастивную цель."""
        with pytest.raises(ValueError, match="contrastive_weight"):
            combined_loss(torch.tensor(1.0), torch.tensor(2.0), contrastive_weight=-0.1)

    @pytest.mark.lmtrust(layer="L2", direction="D1")
    def test_margin_loss_boundary_exact_margin_is_zero(self):
        """Граница: score разрыв ровно = margin -> лосс 0 (hinge насыщен)."""
        pos = torch.tensor([1.5])
        neg = torch.tensor([1.0])
        assert margin_loss(pos, neg, margin=0.5).item() == pytest.approx(0.0, abs=1e-6)

    @pytest.mark.lmtrust(layer="L3", direction="D10")
    def test_margin_loss_nonneg_invariant(self):
        """Инвариант: hinge-лосс >= 0 на любых входах."""
        g = torch.Generator().manual_seed(7)
        for _ in range(50):
            pos = torch.randn(4, generator=g)
            neg = torch.randn(4, generator=g)
            assert margin_loss(pos, neg).item() >= 0.0

    @pytest.mark.lmtrust(layer="L1", direction="D7")
    def test_info_nce_exact_value(self):
        """Точное значение лосса ловит подмену /temperature на *temperature
        (мутант выживал на тестах 'loss < 1.0')."""
        q = torch.tensor([[1.0, 0.0]])
        pos = torch.tensor([[1.0, 0.0]])
        neg = torch.tensor([[1.0, 0.0]])  # negative с тем же скором, что positive
        # logits = [1/T, 1/T] -> CE = log(2)
        loss = info_nce_loss(q, pos, [neg], temperature=0.5)
        assert torch.isclose(loss, torch.tensor(0.6931), atol=1e-3), f"{loss.item()}"

    @pytest.mark.lmtrust(layer="L2", direction="D1")
    def test_combined_loss_zero_weight_allowed(self):
        """Граница: weight=0.0 валиден (чистый supervised), запрещён только < 0."""
        sup = torch.tensor(1.5)
        out = combined_loss(sup, torch.tensor(100.0), contrastive_weight=0.0)
        assert torch.isclose(out, sup)


# ═══════════════════════════════════════════════════════════════════════════
# ranking.py — stateful объект: L1, L5, L9 (D2, D4, D7)
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def tiny_ranking():
    torch.manual_seed(0)
    from src.leeth.model import LeethConfig, LeethModel
    cfg = LeethConfig(vocab_size=64, hidden_size=32, num_layers=2, num_heads=4)
    backbone = LeethModel(cfg)
    return LeethRankingModel(backbone, embed_dim=16, trainable_backbone=False)


def _tiny_backbone():
    from src.leeth.model import LeethConfig, LeethModel
    cfg = LeethConfig(vocab_size=64, hidden_size=32, num_layers=2, num_heads=4)
    return LeethModel(cfg)


def _ids(n_seqs, seq_len=8, vocab=64):
    return (torch.randint(1, vocab, (n_seqs, seq_len)), torch.ones((n_seqs, seq_len), dtype=torch.long))


class TestRankingContract:
    """L1 × D7: явные контракты вместо молчаливых срезов."""

    @pytest.mark.lmtrust(layer="L1", direction="D7")
    def test_rank_rejects_batch_of_queries(self, tiny_ranking):
        """Регрессия: rank() молча брал query_ids[:1] из батча."""
        q_ids, q_mask = _ids(3)
        i_ids, i_mask = _ids(5)
        with pytest.raises(ValueError, match="один запрос"):
            tiny_ranking.rank(q_ids, q_mask, i_ids, i_mask)

    @pytest.mark.lmtrust(layer="L1", direction="D7")
    def test_rank_accepts_unsqueezed_single_query(self, tiny_ranking):
        """Удобство: [seq] без batch-оси тоже валиден."""
        q_ids, q_mask = _ids(1)
        i_ids, i_mask = _ids(4)
        res = tiny_ranking.rank(q_ids[0], q_mask[0], i_ids, i_mask)
        assert len(res) == 4

    @pytest.mark.lmtrust(layer="L1", direction="D7")
    def test_forward_rejects_mismatched_batches(self, tiny_ranking):
        q_ids, q_mask = _ids(2)
        i_ids, i_mask = _ids(3)
        with pytest.raises(ValueError, match="не парны"):
            tiny_ranking.forward(q_ids, q_mask, i_ids, i_mask)


class TestRankingState:
    """L5 × D2: режимы train/eval не должны искажать инференс."""

    @pytest.mark.lmtrust(layer="L5", direction="D2")
    def test_rank_deterministic_even_in_train_mode(self, tiny_ranking):
        """Регрессия: rank() в train-режиме гонял dropout -> недетерминизм."""
        model = tiny_ranking
        model.train()
        q_ids, q_mask = _ids(1)
        i_ids, i_mask = _ids(4)
        s1 = [r.score for r in model.rank(q_ids, q_mask, i_ids, i_mask)]
        s2 = [r.score for r in model.rank(q_ids, q_mask, i_ids, i_mask)]
        assert s1 == s2, "dropout активен в инференсе: скоры плывут"

    @pytest.mark.lmtrust(layer="L5", direction="D2")
    def test_rank_restores_training_mode(self, tiny_ranking):
        """eval-гард не должен ломать состояние обучающего цикла."""
        model = tiny_ranking
        model.train()
        q_ids, q_mask = _ids(1)
        i_ids, i_mask = _ids(2)
        model.rank(q_ids, q_mask, i_ids, i_mask)
        assert model.training, "rank() не вернул train-режим"


class TestRankingFallback:
    """L4b × D4: fallback-контракт load_ranking_head."""

    @pytest.mark.lmtrust(layer="L4b", direction="D4")
    def test_load_corrupt_file_returns_none(self, tmp_path):
        """Регрессия: битый файл давал RuntimeError вместо None."""
        bad = tmp_path / "corrupt.pt"
        bad.write_bytes(b"not a torch file at all")
        assert load_ranking_head(_tiny_backbone(), str(bad)) is None

    @pytest.mark.lmtrust(layer="L4b", direction="D4")
    def test_load_foreign_state_dict_returns_none(self, tmp_path):
        """Регрессия: чужой state_dict (без ключей query_proj) -> None, не KeyError."""
        backbone = _tiny_backbone()
        foreign = tmp_path / "foreign.pt"
        torch.save({"some_key": torch.zeros(3)}, foreign)
        assert load_ranking_head(backbone, str(foreign)) is None

    @pytest.mark.lmtrust(layer="L2", direction="D6")
    def test_load_embed_dim_as_tensor(self, tmp_path):
        """embed_dim, сохранённый тензором, не должен ломать загрузку."""
        backbone = _tiny_backbone()
        ranking = LeethRankingModel(backbone, embed_dim=16)
        path = tmp_path / "head.pt"
        torch.save({
            "query_proj": ranking.rank_query_proj.state_dict(),
            "item_proj": ranking.rank_item_proj.state_dict(),
            "embed_dim": torch.tensor(16),
        }, path)
        loaded = load_ranking_head(backbone, str(path))
        assert loaded is not None


# ═══════════════════════════════════════════════════════════════════════════
# safety_gates.py — security: L4, L9 (D8, D4)
# ═══════════════════════════════════════════════════════════════════════════

class TestBrowserGatePrecedence:
    """L4 × D8: deny-лист (блок) старше dangerous-слов (подтверждение)."""

    @pytest.mark.lmtrust(layer="L4", direction="D8")
    def test_deny_command_hidden_in_button_text_is_blocked(self):
        """Регрессия приоритета: текст матчит ОБА списка (dangerous + deny).
        Старый порядок (dangerous первым) возвращал allowed=True с
        подтверждением вместо блока."""
        v = gate_browser_action("Confirm format C: drive")
        assert not v.allowed, f"деструктив в тексте кнопки разрешён: {v}"

    @pytest.mark.lmtrust(layer="L4", direction="D8")
    def test_dangerous_word_still_requires_confirmation(self):
        """Позитивный путь: опасное слово без деструктива -> подтверждение."""
        v = gate_browser_action("Confirm purchase")
        assert v.allowed and v.requires_confirmation

    @pytest.mark.lmtrust(layer="L9", direction="D3")
    def test_benign_button_allowed_without_confirmation(self):
        """Негативное пространство: обычная кнопка не требует подтверждения."""
        v = gate_browser_action("Documentation")
        assert v.allowed and not v.requires_confirmation


class TestPluginGateVerdicts:
    """L4b × D4: сбои CDCL не должны превращать блок в разрешение."""

    @pytest.mark.lmtrust(layer="L4b", direction="D4")
    def test_cdcl_unsat_with_none_core_still_blocks(self, monkeypatch):
        """Регрессия: core=None крэшил reason и молча разрешал команду."""
        from src.leeth import safety_gates as sg

        class FakeResult:
            sat = False
            core = None

        class FakeArb:
            def arbitrate(self, **kw):
                return FakeResult()

        monkeypatch.setattr(sg, "get_arbitrator", lambda: FakeArb())
        v = sg.gate_plugin_command("echo безопасная но UNSAT")
        assert not v.allowed, "UNSAT с core=None должен блокировать"

    @pytest.mark.lmtrust(layer="L1", direction="D7")
    def test_cdcl_pass_reports_cdcl_source(self, monkeypatch):
        """Регрессия: SAT-пропуск отчитывался как procedural."""
        from src.leeth import safety_gates as sg

        class FakeResult:
            sat = True
            core = []

        class FakeArb:
            def arbitrate(self, **kw):
                return FakeResult()

        monkeypatch.setattr(sg, "get_arbitrator", lambda: FakeArb())
        v = sg.gate_plugin_command("echo hello")
        assert v.allowed and v.source == "cdcl"

    @pytest.mark.lmtrust(layer="L4b", direction="D4")
    def test_solver_error_is_marked_cdcl_skipped(self, monkeypatch):
        """Fail-soft задокументирован: сбой солвера виден в reason."""
        from src.leeth import safety_gates as sg

        class BoomArb:
            def arbitrate(self, **kw):
                raise RuntimeError("solver down")

        monkeypatch.setattr(sg, "get_arbitrator", lambda: BoomArb())
        v = sg.gate_plugin_command("echo hello")
        assert v.allowed  # fail-soft: процедурный слой уже отработал
        assert "cdcl_skipped" in (v.reason or ""), "пропуск CDCL должен быть виден"

    @pytest.mark.lmtrust(layer="L4", direction="D8")
    def test_plugin_gate_blocks_deny_list(self):
        """Процедурный слой работает и без солвера (анкер ^ паттерна)."""
        v = gate_plugin_command("rd /s /q C:\\Windows")
        assert not v.allowed


class TestDenyDomainGaps:
    """L4 × D8: дыры deny-домена, найденные Mode E (формальное доказательство)."""

    @pytest.mark.lmtrust(layer="L4", direction="D8")
    def test_rm_long_form_recursive_blocked(self):
        """Регрессия: rm --recursive /home не матчился (long-form флаг)."""
        v = gate_plugin_command("rm --recursive /home")
        assert not v.allowed, "rm --recursive прошёл deny-лист"

    @pytest.mark.lmtrust(layer="L4", direction="D8")
    def test_rm_long_form_force_blocked(self):
        v = gate_plugin_command("rm --force important.db")
        assert not v.allowed

    @pytest.mark.lmtrust(layer="L4", direction="D8")
    def test_chained_rm_long_form_blocked(self):
        v = gate_plugin_command("git status && rm --recursive /home")
        assert not v.allowed

    @pytest.mark.lmtrust(layer="L4", direction="D8")
    def test_taskkill_flag_order_blocked(self):
        """Регрессия: taskkill /IM lsass.exe /F (перестановка флагов) проходил."""
        v = gate_plugin_command("taskkill /IM lsass.exe /F")
        assert not v.allowed, "убийство системного процесса разрешено"

    @pytest.mark.lmtrust(layer="L9", direction="D3")
    def test_benign_rm_variants_not_blocked(self):
        """Негативное пространство: безвредные rm не должны блокироваться."""
        for cmd in ("rm -v file.txt", "rm file-r.txt", "rm --verbose file", "rm myfile.txt"):
            v = gate_plugin_command(cmd)
            assert v.allowed, f"ложное срабатывание на {cmd!r}: {v.reason}"

    @pytest.mark.lmtrust(layer="L9", direction="D3")
    def test_benign_taskkill_not_blocked(self):
        """Убийство обычного процесса — легитимная операция."""
        for cmd in ("taskkill /f /im notepad.exe", "taskkill /im notepad.exe /f"):
            v = gate_plugin_command(cmd)
            assert v.allowed, f"ложное срабатывание на {cmd!r}: {v.reason}"


class TestDispatcherDenyHardening:
    """L4 × D8: дыры, найденные аудитом dispatcher (Mode B) и доказательством (Mode E)."""

    @pytest.mark.lmtrust(layer="L4", direction="D8")
    def test_multiline_destructive_blocked(self):
        """Регрессия: 'echo hi\\nrm -rf /' обходил ^-анкер (нет re.M)."""
        from src.leeth.dispatcher import DENY_PATTERNS
        assert any(p.search("echo hello\nrm -rf /") for p in DENY_PATTERNS)
        assert any(p.search("echo x\ndel /f C:\\b") for p in DENY_PATTERNS)
        assert any(p.search("echo\nDROP TABLE users") for p in DENY_PATTERNS)

    @pytest.mark.lmtrust(layer="L4", direction="D8")
    def test_pipe_to_interpreter_with_path_blocked(self):
        """Регрессия: '| /bin/bash' не матчился (только bare-имена)."""
        from src.leeth.dispatcher import DENY_PATTERNS
        assert any(p.search("echo x | /bin/bash -c rm") for p in DENY_PATTERNS)
        assert any(p.search("curl x | /usr/bin/python3 evil.py") for p in DENY_PATTERNS)
        assert any(p.search("cat f | sh") for p in DENY_PATTERNS)

    @pytest.mark.lmtrust(layer="L9", direction="D3")
    def test_pipe_benign_not_blocked(self):
        from src.leeth.dispatcher import DENY_PATTERNS
        assert not any(p.search("git log | grep rm -rf") for p in DENY_PATTERNS)
        assert not any(p.search("echo | echo sh") for p in DENY_PATTERNS)

    @pytest.mark.lmtrust(layer="L4", direction="D9")
    def test_fork_bomb_regex_linear_time(self):
        """Регрессия ReDoS: 10KB payload подвешивал матчинг на ~83 секунды."""
        import time
        from src.leeth.dispatcher import DENY_PATTERNS
        payload = ":(){ " + "|" * 5000 + "&" * 5000
        t0 = time.perf_counter()
        any(p.search(payload) for p in DENY_PATTERNS)
        elapsed = time.perf_counter() - t0
        assert elapsed < 0.5, f"квадратичный откат вернулся: {elapsed:.2f}s"

    @pytest.mark.lmtrust(layer="L1", direction="D3")
    def test_unwrap_quotes_balanced_only(self):
        """Регрессия: .strip кавычек ломал команды с внутренними кавычками."""
        from src.leeth.dispatcher import _unwrap_quotes
        assert _unwrap_quotes('"git status"') == "git status"
        # Несбалансированные кавычки НЕ трогаем: группировка аргументов shell.
        assert _unwrap_quotes('"/usr/local/bin/my app" --help') == '"/usr/local/bin/my app" --help'
        assert _unwrap_quotes("'привет'") == "привет"
        assert _unwrap_quotes("plain command") == "plain command"

    @pytest.mark.lmtrust(layer="L1", direction="D3")
    def test_russian_connector_longest_first(self):
        """Регрессия: 'и ещё X' обрезался до 'ещё X' (альтернатива 'и' раньше)."""
        from src.leeth.dispatcher import LeethDispatcher
        d = LeethDispatcher.get_instance()
        rest = d._strip_leading_span("и ещё найди файл", "найди файл")
        assert "ещё" not in rest.split("найди")[0], f"коннектор обрезан криво: {rest!r}"