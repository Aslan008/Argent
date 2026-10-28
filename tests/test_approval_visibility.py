"""Авто-одобрение должно быть видно, а не тонуть в фоне.

В /vibe решение принимается без человека, и единственный след этого — одна строка
в чате. Онa печаталась через `[dim]`: на тёмной теме dim уводит цвет почти в фон, и
строка «✓ Авто-одобрено» перестаёт читаться — то есть единственное доказательство,
что команда прошла сама, исчезало. Тест держит цвет на месте: тусклых строк
авто-решения быть не должно, отказ — тем более.
"""

import pytest

import argent.approval as approval
import argent.ui


class _Recorder:
    """Собирает то, что консоль напечатала, не трогая настоящий вывод."""

    def __init__(self):
        self.lines = []

    def print(self, *args, **kwargs):
        self.lines.append(" ".join(str(a) for a in args))


@pytest.fixture
def recorder(monkeypatch):
    rec = _Recorder()
    monkeypatch.setattr(argent.ui, "console", rec)
    monkeypatch.setattr(approval, "_nobody_is_watching", lambda: False)
    yield rec
    approval.set_policy(approval.POLICY_ASK)


def test_an_auto_approved_action_is_printed_not_dimmed(recorder):
    approval.set_policy(approval.POLICY_AUTO)

    assert approval.request_approval("выполнить команду: git status") is True

    assert recorder.lines, "строка авто-одобрения не напечатана"
    line = recorder.lines[-1]
    assert "Авто-одобрено" in line, line
    assert "[dim]" not in line, f"строка сливается с фоном: {line}"
    assert "dim" not in line, f"строка сливается с фоном: {line}"


def test_an_auto_denied_action_is_printed_not_dimmed(recorder):
    approval.set_policy(approval.POLICY_AUTO)

    assert approval.request_approval("rm -rf everything", destructive=True) is False

    assert recorder.lines, "строка авто-отклонения не напечатана"
    line = recorder.lines[-1]
    assert "Авто-отклонено" in line, line
    assert "dim" not in line, f"отказ не виден на фоне: {line}"
