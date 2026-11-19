"""Чеклист в ответе агента должен выглядеть чеклистом, а не скобками.

Rich's Markdown не знает task list: `- [x]` доходил до экрана литералом.
Premium рисовал свои боксы, Classic — нет; здесь проверяется общий путь.
"""

import re

import pytest

import argent.ui as ui
from src.bridge.render import render_block

ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def render(md: str, style: str = "classic") -> str:
    return ANSI.sub("", render_block("markdown", {"text": md, "style": style}, 70))


class TestTaskItems:
    def test_done_and_pending_get_glyphs(self):
        out = render("- [x] собрать\n- [ ] проверить\n")
        assert "☑" in out and "☐" in out
        assert "[x]" not in out and "[ ]" not in out

    @pytest.mark.parametrize("mark", ["x", "X"])
    def test_capital_x_counts_as_done(self, mark):
        assert "☑" in render(f"- [{mark}] пункт\n")

    def test_a_plain_bullet_is_untouched(self):
        out = render("- обычный пункт\n")
        assert "обычный пункт" in out
        assert "☑" not in out and "☐" not in out

    def test_indented_tasks_are_marked_too(self):
        assert "☐" in render("  - [ ] вложенная задача\n")

    def test_the_transform_is_idempotent(self):
        once = ui._mark_task_items("- [x] a\n- [ ] b\n")
        assert ui._mark_task_items(once) == once
