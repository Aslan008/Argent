"""A compaction keeps what a paraphrase cannot: the person's words, their rules, the facts, a way back.

src/agent/compaction.py holds the parts that need no model; src/agent/trimmer.py builds the distillate from
them. The scenario that matters is the SECOND compaction: it used to rebuild the facts from the slice it was
compressing, summarise the previous distillate (file bodies included) as if it were conversation, and keep
only the last three requests — so what the first compaction knew faded. These tests run two in a row.
"""

import json
import os
from pathlib import Path

import pytest

import src.agent.trimmer as trimmer
from src.agent import compaction as C


def _msg(role, content, **kw):
    return {"role": role, "content": content, **kw}


def _call(name, **args):
    return {"id": f"c-{name}-{len(args)}", "function": {"name": name, "arguments": json.dumps(args)}}


# ── the archive ─────────────────────────────────────────────────────────────────────────────────────────

class TestArchive:
    def test_the_dropped_messages_are_written_whole(self, tmp_path):
        big = "line of output\n" * 5000                          # 75 000 chars: far beyond any summary cap
        msgs = [_msg("user", "запусти тесты"),
                _msg("assistant", "запускаю", tool_calls=[_call("run_command", command="pytest -q")]),
                _msg("tool", big, tool_call_id="c-run_command-1")]
        path = C.archive_transcript(msgs, 3, root=tmp_path)
        text = Path(path).read_text(encoding="utf-8")
        assert Path(path).parent == tmp_path / ".argent" / "transcripts" and path.endswith("-3.md")
        assert "запусти тесты" in text and "pytest -q" in text and "→ вызов run_command" in text
        assert text.count("line of output") > 1000              # the output is there (capped only at 40 000 chars)

    def test_an_earlier_distillate_is_not_copied_in(self, tmp_path):
        msgs = [_msg("system", "=== CONTEXT DISTILLATE (compression #1) ===\nold\n=== END DISTILLATE ==="),
                _msg("user", "новое")]
        text = Path(C.archive_transcript(msgs, 2, root=tmp_path)).read_text(encoding="utf-8")
        assert "новое" in text and "CONTEXT DISTILLATE" not in text

    def test_old_archives_are_pruned(self, tmp_path, monkeypatch):
        monkeypatch.setattr(C, "ARCHIVE_KEEP", 3)
        paths = []
        for i in range(6):
            paths.append(C.archive_transcript([_msg("user", f"message number {i}")], i, root=tmp_path))
            os.utime(paths[-1], (1000 + i, 1000 + i))
        left = sorted(p.name for p in (tmp_path / ".argent" / "transcripts").glob("compact-*.md"))
        assert len(left) == 3 and Path(paths[-1]).exists()

    def test_an_unwritable_place_costs_nothing(self, tmp_path):
        blocker = tmp_path / "file"
        blocker.write_text("x")
        assert C.archive_transcript([_msg("user", "hello there")], 1, root=blocker) is None


# ── what the person said ────────────────────────────────────────────────────────────────────────────────

class TestLedger:
    def test_only_the_persons_words_count(self):
        msgs = [_msg("user", "Сделай экспорт в CSV для отчётов"),
                _msg("user", "ок"),
                _msg("user", "/compact"),
                _msg("user", "You stopped after thinking, without producing anything. Either CALL A TOOL"),
                _msg("user", "[Heartbeat] проверка"),
                _msg("assistant", "Готово"),
                _msg("user", "Теперь добавь тест\nна пустой файл")]
        assert C.user_texts(msgs) == ["Сделай экспорт в CSV для отчётов", "Теперь добавь тест\nна пустой файл"]
        assert C.ledger_entries(C.user_texts(msgs))[1] == "Теперь добавь тест ⏎ на пустой файл"

    def test_a_long_brief_is_kept_far_beyond_the_old_500_characters(self):
        brief = "Требование к модулю. " * 40                       # ~840 chars
        assert C.ledger_entries([brief])[0].startswith("Требование к модулю. Требование") and len(C.ledger_entries([brief])[0]) > 800

    def test_over_budget_old_entries_shrink_first_then_the_middle_goes(self):
        entries = [f"сообщение {i}: " + "слово " * 120 for i in range(40)]
        fitted = C.fit_ledger(entries, budget=3000)
        assert sum(len(e) + 6 for e in fitted) <= 3000 + 200
        assert fitted[0] == entries[0] and fitted[-1] == entries[-1]          # the first task and the latest request
        assert any(e.startswith("… пропущено") for e in fitted)

    def test_roundtrip_through_the_rendered_block(self):
        entries = ["первое", "второе ⏎ с переносом", "… пропущено сообщений: 3 …", "последнее"]
        assert C.parse_ledger(C.render_ledger(entries)) == entries


class TestRules:
    def test_standing_instructions_are_picked_out(self):
        rules = C.extract_rules([
            "Нужно починить сборку. Всё запускай исключительно на Kaggle. Потом посмотрим.",
            "Не коммить и не пушь без просьбы.\nИ ещё: never touch the vendored folder",
            "Добавь кнопку в шапку."])
        joined = " | ".join(rules)
        assert "исключительно на Kaggle" in joined and "Не коммить" in joined and "never touch" in joined
        assert "кнопку" not in joined and "починить сборку" not in joined

    def test_the_latest_wording_wins_and_the_list_is_bounded(self):
        merged = C.merge_rules(["Не коммить без просьбы."], ["не коммить без просьбы!"] + [f"Никогда не делай {i} пункт" for i in range(30)])
        assert len(merged) == C.RULES_LIMIT
        assert merged[-1].endswith("29 пункт")


class TestFacts:
    def test_facts_merge_across_compactions_and_survive_a_render_parse_roundtrip(self):
        first = C.collect_facts([_msg("assistant", "Правлю модуль конфигурации проекта", tool_calls=[
            _call("write_file", file_path="src/a.py"), _call("run_command", command="pytest tests/a")])])
        second = C.collect_facts([_msg("assistant", "x", tool_calls=[_call("replace_in_file", file_path="src/b.py")]),
                                  _msg("tool", "Traceback (most recent call last):\nValueError: boom")])
        rendered = C.render_facts(C.merge_facts(first, second), include_requests=False)
        again = C.parse_facts(rendered)
        assert again.written == ["src/a.py", "src/b.py"]
        assert "pytest tests/a" in again.commands and again.errors == ["ValueError: boom"]


# ── writing and reading a distillate ────────────────────────────────────────────────────────────────────

class TestDistillateRoundtrip:
    def test_everything_written_is_read_back(self):
        text = C.assemble(goal="цель", summary="## Done\n- сделано", archive=["/p/compact-1.md"],
                          ledger=["a", "b"], rules=["Не коммить."], facts="**Изменённые файлы:** src/a.py",
                          files="### a.py\n```\nBODY\n```", extra=[f"{C.L_MEMORY}\nпамять"])
        parts = C.split_distillate(text)
        assert parts.summary == "## Done\n- сделано"
        assert parts.archive == ["/p/compact-1.md"] and parts.ledger == ["a", "b"] and parts.rules == ["Не коммить."]
        assert parts.facts.written == ["src/a.py"]
        assert "BODY" not in text.split(C.L_FILES)[0] and "BODY" not in parts.summary

    def test_the_older_unstructured_format_still_yields_summary_and_facts(self):
        legacy = ("**ACTIVE GOAL:** цель\n\n## Done\n- было\n\n\n---\n**Верифицированные факты:**\n"
                  "**Изменённые файлы:** old.py\n\n\n---\n**Актуальное содержимое ключевых файлов:**\n### old.py\nX")
        parts = C.split_distillate(legacy)
        assert "## Done" in parts.summary and "ACTIVE GOAL" not in parts.summary
        assert parts.facts.written == ["old.py"]

    def test_nothing_in_nothing_out(self):
        assert C.split_distillate(None).summary == "" and C.split_distillate("").ledger == []


# ── two compactions in a row ────────────────────────────────────────────────────────────────────────────

def _turns(start, count, note=""):
    msgs = []
    for i in range(start, start + count):
        msgs.append(_msg("user", f"Задача номер {i}: поправь модуль_{i}.py {note}"))
        msgs.append(_msg("assistant", f"Поправил модуль_{i}.py, тесты проходят, дальше идём."))
    return msgs


class TestTwoCompactionsInARow:
    @pytest.fixture
    def run(self, monkeypatch, tmp_path):
        self.prompts = []
        self.answers = ["## Intent\n- первый заход\n## Done\n- шаг один", "## Intent\n- первый и второй заход\n## Done\n- шаг один и два"]

        def fake_summary(pre, goal, prior_context, model, timeout=120.0, ui=None):
            self.prompts.append((pre, prior_context))
            return self.answers.pop(0) if self.answers else None

        monkeypatch.setattr(trimmer, "_summarize_with_llm", fake_summary)
        self.project = tmp_path
        return lambda msgs: trimmer.soft_trim_with_summarization(msgs, "mock", 20, 30000)

    def _distillate(self, msgs):
        found = [m["content"] for m in msgs if "CONTEXT DISTILLATE" in m.get("content", "")]
        assert len(found) == 1
        return found[0]

    def test_the_second_compaction_forgets_nothing_the_first_knew(self, run):
        edited = self.project / "core.py"
        edited.write_text("BODY_OF_CORE = 1\n", encoding="utf-8")
        first = [_msg("system", "sys"),
                 _msg("user", "Всё запускай исключительно на Kaggle, а не локально. Начнём с модуля core."),
                 _msg("assistant", "Правлю core", tool_calls=[_call("write_file", file_path=str(edited)),
                                                              _call("run_command", command="pytest tests/core")]),
                 _msg("tool", "Traceback (most recent call last):\nKeyError: 'depth'", tool_call_id="x")]
        first += _turns(0, 24)
        once = run(first)
        d1 = self._distillate(once)
        assert "исключительно на Kaggle" in d1 and "шаг один" in d1 and str(edited) in d1
        assert "BODY_OF_CORE" in d1                                   # the file in play is re-read from disk

        second = once + _turns(100, 24, note="уже на втором круге")
        twice = run(second)
        d2 = self._distillate(twice)

        assert d2.count("исключительно на Kaggle") >= 1                # the rule outlived the first summary
        assert "Начнём с модуля core" in d2                            # and so did the person's first words
        assert "Задача номер 100" in d2 or "Задача номер 101" in d2     # alongside the new ones
        assert str(edited) in d2 and "pytest tests/core" in d2 and "KeyError: 'depth'" in d2   # facts carried over
        assert "шаг один и два" in d2
        parts = C.split_distillate(d2)
        assert len(parts.archive) == 2 and all(Path(p).exists() for p in parts.archive)

    def test_the_summariser_updates_the_old_summary_and_is_not_fed_file_bodies(self, run):
        core = self.project / "core.py"
        core.write_text("SECRET_FILE_BODY_MARKER = 42\n", encoding="utf-8")
        msgs = [_msg("system", "sys"), _msg("user", "Начни работу над модулем"),
                _msg("assistant", "пишу", tool_calls=[_call("write_file", file_path=str(core))])] + _turns(0, 24)
        once = run(msgs)
        assert "SECRET_FILE_BODY_MARKER" in self._distillate(once)
        run(once + _turns(50, 24))
        pre, prior_context = self.prompts[1]
        assert "шаг один" in prior_context                             # the previous SUMMARY is handed over
        assert "SECRET_FILE_BODY_MARKER" not in pre and "SECRET_FILE_BODY_MARKER" not in prior_context
        assert "CONTEXT DISTILLATE" not in pre                         # the old distillate is not "conversation"

    def test_when_the_model_fails_the_second_time_the_first_summary_is_kept(self, run):
        once = run([_msg("system", "sys")] + _turns(0, 25))
        self.answers.clear()                                           # the next call finds no model
        twice = run(once + _turns(40, 24))
        d2 = self._distillate(twice)
        assert "шаг один" in d2 and "Задача номер 40" in d2

    def test_when_there_never_was_a_summary_the_agent_is_told_so(self, run):
        self.answers.clear()
        d = self._distillate(run([_msg("system", "sys")] + _turns(0, 25)))
        assert "Сводку написать не удалось" in d and "Задача номер 0" in d

    def test_the_distillate_points_to_the_transcript_and_says_how_to_continue(self, run):
        d = self._distillate(run([_msg("system", "sys")] + _turns(0, 25)))
        assert "read_file" in d and "Current State" in d
        path = C.split_distillate(d).archive[-1]
        assert "Задача номер 0" in Path(path).read_text(encoding="utf-8")


class TestEmergencyReset:
    def test_carries_the_previous_summary_ledger_and_archive(self, monkeypatch):
        monkeypatch.setattr(trimmer, "_summarize_with_llm", lambda *a, **k: "## Done\n- было раньше")
        once = trimmer.soft_trim_with_summarization(
            [_msg("system", "sys"), _msg("user", "Никогда не трогай папку vendor, это важно.")] + _turns(0, 24), "mock", 20, 30000)
        reset = trimmer.hard_reset_with_memory(once + [_msg("user", "Что дальше?")], "mock")
        assert len(reset) == 3 and reset[-1]["content"] == "Что дальше?"
        d = reset[1]["content"]
        assert "emergency reset" in d and "было раньше" in d and "не трогай папку vendor" in d
        assert len(C.split_distillate(d.split("===\n", 1)[1]).archive) == 2          # the earlier one and the emergency one


# ── the summariser's input ──────────────────────────────────────────────────────────────────────────────

class TestSummarizerInput:
    def test_a_long_user_brief_reaches_the_summariser(self):
        brief = "ТРЕБОВАНИЕ " * 150                                    # 1650 chars; the old cap was 500
        text = trimmer._prepare_for_summarizer([_msg("user", brief)])
        assert text.count("ТРЕБОВАНИЕ") == 150

    def test_errors_keep_more_than_ordinary_output(self):
        plain = "ok line\n" * 300
        err = "Traceback (most recent call last):\n" + "frame\n" * 20 + "ValueError: the exact message\n" + "x\n" * 300
        out = trimmer._prepare_for_summarizer([_msg("tool", plain), _msg("tool", err)])
        assert "ValueError: the exact message" not in plain
        assert "Traceback (most recent call last)" in out

    def test_a_tiny_budget_drops_the_middle_not_the_ends(self):
        msgs = [_msg("user", f"сообщение {i} " + "текст " * 30) for i in range(60)]
        out = trimmer._prepare_for_summarizer(msgs, budget_chars=2500)
        assert "сообщение 0 " in out and "сообщение 59 " in out
        assert "пропущено сообщений из середины" in out and len(out) <= 2700


class TestRehydration:
    def test_recently_read_files_fill_the_free_places(self, tmp_path):
        edited, a, b, c = (tmp_path / n for n in ("edited.py", "a.py", "b.py", "c.py"))
        for p in (edited, a, b, c):
            p.write_text(f"# {p.name}\n", encoding="utf-8")
        msgs = [_msg("assistant", "", tool_calls=[_call("read_file", file_path=str(a))]),
                _msg("assistant", "", tool_calls=[_call("read_file", file_path=str(b))]),
                _msg("assistant", "", tool_calls=[_call("read_file", file_path=str(c))]),
                _msg("assistant", "", tool_calls=[_call("write_file", file_path=str(edited))])]
        out = trimmer._rehydrate_files(msgs, max_files=3)
        assert "edited.py" in out and "c.py" in out and "b.py" in out and "a.py" not in out.replace("edited.py", "")
