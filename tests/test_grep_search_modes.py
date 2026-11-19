"""grep_search modes: an exact word, exact letter case, plain text -- through ripgrep and through the Python fallback alike."""
import re
import shutil
import subprocess

import pytest

from src.agent.arg_coercion import coerce_and_log
from src.agent.parser import normalize_tool_params
from argent.tools import search_ops
from argent.tools.schemas import TOOL_SCHEMAS
from argent.tools.search_ops import grep_search

TEXT = "rea\nread the file\nreason\narea\nREA upper\nRea title\nкорея\nрея тут\ncall read(x)\na.b here\naxb there\nfoo\\|bar\n-v flag\n"


def found(out: str):
    return [m.group(1).strip() for m in re.finditer(r"^\s+\d+: (.*)$", out, re.M)]


@pytest.fixture(params=["ripgrep", "python"])
def search(request, tmp_path, monkeypatch):
    if request.param == "ripgrep" and not shutil.which("rg"):
        pytest.skip("ripgrep is not installed")
    if request.param == "python":                     # no rg on the machine: the fallback must give the same answers
        real = subprocess.run

        def no_rg(cmd, *a, **k):
            if cmd and cmd[0] == "rg":
                raise FileNotFoundError("rg")
            return real(cmd, *a, **k)

        monkeypatch.setattr(subprocess, "run", no_rg)
    (tmp_path / "a.txt").write_text(TEXT, encoding="utf-8")
    return lambda pattern, **kw: grep_search(path=str(tmp_path), pattern=pattern, **kw)


def test_default_is_a_case_insensitive_substring_regex(search):
    assert found(search("rea")) == ["rea", "read the file", "reason", "area", "REA upper", "Rea title", "call read(x)"]


def test_whole_word(search):
    out = search("rea", whole_word=True)
    assert found(out) == ["rea", "REA upper", "Rea title"] and "[whole word]" in out
    assert found(search("рея", whole_word=True)) == ["рея тут"]          # Cyrillic word boundaries too


def test_case_sensitive(search):
    assert found(search("Rea", case_sensitive=True)) == ["Rea title"]
    assert found(search("rea", whole_word=True, case_sensitive=True)) == ["rea"]


def test_literal_text_with_regex_characters(search):
    assert found(search("read(", literal=True)) == ["call read(x)"]
    assert found(search("a.b", literal=True)) == ["a.b here"]                # not "axb"
    assert found(search(r"foo\|bar", literal=True)) == [r"foo\|bar"]       # no BRE-alternation rewrite in literal mode


def test_an_invalid_regex_points_to_literal(search):
    out = search("read(")
    assert "Invalid regex" in out and "literal=true" in out


def test_a_pattern_starting_with_a_dash_is_a_pattern(search):
    assert found(search("-v", literal=True)) == ["-v flag"]


def test_flags_arrive_as_text_from_a_model(tmp_path):
    (tmp_path / "a.txt").write_text(TEXT, encoding="utf-8")
    args = coerce_and_log("grep_search", {"directory": str(tmp_path), "pattern": "rea", "whole_word": "true"})
    assert args["whole_word"] is True
    assert found(grep_search(**args)) == ["rea", "REA upper", "Rea title"]


def test_grep_flag_names_are_aliases():
    res = normalize_tool_params({"name": "grep_search", "arguments": {"pattern": "x", "word_regexp": True, "fixed_strings": True, "match_case": True}})
    assert res["arguments"] == {"pattern": "x", "whole_word": True, "literal": True, "case_sensitive": True}


def test_the_model_is_told_about_the_modes():
    fn = next(t["function"] for t in TOOL_SCHEMAS if t["function"]["name"] == "grep_search")
    props = fn["parameters"]["properties"]
    assert all(props[k]["type"] == "boolean" for k in ("whole_word", "case_sensitive", "literal"))
    assert "whole_word=true" in fn["description"] and "literal=true" in fn["description"]


def test_search_ops_fallback_regex_matches_rg_word_semantics():
    rx = search_ops._search_regex("(x)", whole_word=True, literal=True)        # starts and ends with non-word characters
    assert rx.search("call (x) now") and not rx.search("a(x)b")
