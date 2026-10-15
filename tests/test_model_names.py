"""Model names as typed vs as the API spells them (model_names.py)."""

import pytest

from claude_catalog import ModelEntry, sort_newest_first
from model_names import (
    canonical_claude, fold, match_known, parse_claude_name, resolve_claude, suggest,
)


def E(mid, created=0):
    return ModelEntry(id=mid, created=created)


CATALOG = sort_newest_first([
    E("claude-fable-5-1-20260401", 310), E("claude-opus-5-5-20260301", 300),
    E("claude-sonnet-5-5", 291), E("claude-sonnet-5-5-20260215", 290),
    E("claude-haiku-4-5-20251001", 210), E("claude-sonnet-4-5-20250929", 200),
    E("claude-3-7-sonnet-20250219", 110), E("claude-3-5-sonnet-20241022", 100),
    E("claude-3-haiku-20240307", 50),
])


class TestParse:
    @pytest.mark.parametrize("text, family, version, date", [
        ("claude-sonnet-4-5", "sonnet", (4, 5), None),
        ("Claude Sonnet 4.5", "sonnet", (4, 5), None),
        ("claude_sonnet_4_5", "sonnet", (4, 5), None),
        ("anthropic/claude-sonnet-4.5", "sonnet", (4, 5), None),
        ("us.anthropic.claude-sonnet-4-5-20250929-v1:0", "sonnet", (4, 5), "20250929"),
        ("claude-3.5-sonnet", "sonnet", (3, 5), None),
        ("claude-3-5-sonnet-latest", "sonnet", (3, 5), None),
        ("claude-3-haiku-20240307", "haiku", (3,), "20240307"),
        ("anthropic/claude-3.7-sonnet:thinking", "sonnet", (3, 7), None),
        ("OPUS", "opus", (), None),
    ])
    def test_family_version_and_date_from_any_spelling(self, text, family, version, date):
        p = parse_claude_name(text)
        assert (p.family, p.version, p.date) == (family, version, date)

    @pytest.mark.parametrize("text", ["gpt-4o", "4.5", "", "llama3.1", "deepseek-chat", "claude", "claude-banana-4"])
    def test_other_names_are_not_claude_names(self, text):
        assert parse_claude_name(text) is None

    def test_trailing_zeros_do_not_make_a_different_model(self):
        assert parse_claude_name("claude-sonnet-4-0").key == parse_claude_name("claude sonnet 4").key


class TestCanonical:
    @pytest.mark.parametrize("typed, api_spelling", [
        ("Claude Sonnet 4.5", "claude-sonnet-4-5"),
        ("claude-sonnet-4.5", "claude-sonnet-4-5"),
        ("anthropic/claude-opus-5.5", "claude-opus-5-5"),
        ("claude-haiku-4.5-20251001", "claude-haiku-4-5-20251001"),
        ("claude-3.5-sonnet", "claude-3-5-sonnet-latest"),
        ("Claude 3.7 Sonnet", "claude-3-7-sonnet-latest"),
        ("claude-3-haiku-20240307", "claude-3-haiku-20240307"),
    ])
    def test_no_dots_no_spaces_the_family_where_the_api_wants_it(self, typed, api_spelling):
        assert canonical_claude(typed) == api_spelling

    def test_a_bare_family_or_a_stranger_has_no_canonical_id(self):
        assert canonical_claude("opus") is None and canonical_claude("gpt-4o") is None


class TestResolve:
    @pytest.mark.parametrize("typed, listed", [
        ("Claude Sonnet 4.5", "claude-sonnet-4-5-20250929"),
        ("claude-sonnet-4.5", "claude-sonnet-4-5-20250929"),
        ("claude_sonnet_4_5", "claude-sonnet-4-5-20250929"),
        ("anthropic/claude-sonnet-4.5", "claude-sonnet-4-5-20250929"),
        ("sonnet 4.5", "claude-sonnet-4-5-20250929"),
        ("claude-sonnet-4-5-20250929", "claude-sonnet-4-5-20250929"),
        ("claude-3.5-sonnet", "claude-3-5-sonnet-20241022"),
        ("claude-3-5-sonnet-latest", "claude-3-5-sonnet-20241022"),
        ("Claude 3.7 Sonnet", "claude-3-7-sonnet-20250219"),
        ("claude-haiku-4.5", "claude-haiku-4-5-20251001"),
        ("claude-3-haiku", "claude-3-haiku-20240307"),
        ("anthropic/claude-3.5-sonnet:beta", "claude-3-5-sonnet-20241022"),
    ])
    def test_any_spelling_finds_the_listed_id(self, typed, listed):
        assert resolve_claude(typed, CATALOG) == listed

    def test_a_bare_family_means_the_newest_of_it(self):
        assert resolve_claude("opus", CATALOG) == "claude-opus-5-5-20260301"
        assert resolve_claude("haiku", CATALOG) == "claude-haiku-4-5-20251001"
        assert resolve_claude("fable", CATALOG) == "claude-fable-5-1-20260401"

    def test_the_undated_alias_wins_over_its_snapshots_when_the_catalogue_lists_it(self):
        assert resolve_claude("sonnet", CATALOG) == "claude-sonnet-5-5"
        assert resolve_claude("Claude Sonnet 5.5", CATALOG) == "claude-sonnet-5-5"
        assert resolve_claude("claude-sonnet-5-5-20260215", CATALOG) == "claude-sonnet-5-5-20260215"

    def test_what_is_not_listed_is_not_invented(self):
        for text in ("claude-opus-9", "claude-sonnet-4.5-20990101", "gpt-4o", "4.5", "", "claude-haiku-3-5"):
            assert resolve_claude(text, CATALOG) is None, text
        assert resolve_claude("sonnet", []) is None

    def test_a_family_the_catalogue_introduces_is_understood(self):
        catalog = [E("claude-quill-6-0-20270101", 1)]
        assert resolve_claude("quill", catalog) == "claude-quill-6-0-20270101"
        assert resolve_claude("Claude Quill 6", catalog) == "claude-quill-6-0-20270101"


class TestAggregators:
    IDS = ["anthropic/claude-sonnet-4.5", "anthropic/claude-3.5-sonnet", "openai/gpt-4o",
           "meta-llama/llama-3.3-70b-instruct:free", "openai/gpt-4o-mini"]

    @pytest.mark.parametrize("typed, listed", [
        ("anthropic/claude-sonnet-4.5", "anthropic/claude-sonnet-4.5"),
        ("Anthropic/Claude-Sonnet-4.5", "anthropic/claude-sonnet-4.5"),
        ("anthropic/claude sonnet 4.5", "anthropic/claude-sonnet-4.5"),
        ("claude-sonnet-4-5", "anthropic/claude-sonnet-4.5"),          # vendor prefix left out
        ("gpt 4o", "openai/gpt-4o"),
        ("llama-3.3-70b-instruct:free", "meta-llama/llama-3.3-70b-instruct:free"),
    ])
    def test_the_listed_spelling_is_returned_as_listed(self, typed, listed):
        assert match_known(typed, self.IDS) == listed

    def test_ambiguity_is_not_resolved_by_guessing(self):
        assert match_known("gpt", self.IDS) is None
        assert match_known("nonsense", self.IDS) is None
        assert match_known("", self.IDS) is None


def test_fold_and_suggest():
    assert fold("  Claude_Sonnet 4.5/x ") == "claude-sonnet-4-5-x"
    ids = [e.id for e in CATALOG]
    assert "claude-sonnet-4-5-20250929" in suggest("claude-sonet-4-5", ids)
