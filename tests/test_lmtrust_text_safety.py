"""Blind-spot tests for text_safety.py.

Covers edge cases that are easy to get wrong: empty strings, reversed
surrogate pairs, all-surrogate strings, tuples passing through repair_structure
unchanged, empty containers, deeply nested structures, and the UTF-8
encodability invariant.
"""

import pytest

from text_safety import (
    _HIGH,
    _LOW,
    REPLACEMENT,
    has_lone_surrogates,
    repair_surrogates,
    repair_structure,
)


# ---------------------------------------------------------------------------
# 1. TestEmptyString (L2×D1)
# ---------------------------------------------------------------------------
class TestEmptyString:
    """The empty string has no surrogates and repairing it must stay empty."""

    def test_has_lone_surrogates_empty_is_false(self):
        assert has_lone_surrogates("") is False

    def test_repair_surrogates_empty_returns_empty(self):
        result = repair_surrogates("")
        assert result == ""
        # The result must be UTF-8 encodable (trivially true for "" but explicit).
        result.encode("utf-8")


# ---------------------------------------------------------------------------
# 2. TestReversedPair (L4×D1)
# ---------------------------------------------------------------------------
class TestReversedPair:
    """A LOW surrogate followed by a HIGH surrogate is NOT a valid pair.

    The repair logic only recombines HIGH-then-LOW. A reversed pair (LOW-HIGH)
    must have both halves replaced with REPLACEMENT, not recombined into a
    bogus codepoint.
    """

    def test_reversed_pair_both_replaced(self):
        text = chr(0xDC00) + chr(0xD800)  # LOW then HIGH
        result = repair_surrogates(text)
        # Neither half should have been recombined into a supplementary char.
        assert result == REPLACEMENT + REPLACEMENT

    def test_reversed_pair_is_encodable(self):
        text = chr(0xDC00) + chr(0xD800)
        result = repair_surrogates(text)
        result.encode("utf-8")  # raises if surrogates remain

    def test_reversed_pair_no_supplementary_codepoint(self):
        text = chr(0xDC00) + chr(0xD800)
        result = repair_surrogates(text)
        # No character in the result should be outside the BMP-surrogate-free
        # range that would indicate a bogus recombination.
        for ch in result:
            assert not (0x10000 <= ord(ch)), "reversed pair was recombined"

    def test_reversed_pair_length(self):
        text = chr(0xDC00) + chr(0xD800)
        result = repair_surrogates(text)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# 3. TestAllSurrogates (L2×D1)
# ---------------------------------------------------------------------------
class TestAllSurrogates:
    """A string composed entirely of high surrogates (D800–DBFF).

    No valid pairs exist (every char is HIGH with no LOW following), so every
    character must become REPLACEMENT.
    """

    def test_all_high_surrogates_each_replaced(self):
        text = "".join(chr(c) for c in range(0xD800, 0xDBFF + 1))
        result = repair_surrogates(text)
        expected = REPLACEMENT * (0xDBFF + 1 - 0xD800)
        assert result == expected

    def test_all_high_surrogates_encodable(self):
        text = "".join(chr(c) for c in range(0xD800, 0xDBFF + 1))
        result = repair_surrogates(text)
        result.encode("utf-8")


# ---------------------------------------------------------------------------
# 4. TestRepairStructureTuple (L4×D7)
# ---------------------------------------------------------------------------
class TestRepairStructureTuple:
    """repair_structure only recurses into str/dict/list.

    A tuple — even one containing a surrogate-laden string — is none of those,
    so it must be returned as the exact same object (identity check).
    """

    def test_tuple_passthrough_identity(self):
        value = ("hello", chr(0xD800) + " world")
        result = repair_structure(value)
        assert result is value

    def test_tuple_with_surrogate_string_unchanged(self):
        bad_str = chr(0xD800) + "x"
        value = (bad_str,)
        result = repair_structure(value)
        assert result is value
        # The surrogate string inside is NOT repaired because tuples are opaque.
        assert result[0] is bad_str

    def test_tuple_nested_dict_inside_still_opaque_tuple(self):
        # The tuple itself is returned as-is; its contents are not traversed.
        value = ({"key": chr(0xD800)},)
        result = repair_structure(value)
        assert result is value

    def test_empty_tuple_identity(self):
        value = ()
        result = repair_structure(value)
        assert result is value

    def test_tuple_of_mixed_types_identity(self):
        value = (1, 2.0, True, None, chr(0xDC00))
        result = repair_structure(value)
        assert result is value

    def test_single_element_tuple_identity(self):
        value = (chr(0xD800),)
        result = repair_structure(value)
        assert result is value

    def test_tuple_in_dict_value_dict_unchanged(self):
        # The tuple value is opaque to repair_structure, so the dict's value is
        # unchanged (identity) and the dict itself is returned as-is.
        value = {"items": (chr(0xD800),)}
        result = repair_structure(value)
        assert result is value
        # The tuple inside is still the same object with the surrogate intact.
        assert result["items"] is value["items"]

    def test_tuple_does_not_raise_on_encode_of_structure(self):
        # repair_structure itself must never raise; encoding is caller's job.
        value = (chr(0xD800) + chr(0xDC00),)
        result = repair_structure(value)
        assert result is value  # still opaque


# ---------------------------------------------------------------------------
# 5. TestRepairStructureEmpty (L2×D3)
# ---------------------------------------------------------------------------
class TestRepairStructureEmpty:
    """Empty dict and empty list have nothing to repair → same object returned."""

    def test_empty_dict_identity(self):
        value = {}
        result = repair_structure(value)
        assert result is value

    def test_empty_list_identity(self):
        value = []
        result = repair_structure(value)
        assert result is value


# ---------------------------------------------------------------------------
# 6. TestRepairStructureDeeplyNested (L2×D9)
# ---------------------------------------------------------------------------
class TestRepairStructureDeeplyNested:
    """A 10-level nested dict with a surrogate at the bottom.

    The repair must recurse all the way down and fix the surrogate, producing
    a fully UTF-8-encodable leaf string.
    """

    def _build_nested(self, depth: int, leaf):
        """Build a dict nested *depth* levels deep with *leaf* at the bottom."""
        current = leaf
        for _ in range(depth):
            current = {"next": current}
        return current

    def test_deeply_nested_surrogate_repaired(self):
        bad_leaf = "before" + chr(0xD800) + "after"
        value = self._build_nested(10, bad_leaf)
        result = repair_structure(value)

        # Walk down 10 levels.
        node = result
        for _ in range(10):
            assert isinstance(node, dict)
            assert "next" in node
            node = node["next"]

        # The leaf should no longer contain a lone surrogate.
        assert has_lone_surrogates(node) is False
        assert node == "before" + REPLACEMENT + "after"

    def test_deeply_nested_result_encodable(self):
        bad_leaf = chr(0xD800) + chr(0xDC00) + chr(0xD800)
        value = self._build_nested(10, bad_leaf)
        result = repair_structure(value)

        node = result
        for _ in range(10):
            node = node["next"]

        # Must be encodable without raising.
        node.encode("utf-8")


# ---------------------------------------------------------------------------
# 7. TestRepairSurrogatesInvariant (L3×D10)
# ---------------------------------------------------------------------------
class TestRepairSurrogatesInvariant:
    """Property: for any input string, repair_surrogates output is UTF-8 encodable.

    Tested with several adversarial edge cases that probe the pair-recombination
    logic from different angles.
    """

    @staticmethod
    def _assert_encodable(text: str):
        """The invariant: result must encode as UTF-8 without error."""
        result = repair_surrogates(text)
        result.encode("utf-8")  # raises UnicodeEncodeError if surrogates remain
        return result

    def test_high_at_end(self):
        # A lone HIGH surrogate at the very end with no following char.
        text = "abc" + chr(0xD800)
        result = self._assert_encodable(text)
        assert result == "abc" + REPLACEMENT

    def test_low_at_start(self):
        # A lone LOW surrogate at the start with no preceding HIGH.
        text = chr(0xDC00) + "xyz"
        result = self._assert_encodable(text)
        assert result == REPLACEMENT + "xyz"

    def test_high_high_low(self):
        # HIGH + HIGH + LOW: the first HIGH is orphaned (next is HIGH, not LOW),
        # the second HIGH pairs with the LOW → one REPLACEMENT + one emoji.
        text = chr(0xD800) + chr(0xD800) + chr(0xDC00)
        result = self._assert_encodable(text)
        # First HIGH → REPLACEMENT; second HIGH + LOW → recombined supplementary.
        assert result[0] == REPLACEMENT
        assert ord(result[1]) >= 0x10000

    def test_multiple_pairs_mixed_with_orphans(self):
        # Two valid pairs with real orphans between them: an orphan HIGH (followed
        # by a non-surrogate) and an orphan LOW (at a position with no preceding HIGH).
        pair1 = chr(0xD83D) + chr(0xDE00)   # 😀
        pair2 = chr(0xD83D) + chr(0xDE0A)   # 😊
        orphan_high = chr(0xD800) + "X"     # HIGH followed by non-surrogate → orphan
        orphan_low = chr(0xDC00)            # LOW with no preceding HIGH → orphan
        text = pair1 + orphan_high + orphan_low + pair2
        result = self._assert_encodable(text)
        # pair1 (1 char) + REPLACEMENT + "X" + REPLACEMENT + pair2 (1 char) = 5 chars.
        assert result == "😀" + REPLACEMENT + "X" + REPLACEMENT + "😊"
        assert len(result) == 5

    def test_only_low_surrogates(self):
        text = chr(0xDC00) + chr(0xDC01) + chr(0xDFFF)
        result = self._assert_encodable(text)
        assert result == REPLACEMENT * 3

    def test_alternating_high_low_valid_pairs(self):
        # Every HIGH is immediately followed by a LOW → all recombined.
        text = (chr(0xD800) + chr(0xDC00)) * 5
        result = self._assert_encodable(text)
        assert len(result) == 5
        for ch in result:
            assert ord(ch) >= 0x10000

    def test_high_followed_by_non_surrogate(self):
        # HIGH followed by a regular char → HIGH is orphaned.
        text = chr(0xD800) + "A"
        result = self._assert_encodable(text)
        assert result == REPLACEMENT + "A"

    def test_low_followed_by_high(self):
        # Already covered in TestReversedPair but also as an invariant check.
        text = chr(0xDC00) + chr(0xD800)
        result = self._assert_encodable(text)
        assert result == REPLACEMENT + REPLACEMENT

    def test_empty_string_invariant(self):
        result = self._assert_encodable("")
        assert result == ""

    def test_clean_string_unchanged(self):
        text = "Hello, world! 🌍"
        result = repair_surrogates(text)
        assert result == text
        result.encode("utf-8")