"""`count` counts non-empty LINES; `count:json:<path>` and `count:regex_all:<pattern>` count ITEMS."""

import pytest

from src.rooms.models import is_derived_spec
from src.rooms.runner import _empty_value, derive_value

BODY = '{"posts": ["a", "b", "c"], "other": {"x": 1}}'
VALUES = {"body": BODY, "output": BODY}


def test_plain_count_is_still_the_number_of_lines():
    assert derive_value("count", VALUES) == 1                 # one line, however many items it holds
    assert derive_value("count", {"body": "a\n\nb\nc\n"}) == 3


def test_count_json_is_the_length_of_the_list_at_the_path():
    assert derive_value("count:json:posts", VALUES) == 3
    assert derive_value("count:json:other", VALUES) == 1      # an object: its keys
    assert derive_value("count:json:missing", VALUES) == 0


def test_count_regex_all_counts_matches():
    assert derive_value(r"count:regex_all:id=(\d+)", {"body": "id=1 id=22 id=333"}) == 3
    assert derive_value(r"count:regex_all:id=(\d+)", {"body": "nothing"}) == 0


def test_a_step_that_did_not_run_counts_zero():
    assert _empty_value("count:json:posts") == 0


@pytest.mark.parametrize("spec,ok", [
    ("count", True), ("count:json:posts", True), ("count:regex_all:a+", True),
    ("count:json:", False), ("count:regex_all:(", False), ("count:lines", False), ("count:", False),
])
def test_the_validator_knows_the_new_forms(spec, ok):
    assert is_derived_spec(spec) is ok
