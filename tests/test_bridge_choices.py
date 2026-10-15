"""Choice normalisation for the TUI's native select modal."""
import questionary

from src.bridge.repl_bridge import _prepare_choices


def test_plain_strings_are_kept_verbatim():
    """Regression: `str` has a METHOD named `title`, so probing getattr(c, 'title')
    turned every plain-string option into "<built-in method title of str ...>"."""
    titles, mapping, checked = _prepare_choices(["glm", "gpt", "claude"])
    assert titles == ["glm", "gpt", "claude"]
    assert mapping == {"glm": "glm", "gpt": "gpt", "claude": "claude"}
    assert checked is None
    assert not any("built-in method" in t for t in titles)


def test_dict_and_object_choices_map_title_to_value():
    class Obj:
        title = "Pretty"
        value = 42
        disabled = False
        checked = True

    titles, mapping, checked = _prepare_choices(
        [{"title": "A", "value": 1}, Obj(), "plain"])
    assert titles == ["A", "Pretty", "plain"]
    assert mapping == {"A": 1, "Pretty": 42, "plain": "plain"}
    assert checked == [False, True, False]


def test_questionary_choice_objects_survive():
    titles, mapping, _ = _prepare_choices(
        [questionary.Choice("Show", value="show"), questionary.Choice("Hide", value="hide")])
    assert titles == ["Show", "Hide"]
    assert mapping["Show"] == "show"


def test_disabled_and_separators_are_dropped():
    titles, _, _ = _prepare_choices(
        [{"title": "off", "disabled": True}, questionary.Separator(), "keep"])
    assert titles == ["keep"]


def test_empty_input():
    assert _prepare_choices(None) == ([], {}, None)
    assert _prepare_choices([]) == ([], {}, None)
