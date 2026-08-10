"""A screenshot is uploaded once, not on every turn for the rest of the session.

An attached image is ~190 KB of base64 and it lived in the history forever, so
every later request re-sent every picture ever taken. It was the one unbounded
thing in a conversation that is otherwise trimmed: tool results get stubbed,
turns get dropped, images accumulated.

The model does not need the pixels twice. By the time the next turn starts it
has already written what it saw, and that text is what the history is for.
"""

import pytest

from src.agent.trimmer import (
    KEEP_RECENT_IMAGES, clean_messages_for_llm, compact_image_history,
    estimate_message_tokens,
)


def _img(label, n=1, payload="A" * 200_000):
    return {"role": "user", "content": f"Вот запрошенное изображение: {label}",
            "images": [{"data": payload, "media_type": "image/png"}] * n}


def _chat(text="ok"):
    return {"role": "assistant", "content": text}


class TestOldPayloadsGo:
    def test_only_the_recent_ones_keep_their_pixels(self):
        history = []
        for i in range(5):
            history += [_img(f"shot{i}"), _chat(f"вижу {i}")]
        out = compact_image_history(history)
        carrying = [m for m in out if m.get("images")]
        assert len(carrying) == KEEP_RECENT_IMAGES
        assert carrying[-1]["content"].endswith("shot4")

    def test_the_caption_survives(self):
        """"the screenshot you took" has to still refer to something."""
        out = compact_image_history([_img("страница логина"), _img("a"), _img("b")])
        assert "страница логина" in out[0]["content"]
        assert "показано ранее" in out[0]["content"]
        assert "images" not in out[0]

    def test_the_count_is_named(self):
        out = compact_image_history([_img("pair", n=3), _img("a"), _img("b")])
        assert "3 изобр." in out[0]["content"]

    def test_nothing_to_do_returns_the_same_object(self):
        history = [_chat(), _img("only")]
        assert compact_image_history(history) is history

    def test_messages_without_images_are_untouched(self):
        history = [_chat("one"), _img("a"), _img("b"), _img("c"), _chat("two")]
        out = compact_image_history(history)
        assert out[0] == {"role": "assistant", "content": "one"}
        assert out[-1] == {"role": "assistant", "content": "two"}


class TestItHappensOnEveryRequest:
    def test_clean_messages_strips_them(self):
        """Not only when the history needs trimming: re-uploading a picture
        from twenty turns ago is waste at any context size."""
        history = [{"role": "system", "content": "s"}]
        for i in range(4):
            history += [_img(f"s{i}"), _chat()]
        cleaned = clean_messages_for_llm(history, strip_reasoning=False)
        assert sum(1 for m in cleaned if m.get("images")) == KEEP_RECENT_IMAGES

    def test_the_meter_matches_what_is_sent(self):
        """The estimate is computed from this same copy, so a stripped image
        must stop being counted as an image."""
        stripped = compact_image_history([_img("a"), _img("b"), _img("c")])[0]
        assert estimate_message_tokens(stripped, "gpt-4", "zai") < 200

    def test_the_original_history_is_not_mutated(self):
        history = [_img("a"), _img("b"), _img("c")]
        compact_image_history(history)
        assert all(m.get("images") for m in history)


class TestTheRecentOnesStillWork:
    def test_the_newest_image_is_sent_whole(self):
        out = compact_image_history([_img("old"), _img("new")])
        assert out[-1]["images"][0]["data"].startswith("A")

    def test_a_before_and_after_pair_both_survive(self):
        """Two is the working pattern: act, screenshot, compare."""
        out = compact_image_history([_img("baseline"), _img("before"), _img("after")])
        kept = [m["content"] for m in out if m.get("images")]
        assert len(kept) == 2 and "before" in kept[0] and "after" in kept[1]
