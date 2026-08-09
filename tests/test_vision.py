"""The model can actually look at an image.

browser_screenshot wrote PNGs that nothing could read: no provider sent
pictures back, so "take a screenshot to see the page" was a wasted turn. Two
things had to be got right, and both are invisible until they break — whether
the model can see at all, and how big the picture is allowed to be.

Verified against a live vision model before these tests were written: given a
generated image, minimax-m3 named the red rectangle, the blue ellipse and read
"ARGENT VISION TEST 7431" — a number it could not have guessed.
"""

import base64
import json

import pytest

import vision
from providers import _images_for_ollama, _images_for_openai
from vision import MAX_EDGE_PIXELS, build_image_message, encode_image, model_supports_vision


@pytest.fixture(autouse=True)
def clean_state():
    vision._CAPABILITY_CACHE.clear()
    vision.drain_images()
    yield
    vision.drain_images()


def _png(tmp_path, size=(40, 30), name="x.png"):
    from PIL import Image
    p = tmp_path / name
    Image.new("RGB", size, (200, 30, 30)).save(p)
    return p


class _Response:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class TestCapability:
    def test_a_vision_model_is_recognised(self, monkeypatch):
        import requests
        monkeypatch.setattr(requests, "post", lambda *a, **k: _Response(
            {"capabilities": ["completion", "vision", "tools"]}))
        assert model_supports_vision("qwen3.5:9b", "ollama") is True

    def test_a_text_only_model_is_recognised(self, monkeypatch):
        """glm-5.2 reports tools and thinking but no vision — sending it an
        image would get a confident answer about a picture it never saw."""
        import requests
        monkeypatch.setattr(requests, "post", lambda *a, **k: _Response(
            {"capabilities": ["thinking", "completion", "tools"]}))
        assert model_supports_vision("glm-5.2:cloud", "ollama") is False

    def test_unknown_is_not_the_same_as_no(self, monkeypatch):
        """Refusing because we could not ask is worse than trying; claiming
        support we never verified is how the confident-hallucination happens."""
        import requests

        def boom(*a, **k):
            raise OSError("connection refused")

        monkeypatch.setattr(requests, "post", boom)
        assert model_supports_vision("m", "ollama") is None

    def test_a_non_ollama_provider_is_unknown(self, monkeypatch):
        import requests
        monkeypatch.setattr(requests, "post",
                            lambda *a, **k: pytest.fail("must not ask Ollama"))
        assert model_supports_vision("gpt-x", "openrouter") is None

    def test_the_answer_is_cached(self, monkeypatch):
        """One /api/show per model, not one per image: the call is on the path
        of a tool the model may use repeatedly."""
        calls = []
        import requests
        monkeypatch.setattr(requests, "post", lambda *a, **k: calls.append(1) or _Response(
            {"capabilities": ["vision"]}))
        model_supports_vision("m", "ollama")
        model_supports_vision("m", "ollama")
        assert len(calls) == 1


class TestEncoding:
    def test_a_small_image_is_encoded_untouched(self, tmp_path):
        data, media, note = encode_image(_png(tmp_path))
        assert base64.b64decode(data) and media == "image/png" and note == ""

    def test_a_large_image_is_downscaled(self, tmp_path):
        """Resolution past what the model attends to costs context linearly and
        buys nothing."""
        big = _png(tmp_path, size=(MAX_EDGE_PIXELS * 2, MAX_EDGE_PIXELS))
        data, media, note = encode_image(big)
        assert "уменьшено" in note and media == "image/jpeg"

        from PIL import Image
        import io
        with Image.open(io.BytesIO(base64.b64decode(data))) as img:
            assert max(img.size) <= MAX_EDGE_PIXELS

    def test_a_missing_file(self, tmp_path):
        data, _, note = encode_image(tmp_path / "nope.png")
        assert data is None and "не найден" in note

    def test_a_file_that_is_not_an_image(self, tmp_path):
        p = tmp_path / "notes.md"
        p.write_text("# заметки", encoding="utf-8")
        data, _, note = encode_image(p)
        assert data is None and "не изображение" in note


class TestProviderDialects:
    """History is stored provider-neutrally because a conversation outlives the
    provider that started it — writing one vendor's spelling into the history
    breaks the moment the user switches provider mid-chat."""

    def _msg(self):
        return build_image_message([("BASE64DATA", "image/png")], "смотри")

    def test_ollama_gets_a_bare_base64_list(self):
        out = _images_for_ollama([self._msg()])
        assert out[0]["images"] == ["BASE64DATA"]
        assert out[0]["content"] == "смотри"

    def test_openai_gets_content_parts_with_a_data_uri(self):
        out = _images_for_openai([self._msg()])
        parts = out[0]["content"]
        assert parts[0] == {"type": "text", "text": "смотри"}
        assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,BASE64DATA")
        assert "images" not in out[0]

    def test_messages_without_images_are_returned_unchanged(self):
        """Identity, not a copy: every request would otherwise rebuild the whole
        history to change nothing."""
        messages = [{"role": "user", "content": "просто текст"}]
        assert _images_for_ollama(messages) is messages
        assert _images_for_openai(messages) is messages

    def test_the_stored_history_is_not_mutated(self):
        msg = self._msg()
        _images_for_ollama([msg])
        _images_for_openai([msg])
        assert isinstance(msg["images"][0], dict) and msg["content"] == "смотри"


class TestTheTool:
    def test_a_queued_image_is_handed_over_once(self, tmp_path, monkeypatch):
        from tools.misc_tools import view_image

        monkeypatch.setattr("config.get_current_model", lambda: "seer")
        monkeypatch.setattr("config.get_provider", lambda: "ollama")
        monkeypatch.setattr(vision, "model_supports_vision", lambda m, p: True)

        out = view_image(str(_png(tmp_path)))
        assert "next message" in out          # the model must not claim to have seen it yet
        assert len(vision.drain_images()) == 1
        assert vision.drain_images() == []    # drained, not duplicated into the next turn

    def test_a_blind_model_is_refused_before_encoding(self, tmp_path, monkeypatch):
        from tools.misc_tools import view_image

        monkeypatch.setattr("config.get_current_model", lambda: "glm-5.2:cloud")
        monkeypatch.setattr("config.get_provider", lambda: "ollama")
        monkeypatch.setattr(vision, "model_supports_vision", lambda m, p: False)
        monkeypatch.setattr(vision, "encode_image",
                            lambda p: pytest.fail("не нужно кодировать для слепой модели"))

        out = view_image(str(_png(tmp_path)))
        assert out.startswith("Error:") and "/model" in out

    def test_unverified_support_is_tried_but_flagged(self, tmp_path, monkeypatch):
        from tools.misc_tools import view_image

        monkeypatch.setattr("config.get_current_model", lambda: "m")
        monkeypatch.setattr("config.get_provider", lambda: "openrouter")
        monkeypatch.setattr(vision, "model_supports_vision", lambda m, p: None)

        out = view_image(str(_png(tmp_path)))
        assert "could not be verified" in out
        assert len(vision.drain_images()) == 1

    def test_nothing_is_queued_when_the_file_is_bad(self, tmp_path, monkeypatch):
        from tools.misc_tools import view_image

        monkeypatch.setattr("config.get_current_model", lambda: "seer")
        monkeypatch.setattr("config.get_provider", lambda: "ollama")
        monkeypatch.setattr(vision, "model_supports_vision", lambda m, p: True)

        assert view_image(str(tmp_path / "nope.png")).startswith("Error:")
        assert vision.drain_images() == []
