"""An attached image costs what an image costs, not what its base64 measures.

Everything measured `str(message)`, which for an attachment counts every
base64 character. Measured on a real 1568px screenshot — 192 KB of base64 —
the estimate was 139,242 tokens against a true cost near 1,400: off by ninety
times. The context meter read past 300k after a single request.

The meter was the visible half. The same estimate drives trimming, so one
screenshot would have evicted the entire conversation to make room for a cost
that never existed.
"""

import pytest

from src.agent.trimmer import IMAGE_TOKEN_COST, estimate_message_tokens, estimate_tokens


def _fake_base64(size=200_000):
    """High-entropy like real base64. A run of "A" compresses to almost nothing
    under a tokenizer, which would understate the very problem being tested."""
    import base64
    import random

    rng = random.Random(1)
    return base64.b64encode(bytes(rng.randrange(256) for _ in range(size * 3 // 4))).decode()


def _image_message(payloads=1, text="Вот изображение"):
    return {
        "role": "user",
        "content": text,
        "images": [{"data": _fake_base64(), "media_type": "image/jpeg"}] * payloads,
    }


class TestImagesAreCountedAsImages:
    def test_the_base64_is_not_measured(self):
        message = _image_message()
        naive = estimate_tokens(str(message), "m", "ollama")
        actual = estimate_message_tokens(message, "m", "ollama")
        assert naive > 40_000                      # what it used to report
        assert actual < 2_000                      # what an image actually costs

    def test_each_image_adds_a_constant(self):
        one = estimate_message_tokens(_image_message(1), "m", "ollama")
        two = estimate_message_tokens(_image_message(2), "m", "ollama")
        assert two - one == IMAGE_TOKEN_COST

    def test_the_text_beside_the_image_still_counts(self):
        short = estimate_message_tokens(_image_message(text="да"), "m", "ollama")
        long = estimate_message_tokens(
            _image_message(text="разбор " * 500), "m", "ollama")
        assert long > short + 200

    def test_the_payload_size_does_not_change_the_answer(self):
        """A 2 MB screenshot and a 20 KB icon cost a vision model the same."""
        small = {"role": "user", "content": "x",
                 "images": [{"data": _fake_base64(20_000), "media_type": "image/png"}]}
        big = {"role": "user", "content": "x",
               "images": [{"data": _fake_base64(2_000_000), "media_type": "image/png"}]}
        assert (estimate_message_tokens(small, "m", "ollama")
                == estimate_message_tokens(big, "m", "ollama"))


class TestEverythingElseIsUnchanged:
    def test_a_plain_message(self):
        message = {"role": "user", "content": "обычное сообщение"}
        assert (estimate_message_tokens(message, "m", "ollama")
                == estimate_tokens(str(message), "m", "ollama"))

    def test_an_empty_images_list_is_not_an_attachment(self):
        message = {"role": "user", "content": "текст", "images": []}
        assert (estimate_message_tokens(message, "m", "ollama")
                == estimate_tokens(str(message), "m", "ollama"))

    def test_a_non_dict_does_not_raise(self):
        assert estimate_message_tokens("просто строка", "m", "ollama") > 0
        assert estimate_message_tokens(None, "m", "ollama") >= 0


class TestTheCallersUseIt:
    def test_the_context_meter_does_not_explode(self, monkeypatch):
        """The reported symptom: one request and the meter passed 300k."""
        import agent as agent_module
        from agent import ArgentAgent

        a = ArgentAgent.__new__(ArgentAgent)
        a.model_name, a.provider = "m", "ollama"
        a.messages = [{"role": "system", "content": "s"}, _image_message()]
        monkeypatch.setattr(agent_module, "get_context_window", lambda: 200_000)
        monkeypatch.setattr(ArgentAgent, "_refresh_system_prompt", lambda self, t=None: False)

        usage = a.get_context_usage()
        assert usage["tokens"] < 5_000

    def test_trimming_does_not_evict_a_conversation_for_one_screenshot(self):
        """The dangerous half: the same estimate decides what to throw away."""
        from src.agent.trimmer import estimate_message_tokens as est

        conversation = [{"role": "user", "content": "важный разговор " * 50}
                        for _ in range(10)]
        talk = sum(est(m, "m", "ollama") for m in conversation)
        image = est(_image_message(), "m", "ollama")
        assert image < talk, "одна картинка не должна весить больше всего разговора"
