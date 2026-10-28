"""Прикреплённые картинки: полоса над вводом, полный просмотр, удаление.

Раньше единственным следом прикреплённого кадра была строка в чате: увидеть,
что именно уходит со следующим сообщением, и убрать его было нельзя.
"""

import base64
import io
import json

import pytest
from PIL import Image

from argent import vision


@pytest.fixture(autouse=True)
def clean_queue():
    vision.clear_images()
    yield
    vision.clear_images()


def _png(w: int = 40, h: int = 20, color=(200, 30, 30)) -> str:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_a_queued_picture_gets_a_strip_preview():
    vision.queue_image(_png(), "image/png", "скриншот из буфера (40×20)")

    items = json.loads(vision.pending_snapshot())

    assert len(items) == 1
    assert items[0]["label"] == "скриншот из буфера (40×20)"
    thumb = items[0]["thumb"]
    assert thumb["cols"] == vision.STRIP_COLS
    assert thumb["rows"] >= 1
    # Клетка — два пикселя по высоте, три байта на пиксель.
    assert len(base64.b64decode(thumb["rgb"])) == thumb["cols"] * thumb["rows"] * 2 * 3


def test_the_full_view_of_an_attachment_is_a_preview_payload():
    vision.queue_image(_png(), "image/png", "кадр")

    payload = json.loads(vision.image_full(0))

    assert payload["caption"] == "кадр"
    assert payload["thumb"]["cols"] == vision.STRIP_COLS
    assert payload["full"]["cols"] > payload["thumb"]["cols"], "крупная сетка крупнее миниатюры"
    assert "seq" in payload


def test_dropping_takes_only_that_picture_off_the_message():
    vision.queue_image(_png(), "image/png", "первый")
    vision.queue_image(_png(), "image/png", "второй")

    assert vision.drop_image(0) == 1

    labels = [i["label"] for i in json.loads(vision.pending_snapshot())]
    assert labels == ["второй"]


def test_an_out_of_range_drop_is_harmless():
    vision.queue_image(_png(), "image/png", "один")

    assert vision.drop_image(5) == 1
    assert vision.image_full(5) == ""


def test_a_picture_that_cannot_be_previewed_still_goes_out():
    vision.queue_image("не base64!", "image/png", "битый")

    items = json.loads(vision.pending_snapshot())

    assert len(items) == 1, "запись на месте"
    assert items[0]["thumb"] is None, "превью нет, но и падения нет"
    assert [e["label"] for e in vision.drain_images()] == ["битый"]


def test_the_snapshot_is_valid_json_when_nothing_is_attached():
    assert json.loads(vision.pending_snapshot()) == []
