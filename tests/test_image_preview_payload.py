"""Сетка превью: пропорции, размер полезной нагрузки и поведение на не-картинках."""

import json

import pytest

from argent import image_preview as ip


@pytest.fixture(autouse=True)
def clean():
    ip.reset()
    yield
    ip.reset()


def _png(tmp_path, w=400, h=300, name="shot.png"):
    pytest.importorskip("PIL")
    from PIL import Image

    path = tmp_path / name
    Image.new("RGB", (w, h), (30, 120, 200)).save(path)
    return str(path)


def test_both_grids_keep_the_picture_proportions(tmp_path):
    payload = ip.build(_png(tmp_path, 400, 300))
    assert payload is not None

    # The poll carries a cheap frame; the screen-sized grid is built on demand
    # (`full_snapshot`), because it is ~200 KB of base64.
    for key, cols in (("thumb", ip.THUMB_COLS), ("full", ip.FRAME_COLS)):
        grid = payload[key]
        assert grid["cols"] == cols
        # Клетка вдвое выше ширины: 4:3 при ширине cols — это cols*3/8 строк.
        assert abs(grid["rows"] - cols * (300 / 400) / 2) <= 1
        # 3 байта на пиксель и два пикселя на клетку.
        raw = __import__("base64").b64decode(grid["rgb"])
        assert len(raw) == grid["cols"] * grid["rows"] * 2 * 3


def test_the_thumbnail_is_smaller_than_the_frame(tmp_path):
    """Свёрнутая карточка — напоминание, а не место, где читают картинку."""
    payload = ip.build(_png(tmp_path))
    assert payload["thumb"]["cols"] < payload["full"]["cols"]


def test_the_full_screen_grid_is_wide(tmp_path):
    """По требованию — сетка шириной в экран, а не в 120 знакомест."""
    ip.publish(_png(tmp_path, 1600, 900), "скриншот")
    big = json.loads(ip.full_snapshot())

    assert big["full"]["cols"] == ip.SCREEN_COLS
    # 16:9 at 320 columns: 320 * (900/1600) / 2 = 90 rows of two pixels each.
    assert big["full"]["rows"] >= 80, "строк должно хватать на весь экран"
    assert big["caption"] == "скриншот"
    assert big["path"].endswith("shot.png")


def test_the_full_screen_grid_needs_a_picture(tmp_path):
    assert ip.full_snapshot() == ""


def test_the_payload_stays_small(tmp_path):
    """Он ездит в опросе состояния, поэтому не должен весить мегабайты."""
    ip.publish(_png(tmp_path, 1600, 900), "скриншот")
    assert len(ip.snapshot()) < 200_000


def test_publish_remembers_only_the_newest_picture(tmp_path):
    first = _png(tmp_path, name="a.png")
    second = _png(tmp_path, name="b.png")

    ip.publish(first, "первый")
    ip.publish(second, "второй")
    data = json.loads(ip.snapshot())

    assert data["path"] == second
    assert data["caption"] == "второй"
    assert data["seq"] == 2, "счётчик растёт, чтобы фронтенд понял: кадр новый"


def test_a_file_that_is_not_a_picture_is_simply_not_previewed(tmp_path):
    text = tmp_path / "notes.txt"
    text.write_text("не картинка", encoding="utf-8")

    assert ip.build(str(text)) is None
    ip.publish(str(text))
    assert ip.snapshot() == "", "нет кадра — нет поля"


def test_the_sequence_starts_with_the_new_session():
    ip.reset()
    assert ip.snapshot() == ""
    assert json.loads(ip.snapshot() or "{}") == {}
