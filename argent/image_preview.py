"""Кадр для терминала: модель читает файл, человек видит картинку рядом с карточкой.

Рисовать настоящее изображение в терминале без декодера на стороне Rust просто:
сетку RGB готовит PIL, а фронтенд рисует каждую пару пикселей одной полублочной
клеткой (`▀` — верхний пиксель цветом текста, нижний цветом фона).
"""

from __future__ import annotations

import base64
import json
from typing import Any, Dict, Optional

from argent.logger import get_logger

log = get_logger("image_preview")

#: Ширина маленького превью в знакоместах — под карточкой инструмента.
#: Меньше, чем раньше (34): свёрнутая карточка — напоминание, что это картинка,
#: а не место, где её читают. Подробности — по клику.
THUMB_COLS = 26
#: Кадр, который едет в опросе состояния: достаточен для раскрытой карточки и дешёв
#: (сетка 320 знакомест — это ~200 КБ base64 каждые пару секунд, в опрос не годится).
FRAME_COLS = 160
#: Сетка, которую строят, когда человек действительно открыл картинку.
SCREEN_COLS = 320

_last: Dict[str, Any] = {}
_seq = 0


def _resample():
    from PIL import Image

    return getattr(Image, "LANCZOS", 1)


def grid(img, cols: int) -> Optional[Dict[str, Any]]:
    """Сетка шириной `cols` знакомест; клетка — два пикселя по высоте."""
    w, h = img.size
    if w <= 0 or h <= 0 or cols <= 0:
        return None
    # Пропорции сохраняются: знакоместо вдвое выше своей ширины.
    rows = max(2, round(cols * (h / w) / 2))
    cell = img.resize((cols, rows * 2), _resample())
    return {
        "cols": cols,
        "rows": rows,
        "rgb": base64.b64encode(cell.tobytes()).decode("ascii"),
    }


def build(path: str, caption: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Две сетки одного кадра: маленькая и крупная. None, если файл не картинка."""
    try:
        from PIL import Image

        with Image.open(path) as raw:
            img = raw.convert("RGB")
        thumb = grid(img, THUMB_COLS)
        full = grid(img, FRAME_COLS)
        if not thumb or not full:
            return None
        return {"path": str(path), "caption": caption or "", "thumb": thumb, "full": full}
    except Exception as e:
        log.debug("no preview for %s: %s", path, e)
        return None


def full_snapshot(path: Optional[str] = None, cols: int = SCREEN_COLS) -> str:
    """Строка для полноэкранного просмотра — сетка по требованию.

    Её нет в опросе состояния намеренно: 320 знакомест RGB — это ~200 КБ base64,
    а опрос идёт каждые пару секунд. Строить её в момент, когда человек нажал на
    картинку, стоит одного чтения файла, который всё равно в кэше ОС.

    Форма та же, что у `build()`: фронтенд разбирает её тем же кодом.
    """
    target = path or _last.get("path")
    if not target:
        return ""
    # Asked for a picture other than the newest one (an older card was clicked): the
    # caption and the small frame belong to THAT file, not to whatever was published
    # last. Rebuilding them costs one file read, and a wrong caption is worse.
    same = not path or str(path) == str(_last.get("path") or "")
    caption = _last.get("caption", "") if same else ""
    thumb = _last.get("thumb") if same else None
    try:
        from PIL import Image

        with Image.open(target) as raw:
            img = raw.convert("RGB")
        big = grid(img, cols)
        if not big:
            return ""
        if not thumb:
            thumb = grid(img, THUMB_COLS) or big
        payload = {
            "path": str(target),
            "caption": caption,
            "thumb": thumb,
            "full": big,
            "seq": _last.get("seq", 0),
        }
        return json.dumps(payload, ensure_ascii=False)
    except Exception as e:
        log.debug("full preview failed for %s: %s", target, e)
        return ""


def publish(path: Optional[str], caption: Optional[str] = None) -> None:
    """Запомнить кадр, который фронтенд нарисует рядом с карточкой инструмента."""
    global _seq
    if not path:
        return
    payload = build(path, caption)
    if not payload:
        return
    _seq += 1
    payload["seq"] = _seq
    _last.clear()
    _last.update(payload)


def snapshot() -> str:
    """Текст для опроса состояния: пустая строка, пока кадров не было."""
    return json.dumps(_last, ensure_ascii=False) if _last else ""


def reset() -> None:
    """Забыть кадр (тесты и новая сессия)."""
    global _seq
    _last.clear()
    _seq = 0
