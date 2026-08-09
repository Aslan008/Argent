"""Letting the model actually look at an image.

browser_screenshot wrote PNGs to disk that nothing could read: no provider sent
pictures back, so "take a screenshot to see the page" was a wasted turn. This
is the missing half.

The work is in two places nobody thinks about until it breaks:

* **Capability.** A model that cannot see does not say so — it answers about an
  image it never received, confidently. Ollama reports `capabilities` from
  /api/show, so the answer is known rather than guessed; asked, minimax-m3 and
  qwen3.5 list "vision" while glm-5.2 does not.
* **Size.** A full-page screenshot is megabytes, and base64 adds a third. Sent
  raw it either overflows the context or is silently truncated into garbage, so
  images are downscaled before encoding — resolution beyond what the model can
  attend to buys nothing and costs tokens linearly.
"""

import base64
import os
from pathlib import Path

from logger import get_logger

log = get_logger("vision")

# Beyond this the extra pixels stop buying accuracy and start costing context.
MAX_EDGE_PIXELS = 1568
# A hard ceiling on the encoded payload, whatever the resizing achieved.
MAX_ENCODED_BYTES = 4 * 1024 * 1024

SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}

_CAPABILITY_CACHE: dict = {}


def _ollama_host() -> str:
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    return host if host.startswith("http") else f"http://{host}"


def model_supports_vision(model: str, provider: str) -> bool | None:
    """True / False / None when it genuinely cannot be determined.

    None matters: refusing to send an image because we could not ask is worse
    than trying and letting the provider complain, and claiming support we did
    not verify is how a model ends up describing a picture it never got.
    """
    if not model:
        return False
    key = (provider, model)
    if key in _CAPABILITY_CACHE:
        return _CAPABILITY_CACHE[key]

    result = None
    if provider == "ollama":
        try:
            import requests
            resp = requests.post(f"{_ollama_host()}/api/show",
                                 json={"model": model}, timeout=6)
            resp.raise_for_status()
            caps = (resp.json() or {}).get("capabilities") or []
            result = "vision" in [str(c).lower() for c in caps]
        except Exception as e:
            log.info("vision capability for %r unknown: %s", model, e)
            result = None

    _CAPABILITY_CACHE[key] = result
    return result


def encode_image(path) -> tuple:
    """(base64 payload, media type, note) — or (None, None, reason)."""
    p = Path(path)
    if not p.is_file():
        return None, None, f"файл не найден: {path}"
    if p.suffix.lower() not in SUPPORTED_SUFFIXES:
        return None, None, (f"{p.suffix} — не изображение. Поддерживаются: "
                            f"{', '.join(sorted(SUPPORTED_SUFFIXES))}")

    raw = p.read_bytes()
    media = "image/jpeg" if p.suffix.lower() in (".jpg", ".jpeg") else \
            f"image/{p.suffix.lower().lstrip('.')}"
    note = ""

    try:
        from PIL import Image

        with Image.open(p) as img:
            width, height = img.size
            if max(width, height) > MAX_EDGE_PIXELS:
                # Downscale before encoding, not after: the cost is in the
                # pixels, and a model cannot attend past this size anyway.
                scale = MAX_EDGE_PIXELS / max(width, height)
                new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
                img = img.convert("RGB").resize(new_size, Image.LANCZOS)
                import io
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=85)
                raw = buffer.getvalue()
                media = "image/jpeg"
                note = f"уменьшено с {width}×{height} до {new_size[0]}×{new_size[1]}"
    except ImportError:
        note = "Pillow не установлен — изображение отправлено без уменьшения"
    except Exception as e:
        log.info("could not resize %s (%s); sending as-is", path, e)

    payload = base64.b64encode(raw).decode("ascii")
    if len(payload) > MAX_ENCODED_BYTES:
        return None, None, (f"изображение слишком большое даже после сжатия "
                            f"({len(payload) // 1024} КБ) — сохраните его меньше")
    return payload, media, note


def build_image_message(payloads: list, text: str) -> dict:
    """A provider-neutral user message carrying images.

    Kept neutral on purpose: each provider spells attachments differently
    (Ollama takes `images`, OpenAI-compatible endpoints take content parts),
    and the history is shared between them — a message stored in one dialect
    would break the moment the user switches provider mid-conversation.
    """
    return {
        "role": "user",
        "content": text,
        "images": [{"data": data, "media_type": media} for data, media in payloads],
    }


# --- pending queue -------------------------------------------------------
# A tool returns a string; it cannot hand back a picture. So view_image records
# what to attach and the turn loop drains it, exactly like the diff events.

_pending: list = []


def queue_image(payload: str, media_type: str, label: str) -> None:
    _pending.append({"data": payload, "media_type": media_type, "label": label})


def drain_images() -> list:
    out = list(_pending)
    _pending.clear()
    return out
