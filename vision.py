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
import io
import os
from pathlib import Path
from typing import Any, Optional, Tuple

from logger import get_logger

log = get_logger("vision")

# Beyond this the extra pixels stop buying accuracy and start costing context.
MAX_EDGE_PIXELS = 1568
# A hard ceiling on the encoded payload, whatever the resizing achieved.
MAX_ENCODED_BYTES = 4 * 1024 * 1024

SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".tif", ".ico"}

_CAPABILITY_CACHE: dict = {}


def _ollama_host() -> str:
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    return host if host.startswith("http") else f"http://{host}"


KNOWN_NON_VISION_PATTERNS = (
    "ornith",
    "deepseek-coder",
    "qwen2.5-coder",
    "deepseek-r1",
    "deepseek-v3",
    "qwq",
    "starcoder",
    "codellama",
    "wizardcoder",
    "sqlcoder",
    "codeqwen",
)

KNOWN_VISION_PATTERNS = (
    "vl",
    "vision",
    "llava",
    "minicpm-v",
    "pixtral",
    "internvl",
    "cogvlm",
    "phi-3-vision",
    "phi-3.5-vision",
    "phi-4-multimodal",
    "claude",
)


def _check_koboldcpp_vision() -> bool | None:
    try:
        from config import get_koboldcpp_url
        import requests
        base_url = get_koboldcpp_url()
        if base_url.endswith("/v1"):
            api_url = base_url[:-3]
        elif base_url.endswith("/v1/"):
            api_url = base_url[:-4]
        else:
            api_url = base_url.rstrip("/")
        resp = requests.get(f"{api_url}/api/extra/props", timeout=1.0)
        if resp.status_code == 200:
            data = resp.json() or {}
            if "has_mmproj" in data:
                return bool(data["has_mmproj"])
    except Exception as e:
        log.debug("KoboldCPP vision probe failed: %s", e)
    return None


def mark_vision_unsupported(provider: str, model: str) -> None:
    """Explicitly cache that a model/provider cannot handle image inputs."""
    if model and provider:
        _CAPABILITY_CACHE[(provider, model)] = False


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
    elif provider == "koboldcpp":
        result = _check_koboldcpp_vision()

    # If backend probe was inconclusive, check model family heuristics
    if result is None and model:
        m_lower = model.lower()
        if any(p in m_lower for p in KNOWN_VISION_PATTERNS):
            result = True
        elif any(p in m_lower for p in KNOWN_NON_VISION_PATTERNS):
            result = False

    _CAPABILITY_CACHE[key] = result
    return result


def _parse_crop_box(crop: Any, width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
    """Parse crop arguments into (left, top, right, bottom) clamped to image dimensions."""
    if not crop:
        return None

    left, top, right, bottom = 0, 0, width, height
    try:
        if isinstance(crop, dict):
            left = int(crop.get("left", crop.get("x", 0)))
            top = int(crop.get("top", crop.get("y", 0)))
            if "right" in crop:
                right = int(crop["right"])
            elif "width" in crop or "w" in crop:
                right = left + int(crop.get("width", crop.get("w", 0)))
            if "bottom" in crop:
                bottom = int(crop["bottom"])
            elif "height" in crop or "h" in crop:
                bottom = top + int(crop.get("height", crop.get("h", 0)))
        elif isinstance(crop, (list, tuple)) and len(crop) == 4:
            c0, c1, c2, c3 = [int(v) for v in crop]
            # Disambiguate [x, y, w, h] vs [x1, y1, x2, y2]
            if c2 > c0 and c3 > c1 and c2 <= width and c3 <= height:
                left, top, right, bottom = c0, c1, c2, c3
            else:
                left, top, right, bottom = c0, c1, c0 + c2, c1 + c3
        elif isinstance(crop, str):
            parts = [int(p.strip()) for p in crop.replace("x", ",").replace(":", ",").split(",") if p.strip()]
            if len(parts) == 4:
                return _parse_crop_box(parts, width, height)

        left = max(0, min(left, width - 1))
        top = max(0, min(top, height - 1))
        right = max(left + 1, min(right, width))
        bottom = max(top + 1, min(bottom, height))
        return left, top, right, bottom
    except Exception as e:
        log.warning("Failed parsing crop argument %r: %s", crop, e)
        return None


def _composite_alpha_on_white(img):
    """Avoid black-background bug by compositing alpha transparency onto clean white."""
    from PIL import Image
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.split()[3])
        return background
    return img.convert("RGB")


def encode_image(path, crop: Any = None) -> tuple:
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
    notes = []

    try:
        from PIL import Image

        with Image.open(p) as img:
            orig_w, orig_h = img.size
            crop_box = _parse_crop_box(crop, orig_w, orig_h)
            needs_downscale = max(orig_w, orig_h) > MAX_EDGE_PIXELS
            has_transparency = (img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info))

            if crop_box or needs_downscale or has_transparency:
                if crop_box:
                    img = img.crop(crop_box)
                    notes.append(f"вырезан фрагмент {crop_box[0]},{crop_box[1]}..{crop_box[2]},{crop_box[3]}")

                if has_transparency:
                    img = _composite_alpha_on_white(img)

                width, height = img.size
                _resample = getattr(Image, "Resampling", Image).LANCZOS
                if max(width, height) > MAX_EDGE_PIXELS:
                    scale = MAX_EDGE_PIXELS / max(width, height)
                    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
                    img = img.resize(new_size, _resample)
                    notes.append(f"уменьшено с {width}×{height} до {new_size[0]}×{new_size[1]}")

                working_img = img.convert("RGB")
                best_payload = None

                for quality in (85, 75, 60):
                    buffer = io.BytesIO()
                    working_img.save(buffer, format="JPEG", quality=quality)
                    test_raw = buffer.getvalue()
                    test_payload = base64.b64encode(test_raw).decode("ascii")
                    best_payload = test_payload
                    raw = test_raw
                    media = "image/jpeg"
                    if len(test_payload) <= MAX_ENCODED_BYTES:
                        break

                if best_payload and len(best_payload) > MAX_ENCODED_BYTES:
                    fallback_size = (max(1, int(working_img.width * 0.75)), max(1, int(working_img.height * 0.75)))
                    working_img = working_img.resize(fallback_size, _resample)
                    buffer = io.BytesIO()
                    working_img.save(buffer, format="JPEG", quality=60)
                    raw = buffer.getvalue()
                    media = "image/jpeg"
                    notes.append(f"адаптивно сжато до {fallback_size[0]}×{fallback_size[1]}")

    except ImportError:
        notes.append("Pillow не установлен — изображение отправлено без уменьшения")
    except Exception as e:
        log.info("could not process %s (%s); sending as-is", path, e)

    payload = base64.b64encode(raw).decode("ascii")
    if len(payload) > MAX_ENCODED_BYTES:
        return None, None, (f"изображение слишком большое даже после сжатия "
                            f"({len(payload) // 1024} КБ) — сохраните его меньше")
    return payload, media, "; ".join(notes)


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
