"""
LMTrust Deep Vision & Screenshot Verification Suite.

Validates:
1. Desktop screen capture & clipboard retrieval (src/automation/desktop.py)
2. Alpha compositing (Black background bug prevention) in vision.py
3. Region of Interest / Crop functionality
4. Extended formats (.tiff, .tif, .ico)
5. Auto-attach in desktop_tools and browser_tools
6. Initial image attachment in agent._run_turn
7. Telegram bot photo and image-document reception
"""

import base64
import os
from pathlib import Path
from unittest.mock import MagicMock, patch
from PIL import Image
import pytest

from vision import (
    SUPPORTED_SUFFIXES,
    encode_image,
    drain_images,
    queue_image,
    build_image_message,
    _parse_crop_box,
)
from src.automation.desktop import DesktopAutomation
from tools.desktop_tools import (
    capture_desktop_screen,
    capture_desktop_window,
    get_clipboard_image,
)
from tools.browser_tools import browser_screenshot


class TestVisionAlphaAndCrop:
    """L2 Boundary & Image Processing tests."""

    def test_l2_alpha_compositing_avoids_black_background(self, tmp_path):
        """L2: Transparent RGBA images must composite onto WHITE background, not black."""
        img_path = tmp_path / "transparent.png"
        # Create a 50x50 image with total transparency (alpha = 0)
        img = Image.new("RGBA", (50, 50), (0, 0, 0, 0))
        img.save(img_path, "PNG")

        payload, media_type, note = encode_image(img_path)
        assert payload is not None
        assert media_type == "image/jpeg"

        # Decode the JPEG payload and check pixel value
        raw = base64.b64decode(payload)
        import io
        decoded = Image.open(io.BytesIO(raw))
        # White background means all channels near 255 (allowing slight JPEG compression artifacts)
        r, g, b = decoded.getpixel((25, 25))[:3]
        assert r > 240 and g > 240 and b > 240, f"Pixel was black or dark instead of white: ({r}, {g}, {b})"

    def test_l2_crop_box_parsing_formats(self):
        """L2: _parse_crop_box must correctly parse dict, list, tuple, and string inputs."""
        w, h = 1920, 1080

        # Dict with x, y, width, height
        d1 = {"x": 100, "y": 200, "width": 300, "height": 400}
        assert _parse_crop_box(d1, w, h) == (100, 200, 400, 600)

        # Dict with left, top, right, bottom
        d2 = {"left": 50, "top": 60, "right": 500, "bottom": 600}
        assert _parse_crop_box(d2, w, h) == (50, 60, 500, 600)

        # List [x1, y1, x2, y2]
        l1 = [10, 20, 300, 400]
        assert _parse_crop_box(l1, w, h) == (10, 20, 300, 400)

        # String "10,20,300,400"
        s1 = "10,20,300,400"
        assert _parse_crop_box(s1, w, h) == (10, 20, 300, 400)

        # Out-of-bounds clamping
        oob = {"x": -50, "y": -50, "width": 3000, "height": 2000}
        res = _parse_crop_box(oob, w, h)
        assert res == (0, 0, 1920, 1080)

    def test_l1_extended_supported_suffixes(self):
        """L1: Formats .tiff, .tif, .ico must be recognized as supported images."""
        assert ".tiff" in SUPPORTED_SUFFIXES
        assert ".tif" in SUPPORTED_SUFFIXES
        assert ".ico" in SUPPORTED_SUFFIXES

    def test_l2_crop_applies_to_image(self, tmp_path):
        """L2: encode_image with crop must produce an image with cropped dimensions."""
        img_path = tmp_path / "canvas.png"
        img = Image.new("RGB", (1000, 800), color=(128, 128, 128))
        img.save(img_path, "PNG")

        payload, media_type, note = encode_image(img_path, crop={"x": 100, "y": 100, "w": 200, "h": 150})
        assert payload is not None
        assert "вырезан фрагмент" in note

        import io
        decoded = Image.open(io.BytesIO(base64.b64decode(payload)))
        assert decoded.width == 200
        assert decoded.height == 150


class TestAutoAttachAndDesktopTools:
    """L5 State & Cross-tool auto-attach tests."""

    def setup_method(self):
        drain_images()

    def teardown_method(self):
        drain_images()

    def test_l5_capture_desktop_window_auto_attaches(self):
        """L5: capture_desktop_window must queue image for vision models automatically."""
        fake_result = {
            "success": True,
            "path": ".argent/artifacts/desktop/test_win.png",
            "title": "Test Window",
            "width": 800,
            "height": 600,
            "base64": "fakeb64payload",
            "media_type": "image/jpeg",
        }
        with patch.object(DesktopAutomation, "is_available", return_value=True):
            with patch.object(DesktopAutomation, "capture_window", return_value=fake_result):
                with patch("config.get_current_model", return_value="gpt-4o"):
                    with patch("config.get_provider", return_value="openai"):
                        res = capture_desktop_window("Test Window")
                        assert "VISION ATTACHED" in res
                        queued = drain_images()
                        assert len(queued) == 1
                        assert queued[0]["data"] == "fakeb64payload"

    def test_l5_capture_desktop_screen_auto_attaches(self):
        """L5: capture_desktop_screen must capture screen and queue image."""
        fake_result = {
            "success": True,
            "path": ".argent/artifacts/desktop/screen.png",
            "title": "Full Desktop Screen",
            "width": 1920,
            "height": 1080,
            "base64": "fakescreenpayload",
            "media_type": "image/jpeg",
        }
        with patch.object(DesktopAutomation, "is_available", return_value=True):
            with patch.object(DesktopAutomation, "capture_screen", return_value=fake_result):
                with patch("config.get_current_model", return_value="gpt-4o"):
                    with patch("config.get_provider", return_value="openai"):
                        res = capture_desktop_screen("primary")
                        assert "VISION ATTACHED" in res
                        queued = drain_images()
                        assert len(queued) == 1
                        assert queued[0]["data"] == "fakescreenpayload"

    def test_l5_get_clipboard_image_auto_attaches(self):
        """L5: get_clipboard_image must queue image when clipboard has an image."""
        fake_result = {
            "success": True,
            "path": ".argent/artifacts/desktop/clipboard.png",
            "title": "Clipboard Screenshot",
            "width": 400,
            "height": 300,
            "base64": "fakeclippayload",
            "media_type": "image/jpeg",
        }
        with patch.object(DesktopAutomation, "get_clipboard_image", return_value=fake_result):
            with patch("config.get_current_model", return_value="gpt-4o"):
                with patch("config.get_provider", return_value="openai"):
                    res = get_clipboard_image()
                    assert "VISION ATTACHED" in res
                    queued = drain_images()
                    assert len(queued) == 1
                    assert queued[0]["data"] == "fakeclippayload"

    def test_l5_browser_screenshot_auto_attaches(self, tmp_path):
        """L5: browser_screenshot must queue image for vision models."""
        dummy_img = tmp_path / "browser_shot.png"
        Image.new("RGB", (100, 100), (0, 128, 0)).save(dummy_img, "PNG")

        with patch("browser_engine.browser_engine.run", return_value=f"Screenshot saved to: {dummy_img}"):
            with patch("config.get_current_model", return_value="gpt-4o"):
                with patch("config.get_provider", return_value="openai"):
                    res = browser_screenshot()
                    assert "VISION ATTACHED" in res
                    queued = drain_images()
                    assert len(queued) == 1
                    assert queued[0]["media_type"] in ("image/png", "image/jpeg")


class TestAgentTurnAndTelegramBotReception:
    """L5 State & L6 Integration tests for turn start and Telegram."""

    def test_l5_agent_run_turn_attaches_queued_images_at_start(self):
        """L5: Pre-queued images must become part of initial user message in agent._run_turn."""
        from agent import ArgentAgent

        drain_images()
        queue_image("init_payload_123", "image/jpeg", "PreAttached.png")

        a = ArgentAgent.__new__(ArgentAgent)
        a.messages = []
        a._refresh_system_prompt = MagicMock()
        a._maybe_auto_retrieve = MagicMock()
        a._trim_history = MagicMock()

        # Generator runs until first yield or break
        # Advance generator to execute initial setup before first stream yield
        gen = a._run_turn("Explain this screenshot")
        try:
            next(gen)
        except Exception:
            pass

        assert len(a.messages) >= 1
        first_msg = a.messages[0]
        assert first_msg["role"] == "user"
        assert first_msg.get("images") is not None
        assert first_msg["images"][0]["data"] == "init_payload_123"

    def test_l6_telegram_bot_handles_photo_message(self, tmp_path):
        """L6: Telegram bot _handle_message must download photo and queue image."""
        from src.telegram.bot import TelegramBridge, TelegramClient

        drain_images()
        dummy_jpg = tmp_path / "tg_download.jpg"
        Image.new("RGB", (200, 200), (0, 0, 255)).save(dummy_jpg, "JPEG")

        bridge = TelegramBridge.__new__(TelegramBridge)
        bridge.client = MagicMock(spec=TelegramClient)
        bridge._is_authorized = MagicMock(return_value=True)
        bridge._run_prompt_turn = MagicMock()

        # Mock get_file and download_file
        bridge.client.get_file.return_value = {"file_path": "photos/photo_1.jpg"}
        def fake_download(remote_path, local_path):
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            Path(local_path).write_bytes(dummy_jpg.read_bytes())
            return True
        bridge.client.download_file.side_effect = fake_download

        msg = {
            "chat": {"id": 123456},
            "from": {"id": 123456},
            "caption": "Why does this fail?",
            "photo": [
                {"file_id": "thumb_id", "file_size": 100},
                {"file_id": "highres_id", "file_size": 5000},
            ]
        }

        bridge._handle_message(msg)

        # Must have called download for highres_id
        bridge.client.get_file.assert_called_with("highres_id")
        bridge._run_prompt_turn.assert_called_once()
        turn_text = bridge._run_prompt_turn.call_args[0][1]
        assert turn_text == "Why does this fail?"

        # Image must be in queue
        queued = drain_images()
        assert len(queued) == 1
        assert queued[0]["media_type"] == "image/jpeg"

    def test_l7_agent_recovers_from_unsupported_image_provider_error(self):
        """L7: If provider rejects image input (e.g. KoboldCPP without mmproj), strip images and retry."""
        from agent import ArgentAgent
        from providers import ProviderError

        drain_images()
        queue_image("init_payload_123", "image/jpeg", "PreAttached.png")

        a = ArgentAgent.__new__(ArgentAgent)
        a.provider = "koboldcpp"
        a.model_name = "ORNITH1.5:35BA3B"
        a.max_context_tokens = 4096
        a.messages = []
        a._current_turn_msg_index = 0
        a.tools_catalog = []
        a.tools_schema = []
        a.strategy = MagicMock()
        a.strategy.wants_constrained_decoding.return_value = False
        a.strategy.step_system_reminder.return_value = None
        a.strategy.supports_native_tools.return_value = True
        a._native_tools_unsupported = False
        a._turn_cost_dollars = 0.0
        a.total_cost_dollars = 0.0
        a._active_subagent = None
        a._running_commands = {}
        a._refresh_system_prompt = MagicMock()
        a._maybe_auto_retrieve = MagicMock()
        a._trim_history = MagicMock()
        a._log_turn_end = MagicMock()

        mock_provider = MagicMock()
        mock_provider.name = "koboldcpp"
        mock_provider.validate_config.return_value = None
        mock_provider.supports_constrained_decoding.return_value = False

        attempt = 0
        def fake_stream(*args, **kwargs):
            nonlocal attempt
            attempt += 1
            if attempt == 1:
                raise ProviderError(
                    "KOBOLDCPP error (HTTP 500): Error code: 500 - "
                    "{'error': {'code': 500, 'message': 'image input is not supported - "
                    "hint: if this is unexpected, you may need to provide the mmproj', 'type': 'server_error'}}"
                )
            yield {"content": "Text-only recovery response."}

        mock_provider.stream_chat.side_effect = fake_stream

        with patch("agent.create_provider", return_value=mock_provider):
            chunks = list(a._run_turn("Analyze this image"))

        # Verify agent yielded notice and continued to get content
        error_notices = [c["content"] for c in chunks if c.get("type") == "error"]
        assert any("не поддерживает чтение картинок" in n for n in error_notices)

        content_chunks = [c["content"] for c in chunks if c.get("type") == "content_stream"]
        assert "Text-only recovery response." in "".join(content_chunks)

        # Verify images are stripped from messages
        assert all("images" not in m for m in a.messages if isinstance(m, dict))

