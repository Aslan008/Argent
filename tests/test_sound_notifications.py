"""Tests for sound notification system and /sound command in Argent."""

import os
import sys
from unittest.mock import patch, MagicMock
import pytest

from config import get_sound_notification, set_sound_notification
from src.cli.sound import play_sound, _play_worker


class TestSoundNotifications:
    """Suite covering sound notifications logic, config persistence, and CLI commands."""

    def test_sound_config_toggle(self, monkeypatch):
        """get_sound_notification and set_sound_notification persist flag correctly."""
        cache = {}
        monkeypatch.setattr("config._CONFIG_CACHE", cache)
        monkeypatch.setattr("config.save_config", lambda c: None)

        set_sound_notification(True)
        assert get_sound_notification() is True

        set_sound_notification(False)
        assert get_sound_notification() is False

    def test_play_sound_respects_config_disabled(self, monkeypatch):
        """When disabled in config and not forced, no sound is dispatched."""
        monkeypatch.setattr("config.get_sound_notification", lambda: False)
        worker_mock = MagicMock()
        monkeypatch.setattr("src.cli.sound._play_worker", worker_mock)

        play_sound("done", force=False, async_play=False)
        worker_mock.assert_not_called()

    def test_play_sound_force_plays_even_if_disabled(self, monkeypatch):
        """force=True plays sound regardless of config setting (e.g. for /sound test)."""
        monkeypatch.setattr("config.get_sound_notification", lambda: False)
        worker_mock = MagicMock()
        monkeypatch.setattr("src.cli.sound._play_worker", worker_mock)

        play_sound("done", force=True, async_play=False)
        worker_mock.assert_called_once_with("done")

    def test_play_sound_windows_messagebeep_calls(self, monkeypatch):
        """On Windows, MessageBeep is called with appropriate icons for done, prompt, and error."""
        monkeypatch.setattr(os, "name", "nt")
        fake_winsound = MagicMock()
        fake_winsound.MB_ICONASTERISK = 0x00000040
        fake_winsound.MB_ICONHAND = 0x00000010
        fake_winsound.MB_ICONEXCLAMATION = 0x00000030

        with patch.dict(sys.modules, {"winsound": fake_winsound}):
            _play_worker("done")
            fake_winsound.MessageBeep.assert_called_with(fake_winsound.MB_ICONASTERISK)

            _play_worker("prompt")
            fake_winsound.MessageBeep.assert_called_with(fake_winsound.MB_ICONEXCLAMATION)

            _play_worker("error")
            fake_winsound.MessageBeep.assert_called_with(fake_winsound.MB_ICONHAND)


    def test_play_sound_windows_fallback_to_beep_on_error(self, monkeypatch):
        """If MessageBeep raises an exception on Windows, falls back to winsound.Beep."""
        monkeypatch.setattr(os, "name", "nt")
        fake_winsound = MagicMock()
        fake_winsound.MessageBeep.side_effect = RuntimeError("Audio device busy")

        with patch.dict(sys.modules, {"winsound": fake_winsound}):
            _play_worker("done")
            fake_winsound.Beep.assert_called_once()

    def test_play_sound_posix_stdout_bell_fallback(self, monkeypatch):
        """On POSIX systems without sound files, terminal bell '\\a' is emitted when isatty is True."""
        monkeypatch.setattr(os, "name", "posix")
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setattr(os.path, "exists", lambda p: False)

        mock_stdout = MagicMock()
        mock_stdout.isatty.return_value = True
        monkeypatch.setattr(sys, "stdout", mock_stdout)

        _play_worker("done")
        mock_stdout.write.assert_called_with("\a")
        mock_stdout.flush.assert_called_once()

    def test_play_sound_posix_redirect_isatty_false_suppresses_bell(self, monkeypatch):
        """When stdout is redirected (not a TTY), bell '\\a' is suppressed to prevent binary pollution."""
        monkeypatch.setattr(os, "name", "posix")
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setattr(os.path, "exists", lambda p: False)

        mock_stdout = MagicMock()
        mock_stdout.isatty.return_value = False
        monkeypatch.setattr(sys, "stdout", mock_stdout)

        _play_worker("done")
        mock_stdout.write.assert_not_called()

    def test_play_sound_debounce_suppresses_rapid_successive_calls(self, monkeypatch):
        """Rapid successive sound calls within MIN_SOUND_INTERVAL are debounced unless force=True."""
        monkeypatch.setattr("config.get_sound_notification", lambda: True)
        worker_mock = MagicMock()
        monkeypatch.setattr("src.cli.sound._play_worker", worker_mock)

        import src.cli.sound as sound_mod
        sound_mod._LAST_PLAY_TIME = 0.0

        # First call passes
        play_sound("done", force=False, async_play=False)
        assert worker_mock.call_count == 1

        # Immediate second call is suppressed by cooldown
        play_sound("done", force=False, async_play=False)
        assert worker_mock.call_count == 1

        # Force call bypasses cooldown
        play_sound("done", force=True, async_play=False)
        assert worker_mock.call_count == 2


    def test_cli_prompt_autocomplete_contains_sound_commands(self):
        """The autocomplete dictionary in cli_prompt includes /sound and its variations."""
        from src.cli.cli_prompt import COMMAND_HELP
        assert "/sound" in COMMAND_HELP
        assert "/sound on" in COMMAND_HELP
        assert "/sound off" in COMMAND_HELP
        assert "/sound test" in COMMAND_HELP
        assert "/sound status" in COMMAND_HELP

    def test_request_approval_triggers_prompt_sound(self, monkeypatch):
        """When an interactive approval prompt is presented, play_sound('prompt') is triggered."""
        from approval import _raw_terminal_prompt
        import questionary

        sound_mock = MagicMock()
        monkeypatch.setattr("src.cli.sound.play_sound", sound_mock)
        monkeypatch.setattr(questionary, "confirm", lambda msg: MagicMock(ask=lambda: True))

        decision = _raw_terminal_prompt("rm -rf /", destructive=True, grant_key=None)
        assert decision == "once"
        sound_mock.assert_called_with("prompt")

    def test_ask_user_questions_triggers_prompt_sound(self, monkeypatch):
        """When asking user questions, play_sound('prompt') is triggered."""
        from tools.misc_tools import ask_user_questions

        sound_mock = MagicMock()
        monkeypatch.setattr("src.cli.sound.play_sound", sound_mock)
        monkeypatch.setattr("prompt_toolkit.prompt", lambda *a, **k: "User answer")

        res = ask_user_questions([{"type": "text", "question": "What is your goal?"}])
        sound_mock.assert_called_with("prompt")

    def test_sound_events_config_and_setters(self, monkeypatch):
        """get_sound_event and set_sound_event manage per-event routing."""
        from config import get_sound_events_config, get_sound_event, set_sound_event
        cache = {}
        monkeypatch.setattr("config._CONFIG_CACHE", cache)
        monkeypatch.setattr("config.save_config", lambda c: None)

        assert get_sound_event("done") == "system"
        assert get_sound_event("prompt") == "system"
        assert get_sound_event("error") == "system"

        set_sound_event("done", "synth")
        assert get_sound_event("done") == "synth"
        assert get_sound_event("prompt") == "system"

        set_sound_event("all", "chord")
        assert get_sound_event("done") == "chord"
        assert get_sound_event("prompt") == "chord"
        assert get_sound_event("error") == "chord"

    def test_play_sound_preset_synth(self, monkeypatch):
        """When synth preset is active on Windows, Beep is called with electronic tick frequencies."""
        monkeypatch.setattr(os, "name", "nt")
        monkeypatch.setattr("config.get_sound_event", lambda ev: "synth")
        fake_winsound = MagicMock()

        with patch.dict(sys.modules, {"winsound": fake_winsound}):
            _play_worker("done")
            fake_winsound.Beep.assert_called_with(1046, 60)

            _play_worker("error")
            fake_winsound.Beep.assert_called_with(261, 110)

    def test_play_sound_preset_chord(self, monkeypatch):
        """When chord preset is active on Windows, two-tone Beep sequences are played."""
        monkeypatch.setattr(os, "name", "nt")
        monkeypatch.setattr("config.get_sound_event", lambda ev: "chord")
        fake_winsound = MagicMock()

        with patch.dict(sys.modules, {"winsound": fake_winsound}):
            _play_worker("done")
            assert fake_winsound.Beep.call_count == 2
            fake_winsound.Beep.assert_any_call(523, 60)
            fake_winsound.Beep.assert_any_call(659, 90)

    def test_play_sound_preset_mute_does_not_play(self, monkeypatch):
        """When event is set to mute/none, no audio calls are made."""
        monkeypatch.setattr(os, "name", "nt")
        monkeypatch.setattr("config.get_sound_event", lambda ev: "mute")
        fake_winsound = MagicMock()

        with patch.dict(sys.modules, {"winsound": fake_winsound}):
            _play_worker("done")
            fake_winsound.Beep.assert_not_called()
            fake_winsound.MessageBeep.assert_not_called()
            fake_winsound.PlaySound.assert_not_called()

    def test_play_sound_custom_wav_file(self, monkeypatch):
        """When event points to an existing WAV file, PlaySound is invoked with SND_FILENAME | SND_NODEFAULT."""
        monkeypatch.setattr(os, "name", "nt")
        fake_path = r"C:\Sounds\alert.wav"
        monkeypatch.setattr("config.get_sound_event", lambda ev: fake_path)
        monkeypatch.setattr(os.path, "isfile", lambda p: os.path.abspath(p) == os.path.abspath(fake_path))

        fake_winsound = MagicMock()
        fake_winsound.SND_FILENAME = 0x00020000
        fake_winsound.SND_NODEFAULT = 0x00000002

        with patch.dict(sys.modules, {"winsound": fake_winsound}):
            _play_worker("prompt")
            fake_winsound.PlaySound.assert_called_once_with(
                fake_path,
                fake_winsound.SND_FILENAME | fake_winsound.SND_NODEFAULT,
            )

    def test_handle_sound_command_set_preset_and_quoted_path(self, monkeypatch):
        """handle_sound_command parses presets, quoted paths, and reports errors cleanly."""
        from src.cli.sound import handle_sound_command
        from config import get_sound_event

        cache = {}
        monkeypatch.setattr("config._CONFIG_CACHE", cache)
        monkeypatch.setattr("config.save_config", lambda c: None)
        monkeypatch.setattr("src.cli.sound.play_sound", lambda *a, **k: None)

        # Set preset
        handle_sound_command("/sound set done synth")
        assert get_sound_event("done") == "synth"

        # Set all to chord
        handle_sound_command("/sound set all chord")
        assert get_sound_event("done") == "chord"
        assert get_sound_event("prompt") == "chord"
        assert get_sound_event("error") == "chord"

        # Quoted valid file path
        test_file = r"C:\Users\test\chime.wav"
        monkeypatch.setattr(os.path, "isfile", lambda p: os.path.abspath(p) == os.path.abspath(test_file))

        handle_sound_command(f'/sound set prompt "{test_file}"')
        assert os.path.abspath(get_sound_event("prompt")) == os.path.abspath(test_file)

        # Nonexistent file doesn't set path
        monkeypatch.setattr(os.path, "isfile", lambda p: False)
        prev_error = get_sound_event("error")
        handle_sound_command('/sound set error "C:\\nonexistent.wav"')
        assert get_sound_event("error") == prev_error

    def test_sound_config_string_booleans(self, monkeypatch):
        """String booleans ('false', '0', 'off') in config are correctly evaluated as False."""
        from config import get_sound_notification
        cache = {}
        monkeypatch.setattr("config._CONFIG_CACHE", cache)

        for false_val in ("false", "0", "off", "no", "disable"):
            cache["sound_notification"] = false_val
            assert get_sound_notification() is False

        for true_val in ("true", "1", "on", "yes", "enable"):
            cache["sound_notification"] = true_val
            assert get_sound_notification() is True

    def test_set_sound_event_rejects_unknown_event(self):
        """set_sound_event raises ValueError for invalid event names."""
        from config import set_sound_event
        with pytest.raises(ValueError, match="Unknown sound event"):
            set_sound_event("invalid_event", "synth")

    def test_play_sound_muted_event_skips_thread_creation(self, monkeypatch):
        """play_sound returns early without spawning threads if event is muted."""
        monkeypatch.setattr("config.get_sound_notification", lambda: True)
        monkeypatch.setattr("config.get_sound_event", lambda ev: "mute")
        worker_mock = MagicMock()
        monkeypatch.setattr("src.cli.sound._play_worker", worker_mock)

        play_sound("done", force=False, async_play=False)
        worker_mock.assert_not_called()




