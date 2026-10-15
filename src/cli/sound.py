"""Sound notification module for Argent CLI.

Plays subtle audio feedback when the AI completes a turn, encounters an
interruption/error, or requests user approval/clarification.
Supports custom WAV audio files and built-in sound presets:
- 'system': native Windows MessageBeep / macOS Ping / Linux freedesktop theme
- 'synth': quick, futuristic electronic ticks (Beep 1046Hz / 784Hz / 261Hz)
- 'chord': pleasant harmonic two-tone chimes (C-E / E-G / dissonance)
- 'mute': silence for that specific event

Designed with zero latency: playback executes in a daemon thread to never
block terminal rendering or user input.
"""

import os
import sys
import time
import threading
import subprocess

_PLAY_LOCK = threading.Lock()
_LAST_PLAY_TIME = 0.0
MIN_SOUND_INTERVAL = 0.25  # Cooldown in seconds to prevent sound flooding / thread storms


def _emit_terminal_bell() -> None:
    """Emit terminal bell '\a' only when stdout is an active interactive terminal (TTY)."""
    try:
        if hasattr(sys.stdout, "isatty") and sys.stdout.isatty():
            sys.stdout.write("\a")
            sys.stdout.flush()
    except Exception:
        pass


def _play_custom_file(filepath: str) -> bool:
    """Attempt to play an audio file via platform-native player. Returns True on success."""
    if not filepath or not os.path.isfile(filepath):
        return False

    try:
        if os.name == "nt":
            import winsound
            # Play synchronously inside our already-daemon thread to avoid cutoff on exit
            winsound.PlaySound(filepath, winsound.SND_FILENAME | winsound.SND_NODEFAULT)
            return True
        elif sys.platform == "darwin":
            res = subprocess.run(
                ["afplay", filepath],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
            return res.returncode == 0
        else:
            for player in ("paplay", "aplay", "play"):
                try:
                    res = subprocess.run(
                        [player, filepath],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=5,
                    )
                    if res.returncode == 0:
                        return True
                except Exception:
                    continue
    except Exception:
        pass
    return False


def _play_synth_preset(st: str) -> None:
    """Minimal futuristic clicks/ticks."""
    if os.name == "nt":
        import winsound
        try:
            if st == "error":
                winsound.Beep(261, 110)  # C4 short dull tap
            elif st in ("prompt", "confirm", "attention"):
                winsound.Beep(784, 50)   # G5
                winsound.Beep(1046, 70)  # C6 rising tick
            else:
                winsound.Beep(1046, 60)  # C6 bright single tick
            return
        except Exception:
            pass
    _emit_terminal_bell()


def _play_chord_preset(st: str) -> None:
    """Pleasant two-tone harmonic chimes."""
    if os.name == "nt":
        import winsound
        try:
            if st == "error":
                winsound.Beep(493, 70)  # B4
                winsound.Beep(440, 110) # A4 minor drop
            elif st in ("prompt", "confirm", "attention"):
                winsound.Beep(659, 60)  # E5
                winsound.Beep(784, 90)  # G5 rising inquiry
            else:
                winsound.Beep(523, 60)  # C5
                winsound.Beep(659, 90)  # E5 major chord success
            return
        except Exception:
            pass
    _emit_terminal_bell()


def _play_system_preset(st: str) -> None:
    """Standard operating system notification chimes."""
    if os.name == "nt":
        import winsound
        if st == "error":
            try:
                winsound.MessageBeep(winsound.MB_ICONHAND)
            except Exception:
                winsound.Beep(440, 150)
        elif st in ("prompt", "confirm", "attention"):
            try:
                winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
            except Exception:
                winsound.Beep(650, 150)
        else:
            try:
                winsound.MessageBeep(winsound.MB_ICONASTERISK)
            except Exception:
                winsound.Beep(880, 120)
    elif sys.platform == "darwin":
        if st == "error":
            sound_file = "/System/Library/Sounds/Basso.aiff"
        elif st in ("prompt", "confirm", "attention"):
            sound_file = "/System/Library/Sounds/Tink.aiff"
        else:
            sound_file = "/System/Library/Sounds/Ping.aiff"
        if os.path.exists(sound_file):
            try:
                subprocess.run(["afplay", sound_file], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=2)
                return
            except Exception:
                pass
        _emit_terminal_bell()
    else:
        if st == "error":
            sound_path = "/usr/share/sounds/freedesktop/stereo/dialog-error.oga"
        elif st in ("prompt", "confirm", "attention"):
            sound_path = "/usr/share/sounds/freedesktop/stereo/dialog-warning.oga"
        else:
            sound_path = "/usr/share/sounds/freedesktop/stereo/complete.oga"
        if os.path.exists(sound_path):
            for player in ("paplay", "aplay", "play"):
                try:
                    subprocess.run([player, sound_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=2)
                    return
                except Exception:
                    continue
        _emit_terminal_bell()


def _play_worker(sound_type: str = "done") -> None:
    """Internal sound player running in a background thread."""
    st = str(sound_type).strip().lower() if sound_type else "done"

    # Acquire lock so audio hardware or driver calls are serialized
    with _PLAY_LOCK:
        try:
            from config import get_sound_event
            target = get_sound_event(st)
        except Exception:
            target = "system"

        target_str = str(target).strip()
        target_lower = target_str.lower()

        if target_lower in ("mute", "none", "off", "0", "false"):
            return

        # Try playing as custom audio file if path exists
        expanded_target = os.path.expanduser(target_str)
        if os.path.isfile(expanded_target):
            if _play_custom_file(expanded_target):
                return

        # Built-in presets
        if target_lower == "synth":
            _play_synth_preset(st)
        elif target_lower == "chord":
            _play_chord_preset(st)
        else:
            _play_system_preset(st)


def play_sound(sound_type: str = "done", force: bool = False, async_play: bool = True) -> None:
    """Play a notification sound.

    Args:
        sound_type: 'done' for turn completion, 'prompt' for approval/question, 'error' for failure.
        force: If True, bypasses cooldown and global config setting (e.g. for /sound test).
        async_play: If True, executes sound in a daemon thread so caller is never delayed.
    """
    global _LAST_PLAY_TIME

    norm_type = str(sound_type).strip().lower() if sound_type else "done"
    if norm_type in ("confirm", "attention"):
        norm_type = "prompt"

    if not force:
        try:
            from config import get_sound_notification, get_sound_event
            if not get_sound_notification():
                return

            # Quick exit if event is muted without spawning an OS thread
            target = str(get_sound_event(norm_type)).strip().lower()
            if target in ("mute", "none", "off", "0", "false"):
                return
        except Exception:
            return

        now = time.monotonic()
        if (now - _LAST_PLAY_TIME) < MIN_SOUND_INTERVAL:
            # Debounce: suppress rapid successive sounds (e.g. error storms)
            return
        _LAST_PLAY_TIME = now

    if async_play:
        t = threading.Thread(target=_play_worker, args=(norm_type,), daemon=True)
        t.start()
    else:
        _play_worker(norm_type)


def handle_sound_command(command: str) -> None:
    """Unified handler for /sound CLI command with Rich table output."""
    from ui import print_system, print_error, console
    from config import (
        get_sound_notification,
        set_sound_notification,
        get_sound_events_config,
        set_sound_event,
    )

    raw = command.strip()
    if raw.startswith("/sound"):
        args_str = raw[len("/sound"):].strip()
    else:
        args_str = raw

    parts = args_str.split(maxsplit=1)
    subcmd = parts[0].lower() if parts else ""
    rest = parts[1].strip() if len(parts) > 1 else ""

    if subcmd in ("on", "1", "true", "enable"):
        set_sound_notification(True)
        print_system("🔔 Звуковые сигналы включены [bold green]ON[/bold green].")
        play_sound("done", force=True)
        return

    if subcmd in ("off", "0", "false", "disable"):
        set_sound_notification(False)
        print_system("🔕 Звуковые сигналы отключены [bold red]OFF[/bold red].")
        return

    if subcmd in ("test", "play"):
        event = rest.lower() if rest else "done"
        if event in ("confirm", "attention"):
            event = "prompt"
        if event not in ("done", "prompt", "error"):
            event = "done"
        cfg = get_sound_events_config()
        curr_target = cfg.get(event, "system")
        if curr_target.lower() in ("mute", "none", "off", "0", "false"):
            print_system(f"🔔 Событие '[cyan]{event}[/cyan]' имеет статус [dim yellow]mute[/dim yellow] (звук отключен).")
        else:
            print_system(f"🔔 Проигрывание тестового сигнала для '[cyan]{event}[/cyan]' (цель: [yellow]{curr_target}[/yellow])...")
            play_sound(event, force=True)
        return

    if subcmd == "set":
        set_parts = rest.split(maxsplit=1)
        if len(set_parts) < 2 or not set_parts[1].strip():
            print_error("Использование: /sound set <done|prompt|error|all> <system|synth|chord|mute|путь.wav>")
            return

        target_event = set_parts[0].lower()
        if target_event in ("confirm", "attention"):
            target_event = "prompt"
        if target_event not in ("done", "prompt", "error", "all"):
            print_error(f"Неизвестное событие: '{target_event}'. Доступны: done, prompt, error, all")
            return

        raw_val = set_parts[1].strip().strip("'\"")
        val_lower = raw_val.lower()

        if val_lower in ("system", "synth", "chord", "mute", "none", "off"):
            set_sound_event(target_event, val_lower)
            print_system(f"🔔 Для '[cyan]{target_event}[/cyan]' установлен пресет: [bold green]{val_lower}[/bold green]")
            if val_lower not in ("mute", "none", "off"):
                sample_event = "done" if target_event == "all" else target_event
                play_sound(sample_event, force=True)
            return

        # Check file path
        expanded_path = os.path.abspath(os.path.expanduser(raw_val))
        if not os.path.isfile(expanded_path):
            # Check ~/.argent/sounds/<name>
            argent_sounds = os.path.expanduser(os.path.join("~", ".argent", "sounds", raw_val))
            if os.path.isfile(argent_sounds):
                expanded_path = os.path.abspath(argent_sounds)

        if os.path.isfile(expanded_path):
            if not expanded_path.lower().endswith(".wav"):
                print_system("[dim yellow]⚠️ Предупреждение: рекомендуется формат .wav (несжатый PCM).[/dim yellow]")
            set_sound_event(target_event, expanded_path)
            print_system(f"🔔 Для '[cyan]{target_event}[/cyan]' назначен аудиофайл: [bold green]{expanded_path}[/bold green]")
            sample_event = "done" if target_event == "all" else target_event
            play_sound(sample_event, force=True)
            return
        else:
            print_error(
                f"Файл не найден: '{raw_val}'.\n"
                "Укажите существующий путь к .wav файлу или выберите пресет: system, synth, chord, mute"
            )
            return

    # Status & Help display
    from rich.table import Table
    is_on = get_sound_notification()
    status_label = "[bold green]ВКЛЮЧЕНЫ (ON)[/bold green]" if is_on else "[bold red]ОТКЛЮЧЕНЫ (OFF)[/bold red]"
    events_cfg = get_sound_events_config()

    table = Table(title=f"🔔 Звуковые сигналы: {status_label}", show_header=True, header_style="bold cyan")
    table.add_column("Событие", style="bold yellow")
    table.add_column("Назначенный звук", style="white")
    table.add_column("Описание", style="dim")

    desc_map = {
        "system": "Системные звуки Windows / macOS",
        "synth": "Электронный футуристичный клик (Beep)",
        "chord": "Двухтональный гармоничный перезвон",
        "mute": "Без звука для этого события",
    }

    event_labels = {
        "done": "Завершение ответа (done)",
        "prompt": "Запрос подтверждения / вопрос (prompt)",
        "error": "Ошибка / прерывание (error)",
    }

    for ev in ("done", "prompt", "error"):
        val = events_cfg.get(ev, "system")
        if os.path.isfile(os.path.expanduser(val)):
            desc = "Пользовательский WAV-файл"
        else:
            desc = desc_map.get(val.lower(), "Пресет")
        table.add_row(event_labels.get(ev, ev), val, desc)

    console.print(table)
    print_system(
        "\n[bold]Использование:[/bold]\n"
        "  /sound [on|off]           — включить или выключить все звуки\n"
        "  /sound set <событие> <звук> — назначить звук (done, prompt, error, all)\n"
        "                              Варианты: system, synth, chord, mute или путь к C:\\sound.wav\n"
        "  /sound test [событие]     — проиграть тестовый сигнал (done / prompt / error)\n"
    )
