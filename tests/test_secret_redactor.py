"""Secrets are hidden in command output, in streams cut at any byte, and in the log."""

import logging

import pytest

import tools.command_ops as command_ops
from src.agent import secret_redactor as sr
from src.agent.secret_redactor import StreamRedactor, redact
from tests.test_command_ops_streaming import _MockProc, _mock_approve

OPENAI = "sk-" + "Ab1" * 12          # sk- + 36 characters
ANTHROPIC = "sk-ant-api03-" + "Zy9_-" * 8
GITHUB = "ghp_" + "a1B2" * 9
JWT = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dBjftJeZ4CVPmB92K27uhbUJU1p1r_wW1gFWFOEjXk"
PEM = ("-----BEGIN RSA PRIVATE KEY-----\n"
       "MIIEowIBAAKCAQEA7vbqajDw4o6gJy8UtmIbkcpnkO3Kwc4qsEnSZp\n"
       "Zm9vYmFyYmF6cXV4\n"
       "-----END RSA PRIVATE KEY-----\n")


@pytest.fixture(autouse=True)
def _on(monkeypatch):
    monkeypatch.setattr(sr, "_enabled_cache", (float("inf"), True))
    monkeypatch.setattr(sr, "KNOWN", sr._Known(collector=lambda: set()))
    yield


# ── what is recognised ─────────────────────────────────────────────────────

@pytest.mark.parametrize("secret,kind", [
    (OPENAI, "openai-key"),
    (ANTHROPIC, "anthropic-key"),
    (GITHUB, "github-token"),
    ("AKIA" + "ABCDEFGHIJKLMNOP", "aws-key-id"),
    ("AIza" + "A" * 35, "google-key"),
    (JWT, "jwt"),
    ("123456789:" + "AAH" + "x" * 32, "telegram-token"),
])
def test_vendor_formats_are_replaced(secret, kind):
    out = redact(f"before {secret} after")
    assert secret not in out
    assert f"[секрет скрыт: {kind}]" in out
    assert out.startswith("before ") and out.endswith(" after")


def test_text_without_secrets_is_returned_untouched():
    text = "Exit code: 0\nOUTPUT:\n3 passed in 0.12s, commit a3f9c2e1b4d5 on master\n"
    assert redact(text) is text


def test_authorization_header_and_url_password():
    out = redact("Authorization: Bearer abcdefghijklmnop1234567890\n")
    assert "abcdefghijklmnop1234567890" not in out and out.startswith("Authorization: Bearer ")
    out = redact("git remote: https://alice:hunter2hunter2@github.com/x/y.git")
    assert "hunter2hunter2" not in out and "alice:" in out and "@github.com/x/y.git" in out


def test_name_value_pairs_keep_the_name():
    out = redact("OPENAI_API_KEY=hello-world-12345\nDB_PASSWORD='correcthorsebattery'\n")
    assert "hello-world-12345" not in out and "correcthorsebattery" not in out
    assert "OPENAI_API_KEY=" in out and "DB_PASSWORD='" in out


def test_json_style_pair():
    out = redact('{"api_key": "abcd1234efgh5678", "model": "x"}')
    assert "abcd1234efgh5678" not in out and '"model": "x"' in out


def test_cli_flag():
    out = redact("mysql --user root --password swordfish12 -h db")
    assert "swordfish12" not in out and "--password" in out and "-h db" in out


def test_names_that_only_mention_tokens_are_not_secrets():
    text = "max_tokens=4096\ntotal_tokens: 1234567\ntoken_count = 99999999\nTOKENIZER=sentencepiece\n"
    assert redact(text) == text


def test_placeholders_are_left_alone():
    text = "API_KEY=<your key here>\nSECRET=$SECRET_FROM_ENV\nPASSWORD=********\nTOKEN=none\n"
    assert redact(text) == text


def test_a_strong_looking_token_value_is_replaced_even_under_a_weak_name():
    out = redact("GITHUB_TOKEN=Zx81kQ92LmNp03VtUw54yRa7")
    assert "Zx81kQ92LmNp03VtUw54yRa7" not in out


def test_pem_block_is_removed_whole():
    out = redact("key follows\n" + PEM + "done\n")
    assert "MIIEowIBAAKCAQEA" not in out and "Zm9vYmFy" not in out
    assert "[секрет скрыт: private-key]" in out and out.endswith("done\n")


def test_known_values_from_config_and_environment():
    sr.KNOWN.add("my-very-own-passphrase-77")
    out = redact("echo my-very-own-passphrase-77 | tee x")
    assert "my-very-own-passphrase-77" not in out and "tee x" in out


def test_disabled_filter_passes_everything(monkeypatch):
    monkeypatch.setattr(sr, "_enabled_cache", (float("inf"), False))
    assert redact(f"key {OPENAI}") == f"key {OPENAI}"
    assert StreamRedactor().feed(f"key {OPENAI}") == f"key {OPENAI}"


# ── streams ────────────────────────────────────────────────────────────────

def _stream(chunks):
    r = StreamRedactor()
    out = "".join(r.feed(c) for c in chunks) + r.flush()
    return out


@pytest.mark.parametrize("cut", range(1, len(OPENAI) + 8, 3))
def test_a_secret_cut_by_a_chunk_boundary_is_still_found(cut):
    text = f"token is {OPENAI} ok\n"
    cut = min(cut, len(text) - 1)
    out = _stream([text[:cut], text[cut:]])
    assert OPENAI not in out
    assert "token is " in out and out.endswith(" ok\n")


def test_one_byte_at_a_time():
    text = f"export KEY={OPENAI}\nnext line\n"
    out = _stream(list(text))
    assert OPENAI not in out and "next line\n" in out and out.startswith("export KEY=")


def test_pem_block_split_across_chunks():
    text = "start\n" + PEM + "end\n"
    chunks = [text[i:i + 7] for i in range(0, len(text), 7)]
    out = _stream(chunks)
    assert "MIIEowIBAAKCAQEA" not in out and "Zm9vYmFy" not in out
    assert out.startswith("start\n") and out.endswith("end\n")


def test_prompt_without_trailing_space_is_released_after_a_pause():
    now = [0.0]
    r = StreamRedactor(clock=lambda: now[0])
    assert r.feed("Enter password:") == "Enter "  # the last word is held: it could start a token
    assert r.release_stale(0.25) == ""            # too soon
    now[0] = 1.0
    assert r.release_stale(0.25) == "password:"
    assert r.pending == ""


def test_plain_stream_text_is_not_delayed_beyond_the_last_word():
    r = StreamRedactor()
    assert r.feed("build finished\nnext step ") == "build finished\nnext step "


def test_unterminated_pem_does_not_swallow_the_rest_forever():
    r = StreamRedactor()
    r.feed("-----BEGIN PRIVATE KEY-----\n")
    flood = "A" * 1000 + "\n"
    out = ""
    for _ in range(sr.MAX_PEM_CHARS // 1000 + 5):
        out += r.feed(flood)
    out += r.feed("visible again\n")
    assert "visible again" in out


# ── where it is wired in ───────────────────────────────────────────────────

def test_run_command_output_is_filtered(monkeypatch):
    _mock_approve(monkeypatch)
    text = f"KEY={OPENAI}\nplain line\n".encode()
    chunks = [text[i:i + 5] for i in range(0, len(text), 5)]
    monkeypatch.setattr(command_ops, "_spawn", lambda *a, **k: _MockProc(chunks))

    out = command_ops.run_command("env")
    assert OPENAI not in out
    assert "plain line" in out and "[секрет скрыт" in out


def test_a_prompt_the_filter_held_back_still_reaches_the_model(monkeypatch):
    _mock_approve(monkeypatch)
    monkeypatch.setattr(command_ops, "_spawn", lambda *a, **k: _MockProc([b"Welcome\n", b"Password:"]))
    out = command_ops.run_command("login")
    assert "Password:" in out


def test_the_log_filter_hides_secrets(tmp_path):
    from logger import _SecretFilter

    record = logging.LogRecord("t", logging.INFO, __file__, 1, "header was %s", (f"Bearer-less {OPENAI}",), None)
    assert _SecretFilter().filter(record) is True
    assert OPENAI not in record.getMessage()
