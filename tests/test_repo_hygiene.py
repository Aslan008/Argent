"""Keys, tokens and personal data must not be in the repository (scripts/repo_hygiene.py).

The program collects requests, feedback and trajectories for local training and Argent checkpoints
commit the whole tree, so the guard has three layers: .gitignore keeps the collected data out, this test
fails when a tracked file holds a credential or the machine owner's identity, and the pre-push hook
(scripts/git_hooks/pre-push) refuses to publish a commit that adds one.
"""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import repo_hygiene as H  # noqa: E402

TERMS = {"names": ["jdoe-account"], "emails": ["jdoe.private@mail.test"], "extra": []}


def fake(prefix: str, n: int = 30) -> str:
    """A credential-shaped string built at run time, so this file does not itself contain one."""
    return prefix + "a1B2c3D4" * (n // 8 + 1)


def git(*args, cwd):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout


class TestScanner:
    @pytest.mark.parametrize("secret", [
        lambda: fake("KGAT_"),
        lambda: fake("sk-or-v1-"),
        lambda: fake("AIza", 36),
        lambda: fake("ghp_", 36),
        lambda: fake("hf_", 36),
        lambda: "123456789:" + fake("AA", 40),
        lambda: "https://up.example/o?uploadType=resumable&upload_id=" + fake("AP6r", 40),
        lambda: "token = '" + fake("", 40) + "'",
        lambda: "MY_API_TOKEN=" + fake("", 24),
        lambda: "Authorization: Bearer " + fake("", 40),
    ])
    def test_credentials_are_found(self, secret):
        text = f"line before\n{secret()}\nline after"
        assert H.scan_text(text, TERMS), text
        assert not H.scan_text(H.redact_text(text, TERMS), TERMS)

    def test_personal_terms_are_found_and_replaced(self):
        text = r'cwd C:\\Users\\jdoe-account\\Desktop\\Argent and /c/Users/jdoe-account/x, mail jdoe.private@mail.test'
        kinds = {k for k, _ in H.scan_text(text, TERMS)}
        assert {"personal_name", "personal_email"} <= kinds
        clean = H.redact_text(text, TERMS)
        assert "jdoe" not in clean and "Users\\\\user\\\\Desktop" in clean and "/c/Users/user/x" in clean
        assert H.PLACEHOLDER_EMAIL in clean

    def test_ordinary_text_is_left_alone(self):
        text = "def sk_learn(): return 'ask-me-anything'\nprint('token budget is 5000')\nBearer of bad news\n"
        assert H.scan_text(text, TERMS) == []
        assert H.redact_text(text, TERMS) == text

    def test_generic_account_names_are_not_treated_as_personal(self, monkeypatch):
        monkeypatch.delenv("USERNAME", raising=False)
        monkeypatch.delenv("USER", raising=False)
        monkeypatch.setattr(H.getpass, "getuser", lambda: "Administrator")
        monkeypatch.setattr(H.Path, "home", staticmethod(lambda: Path("C:/Users/user")))
        monkeypatch.setattr(H, "_git", lambda *a, **k: "")
        assert H.personal_terms()["names"] == []


class TestRepository:
    def test_tracked_files_hold_no_keys_or_personal_data(self):
        if not (ROOT / ".git").exists():
            pytest.skip("not a git checkout")
        found = H.scan_files(H.tracked_files(), H.personal_terms())
        assert not found, (
            "keys or personal data in tracked files (run `python scripts/repo_hygiene.py --fix`, and put "
            f"collected data in .gitignore): { {k: sorted({kind for kind, _ in v}) for k, v in found.items()} }")

    @pytest.mark.parametrize("path", [
        "src/leeth/data/leeth_feedback.jsonl",
        "src/leeth/data/leeth_trajectories.jsonl",
        "src/leeth/data/shadow_mined.jsonl",
        "src/micromodels/data/training_pairs.jsonl",
        "src/leeth/data_400m/leeth.onnx",
        ".env",
    ])
    def test_collected_data_and_weights_are_ignored(self, path):
        if not (ROOT / ".git").exists():
            pytest.skip("not a git checkout")
        done = subprocess.run(["git", "check-ignore", "-q", path], cwd=ROOT)
        assert done.returncode == 0, f"{path} must be in .gitignore"

    def test_nothing_tracked_is_something_the_ignore_rules_exclude(self):
        """A file that was committed before its ignore rule existed stays tracked (and keeps being
        committed by every checkpoint): chat exports, collected data, weights."""
        if not (ROOT / ".git").exists():
            pytest.skip("not a git checkout")
        out = subprocess.run(["git", "ls-files", "-ci", "--exclude-standard"], cwd=ROOT,
                             capture_output=True, text=True).stdout.split()
        assert not out, f"tracked although ignored (git rm --cached them): {out[:10]}"

    def test_the_pre_push_hook_is_shipped_and_uses_the_scanner(self):
        hook = (ROOT / "scripts" / "git_hooks" / "pre-push").read_text(encoding="utf-8")
        assert "repo_hygiene.py" in hook and "--range" in hook and "refs/heads" in hook


class TestRangeScan:
    @pytest.fixture
    def repo(self, tmp_path):
        git("init", "-q", cwd=tmp_path)
        git("config", "user.email", "t@example.com", cwd=tmp_path)
        git("config", "user.name", "t", cwd=tmp_path)
        (tmp_path / "ok.txt").write_text("nothing here\n", encoding="utf-8")
        git("add", "-A", cwd=tmp_path)
        git("commit", "-q", "-m", "base", cwd=tmp_path)
        return tmp_path

    def scan(self, repo, *revs):
        old = H.ROOT
        H.ROOT = repo
        try:
            return H.scan_range(list(revs), TERMS)
        finally:
            H.ROOT = old

    def test_an_added_secret_is_found_even_if_a_later_commit_removes_it(self, repo):
        base = git("rev-parse", "HEAD", cwd=repo).strip()
        (repo / "cfg.py").write_text(f'KEY = "{fake("sk-")}"\n', encoding="utf-8")
        git("add", "-A", cwd=repo)
        git("commit", "-q", "-m", "add config", cwd=repo)
        (repo / "cfg.py").write_text("KEY = None\n", encoding="utf-8")
        git("commit", "-q", "-am", "remove it", cwd=repo)
        found = self.scan(repo, f"{base}..HEAD")
        assert "cfg.py" in found                      # still in the history that would be pushed

    def test_removing_a_secret_is_not_a_finding(self, repo):
        (repo / "cfg.py").write_text(f'KEY = "{fake("sk-")}"\n', encoding="utf-8")
        git("add", "-A", cwd=repo)
        git("commit", "-q", "-m", "add config", cwd=repo)
        base = git("rev-parse", "HEAD", cwd=repo).strip()
        (repo / "cfg.py").write_text("KEY = None\n", encoding="utf-8")
        git("commit", "-q", "-am", "remove it", cwd=repo)
        assert self.scan(repo, f"{base}..HEAD") == {}

    def test_personal_data_in_a_commit_message_and_clean_commits(self, repo):
        base = git("rev-parse", "HEAD", cwd=repo).strip()
        (repo / "a.txt").write_text("fine\n", encoding="utf-8")
        git("add", "-A", cwd=repo)
        git("commit", "-q", "-m", "mail me at jdoe.private@mail.test", cwd=repo)
        found = self.scan(repo, f"{base}..HEAD")
        assert any(k.startswith("commit message") for k in found)
        (repo / "b.txt").write_text("fine too\n", encoding="utf-8")
        git("add", "-A", cwd=repo)
        git("commit", "-q", "-m", "clean one", cwd=repo)
        head = git("rev-parse", "HEAD", cwd=repo).strip()
        assert self.scan(repo, f"{head}~1..{head}") == {}

    def test_a_file_a_git_host_would_refuse_is_reported(self, repo, monkeypatch):
        monkeypatch.setattr(H, "OVERSIZE_BYTES", 1000)
        base = git("rev-parse", "HEAD", cwd=repo).strip()
        (repo / "model.bin").write_bytes(b"\x01" * 5000)
        git("add", "-A", cwd=repo)
        git("commit", "-q", "-m", "weights", cwd=repo)
        found = self.scan(repo, f"{base}..HEAD")
        assert any(kind == "oversize" for hits in found.values() for kind, _ in hits)
