"""Keys, tokens and personal data must never reach the repository.

One scanner for three jobs:

  python scripts/repo_hygiene.py                 # scan every tracked file, exit 1 on a finding
  python scripts/repo_hygiene.py --staged        # scan what is staged (pre-commit)
  python scripts/repo_hygiene.py --range A..B    # scan the added lines of commits A..B (pre-push);
                                                 # everything after --range goes to `git log`
  python scripts/repo_hygiene.py --fix           # redact tracked text files in place

What counts as a finding:
  * credentials: API keys and tokens of the common providers, bot tokens, JWTs, private keys, and
    `api_key = "..."`-style assignments with a long literal value;
  * personal data of whoever runs this: their OS user name (also inside C:/Users/<name> paths), their
    git e-mail address, and any extra terms they list in `.argent/privacy_terms.txt` (one per line).
    Nothing personal is hard-coded here: the terms are read from the machine, so this file itself stays clean.

Fixtures that must contain fake credentials (the redactor's own tests) are listed in ALLOWED_FILES.
"""

import getpass
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (name, regex, group holding the secret itself or 0)
KEY_PATTERNS = [
    ("kaggle_token", re.compile(r"KGAT_[A-Za-z0-9]{16,}"), 0),
    ("api_key_sk", re.compile(r"\bsk-(?:proj-|or-v1-|ant-)?[A-Za-z0-9_-]{24,}"), 0),
    ("google_api_key", re.compile(r"AIza[0-9A-Za-z_-]{30,}"), 0),
    ("github_token", re.compile(r"\b(?:ghp|gho|ghs|ghu)_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}"), 0),
    ("slack_token", re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"), 0),
    ("hf_token", re.compile(r"\bhf_[A-Za-z0-9]{30,}"), 0),
    ("aws_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b"), 0),
    ("telegram_bot_token", re.compile(r"\b\d{8,10}:AA[A-Za-z0-9_-]{30,}"), 0),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{15,}\.[A-Za-z0-9_-]{15,}\.[A-Za-z0-9_-]{10,}"), 0),
    ("private_key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"), 0),
    ("url_credential",
     re.compile(r"(?i)[?&](?:upload_id|api[_-]?key|access_token|token|sig|signature|x-goog-signature|x-amz-signature)=([A-Za-z0-9%_\-.]{24,})"), 1),
    ("bearer_token", re.compile(r"(?i)\bBearer\s+([A-Za-z0-9._\-]{30,})"), 1),
    ("secret_assignment",
     re.compile(r"""(?i)["']?(?:api[_-]?key|secret|token|password|passwd)["']?\s*[:=]\s*["']([A-Za-z0-9_\-./+=]{20,})["']"""), 1),
    ("env_secret",
     re.compile(r"\b[A-Z][A-Z0-9_]*(?:TOKEN|SECRET|API_KEY|PASSWORD)=([A-Za-z0-9_\-./+=]{16,})"), 1),
]

# Files whose job is to hold fake credentials, and this scanner.
ALLOWED_FILES = {
    "tests/test_secret_redactor.py",
    "src/agent/secret_redactor.py",
    "scripts/repo_hygiene.py",
    "tests/test_repo_hygiene.py",
}

BINARY_SUFFIXES = {
    ".pt", ".onnx", ".data", ".png", ".jpg", ".jpeg", ".gif", ".ico", ".bin", ".npy", ".npz", ".gz", ".zip",
    ".woff", ".woff2", ".pdf", ".so", ".dll", ".exe", ".pyc", ".parquet", ".safetensors", ".wav", ".mp3",
    ".mp4", ".webp", ".db", ".sqlite", ".pkl", ".whl", ".lock",
}
MAX_BYTES = 80 * 1024 * 1024
OVERSIZE_BYTES = 95 * 1000 * 1000      # git hosts refuse a file over 100 MB
PLACEHOLDER_USER = "user"
PLACEHOLDER_EMAIL = "user@example.com"
_GENERIC_NAMES = {"user", "users", "admin", "root", "default", "public", "test", "guest", "owner", "runner",
                  "system", "default user", "all users", "administrator"}


def _git(*args: str, cwd=None) -> str:
    done = subprocess.run(["git", *args], cwd=cwd or ROOT, capture_output=True)
    return done.stdout.decode("utf-8", errors="replace")


def personal_terms() -> dict:
    """{'names': [...], 'emails': [...], 'extra': [...]} for the person running this."""
    names = set()
    for candidate in (getpass.getuser(), Path.home().name, os.environ.get("USERNAME", ""), os.environ.get("USER", "")):
        if candidate and len(candidate) >= 4 and candidate.lower() not in _GENERIC_NAMES:
            names.add(candidate)
    emails = set()
    mail = _git("config", "user.email").strip()
    if mail and "@" in mail and not mail.lower().endswith(("@example.com", "@users.noreply.github.com")):
        emails.add(mail)
        local = mail.split("@", 1)[0]
        if len(local) >= 6:
            names.add(local)
    extra = []
    terms_file = ROOT / ".argent" / "privacy_terms.txt"
    if terms_file.is_file():
        for line in terms_file.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if len(line) >= 4 and not line.startswith("#"):
                extra.append(line)
    return {"names": sorted(names, key=len, reverse=True), "emails": sorted(emails), "extra": extra}


def _name_regex(terms: dict):
    words = [re.escape(t) for t in terms["names"] + terms["extra"]]
    if not words:
        return None
    return re.compile(r"(?<![A-Za-z0-9_])(?:" + "|".join(words) + r")(?![A-Za-z0-9_])", re.I)


def scan_text(text: str, terms: dict) -> list:
    """[(kind, masked_excerpt)] for every finding in `text`."""
    found = []
    for kind, rx, group in KEY_PATTERNS:
        for m in rx.finditer(text):
            value = m.group(group) if group else m.group(0)
            found.append((kind, value[:4] + "…" + f"({len(value)})"))
    for mail in terms["emails"]:
        if mail.lower() in text.lower():
            found.append(("personal_email", mail[:2] + "…"))
    rx = _name_regex(terms)
    if rx is not None:
        for m in rx.finditer(text):
            found.append(("personal_name", m.group(0)[:2] + "…"))
    return found


def redact_text(text: str, terms: dict) -> str:
    """The same text with keys replaced by <REDACTED:kind> and personal terms by neutral placeholders."""
    for kind, rx, group in KEY_PATTERNS:
        if group:
            text = rx.sub(lambda m, k=kind, g=group: m.group(0).replace(m.group(g), f"<REDACTED:{k}>"), text)
        else:
            text = rx.sub(f"<REDACTED:{kind}>", text)
    for mail in terms["emails"]:
        text = re.sub(re.escape(mail), PLACEHOLDER_EMAIL, text, flags=re.I)
    rx = _name_regex(terms)
    if rx is not None:
        text = rx.sub(PLACEHOLDER_USER, text)
    return text


def is_text_candidate(path: str) -> bool:
    return Path(path).suffix.lower() not in BINARY_SUFFIXES


def tracked_files() -> list:
    return [p for p in _git("ls-files", "-z").split("\0") if p]


def staged_files() -> list:
    return [p for p in _git("diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z").split("\0") if p]


def scan_files(paths, terms: dict) -> dict:
    """{path: [(kind, masked)]} for the paths that have findings."""
    out = {}
    for rel in paths:
        if rel in ALLOWED_FILES or not is_text_candidate(rel):
            continue
        p = ROOT / rel
        try:
            if not p.is_file() or p.stat().st_size > MAX_BYTES:
                continue
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        hits = scan_text(text, terms)
        if hits:
            out[rel] = hits
    return out


def scan_range(revs, terms: dict) -> dict:
    """Findings in the ADDED lines of the commits selected by `revs` (a list of `git log` revision
    arguments, e.g. ["A..B"] or ["sha", "--not", "--remotes"]), in their messages, and over-sized files.
    Text files only; deleted lines are not looked at (removing a secret is the cure, not the disease)."""
    revs = [revs] if isinstance(revs, str) else list(revs)
    out = {}
    for chunk in _git("log", *revs, "--format=%H%x00%B%x01").split("\x01"):
        if "\x00" not in chunk:
            continue
        sha, message = chunk.strip().split("\x00", 1)
        hits = scan_text(message, terms)
        if hits:
            out[f"commit message {sha[:10]}"] = hits
    proc = subprocess.Popen(
        ["git", "log", *revs, "-p", "-U0", "--no-color", "--no-renames", "--diff-filter=AM", "--format=commit %H"],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    current = None
    for raw in proc.stdout:
        line = raw.decode("utf-8", errors="ignore")
        if line.startswith("+++ "):
            name = line[4:].strip()
            current = name[2:] if name.startswith("b/") else None
            if current and (current in ALLOWED_FILES or not is_text_candidate(current)):
                current = None
            continue
        if current and line.startswith("+") and not line.startswith("+++"):
            hits = scan_text(line[1:], terms)
            if hits:
                out.setdefault(current, []).extend(hits)
    proc.wait()
    # a file a git host will refuse (100 MB), added by any commit of the range
    sizes = _git("rev-list", "--objects", *revs)
    names = {}
    for entry in sizes.splitlines():
        sha, _, path = entry.partition(" ")
        if path:
            names[sha] = path
    if names:
        info = subprocess.run(["git", "cat-file", "--batch-check=%(objectname) %(objectsize)"], cwd=ROOT,
                              input="\n".join(names).encode(), capture_output=True).stdout.decode()
        for row in info.splitlines():
            sha, _, size = row.partition(" ")
            if size.isdigit() and int(size) > OVERSIZE_BYTES:
                out.setdefault(names[sha], []).append(("oversize", f"{int(size) // 1_000_000} MB"))
    return out


def fix_files(paths, terms: dict) -> list:
    changed = []
    for rel in paths:
        if rel in ALLOWED_FILES or not is_text_candidate(rel):
            continue
        p = ROOT / rel
        try:
            if not p.is_file() or p.stat().st_size > MAX_BYTES:
                continue
            raw = p.read_bytes()
            text = raw.decode("utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        new = redact_text(text, terms)
        if new != text:
            p.write_bytes(new.encode("utf-8"))
            changed.append(rel)
    return changed


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    terms = personal_terms()
    if "--fix" in argv:
        changed = fix_files(tracked_files(), terms)
        print(f"redacted {len(changed)} file(s)")
        return 0
    if "--range" in argv:
        found = scan_range(argv[argv.index("--range") + 1:], terms)
    elif "--staged" in argv:
        found = scan_files(staged_files(), terms)
    else:
        found = scan_files(tracked_files(), terms)
    for rel, hits in sorted(found.items()):
        kinds = sorted({k for k, _ in hits})
        print(f"{rel}: {len(hits)} finding(s) [{', '.join(kinds)}]")
    if found:
        print(f"\n{len(found)} file(s) with keys or personal data — not safe to publish.", file=sys.stderr)
        return 1
    print("clean: no keys or personal data in the scanned files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
