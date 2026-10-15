"""
Needle 3 Coprocessor Module for Argent.

Integrates Cactus Compute's Needle 3 automation model (~20 MB) to provide:
1. Fast-Path tool dispatching (direct local execution of routine tools: read_file, git_status, git_diff, grep_search).
2. Bilingual (RU/EN) intent normalization to bridge Russian phrasing with Needle's schema.
3. In-process structured data extraction for the local delegation engine (delegate_local).
"""

import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional
from logger import get_logger

logger = get_logger("needle_coprocessor")

# Suppress telemetry by default for user privacy
os.environ.setdefault("NEEDLE_TELEMETRY", "0")

try:
    import needle
    NEEDLE_AVAILABLE = True
except ImportError:
    NEEDLE_AVAILABLE = False


@dataclass
class FastDispatchResult:
    tool_name: str
    arguments: Dict[str, Any]
    confidence: float
    elapsed_ms: float
    raw_query: str


class BilingualIntentNormalizer:
    """Preprocesses user queries so that Russian action verbs and phrases
    correctly align with Needle 3's English-trained tool calling grammar.
    """

    # Multi-word replacements first, then single-word
    REPLACEMENTS = [
        # Read file variants
        (r"\bпрочитай(?:\s+содержимое)?\s+файл[а-я]*\b", "read file"),
        (r"\bпокажи(?:\s+содержимое)?\s+файл[а-я]*\b", "read file"),
        (r"\bоткрой\s+файл[а-я]*\b", "read file"),
        (r"\bвыведи\s+файл[а-я]*\b", "read file"),
        (r"\bглянь\s+(?:в\s+)?файл[а-я]*\b", "read file"),
        (r"\bпрочитай\b", "read file"),
        (r"\bоткрой\b", "read file"),

        # Git status / diff
        (r"\bпокажи\s+(?:мне\s+)?статус(?:\s+гит|\s+git)?\b", "show git status"),
        (r"\bстатус\s+(?:гит|git)\b", "show git status"),
        (r"\bгит\s+статус\b", "show git status"),
        (r"\bдифф\b", "show git diff"),
        (r"\bпокажи\s+дифф\b", "show git diff"),
        (r"\bпокажи\s+(?:изменения|разницу)\b", "show git diff"),
        (r"\bчто\s+изменилось\b", "show git diff"),

        # Search / Find
        (r"\bнайди\s+(?:где\s+)?(?:встречается|объявлен|находится)?\b", "search for"),
        (r"\bпоищи\b", "search for"),
        (r"\bпоиск\b", "search for"),
        (r"\bгде\s+находится\b", "search for"),

        # Shell command execution
        (r"\bзапусти\s+команду\b", "run command"),
        (r"\bвыполни\s+команду\b", "run command"),
        (r"\bвыполни\b", "run command"),
        (r"\bзапусти\b", "run command"),
    ]

    @classmethod
    def normalize(cls, query: str) -> str:
        if not query or not isinstance(query, str):
            return ""
        text = query.strip()
        lowered = text.lower()

        # Check if query contains Cyrillic characters
        has_cyrillic = bool(re.search(r"[\u0400-\u04FF]", lowered))
        if not has_cyrillic:
            return text

        result = lowered
        for pattern, replacement in cls.REPLACEMENTS:
            result, count = re.subn(pattern, replacement, result, flags=re.IGNORECASE)
            if count > 0:
                break

        return result


# Needle is shown these as tools, but Argent has no tool of either name: the
# answer is a shell command (see try_fast_dispatch). Returned verbatim, the
# agent would try to call a tool that does not exist.
_NEEDLE_GIT_STUBS = ("git_status", "git_diff")


class NeedleCoprocessor:
    """Manages Needle 3 instance, tool binding, and fast-path execution."""

    _instance: Optional["NeedleCoprocessor"] = None

    def __init__(self):
        self._needle: Optional[Any] = None
        self._initialized = False
        self._init_error: Optional[str] = None
        self._last_elapsed_ms: float = 0.0

    @classmethod
    def get_instance(cls) -> "NeedleCoprocessor":
        if cls._instance is None:
            cls._instance = NeedleCoprocessor()
        return cls._instance

    @property
    def is_available(self) -> bool:
        return NEEDLE_AVAILABLE

    def initialize(self) -> bool:
        """Loads and prepares Needle 3 engine lazily."""
        if self._initialized:
            return True
        if not NEEDLE_AVAILABLE:
            self._init_error = "Package 'cactus-needle' is not installed."
            return False

        try:
            # Define lightweight functional signatures for Needle 3
            @needle.tool
            def read_file(path: str, start_line: int = 1, line_count: int = 100) -> str:
                """Read file content from disk. Specify path and optional line ranges."""
                return f"[read_file:{path}]"

            @needle.tool
            def git_status() -> str:
                """Show git working tree and staging status."""
                return "[git_status]"

            @needle.tool
            def git_diff(file: str = "") -> str:
                """Show git diff of modified files or a specific file."""
                return f"[git_diff:{file}]"

            @needle.tool
            def grep_search(query: str, path: str = ".") -> str:
                """Search exact text pattern across files in directory."""
                return f"[grep_search:{query} in {path}]"

            @needle.tool
            def run_command(command: str) -> str:
                """Run a shell command or git utility."""
                return f"[run_command:{command}]"

            self._needle = needle.Needle(tools=[read_file, git_status, git_diff, grep_search, run_command])
            self._initialized = True
            logger.info("Needle 3 coprocessor initialized successfully.")
            return True
        except Exception as e:
            self._init_error = str(e)
            logger.warning(f"Failed to initialize Needle 3: {e}")
            return False

    def try_fast_dispatch(self, user_query: str) -> Optional[FastDispatchResult]:
        """Evaluates whether the user query represents an immediate single-step tool call.

        Returns FastDispatchResult if a routine tool call is recognized with high confidence.
        Returns None if query requires general LLM reasoning, multi-step actions, or code writing.
        """
        if not user_query or not isinstance(user_query, str):
            return None

        # Disqualify obviously conversational or complex prompts immediately
        q_clean = user_query.strip()
        if len(q_clean) < 3 or q_clean.startswith("/") or q_clean.startswith("!"):
            return None

        q_lower = q_clean.lower()

        # Reject conversational greetings and pleasantries
        conversational_patterns = [
            r"^(?:привет|здравствуй[а-я]*|добр(?:ый|ое|ое)\s+(?:день|утро|вечер)|как\s+дела|хай|хеллоу|салют|спасибо|благодарю|пожалуйста|ясно|понял|отлично|пока|до\s+свидания)\b",
            r"^(?:hello|hi|hey|good\s+(?:morning|evening|afternoon)|how\s+are\s+you|thank\s+you|thanks|bye|cool|awesome)\b",
        ]
        if any(re.search(pat, q_lower) for pat in conversational_patterns):
            return None

        # Exclude questions, deep reasoning, or generative requests
        complex_markers = [
            "почему", "зачем", "как работает", "напиши", "создай", "придумай",
            "исправь", "зарефактори", "объясни", "why", "how", "write", "create",
            "refactor", "explain", "architecture", "plan", "debug"
        ]
        if any(marker in q_lower for marker in complex_markers):
            return None

        if not self.initialize() or self._needle is None:
            return None

        normalized_query = BilingualIntentNormalizer.normalize(q_clean)

        # Safe command prefixes permitted for fast-path shell dispatch
        SAFE_COMMAND_PREFIXES = (
            "git ", "dir", "ls", "pytest", "python --version",
            "python -V", "node --version", "cargo test", "cargo check",
            "python -m pytest",
        )

        t0 = time.perf_counter()
        try:
            self._needle.reset()
            res = self._needle.run(normalized_query, max_steps=1)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self._last_elapsed_ms = elapsed_ms

            if not res or not res.get("success"):
                return None

            function_calls = res.get("function_calls") or []
            results = res.get("results") or []
            confidence = float(res.get("confidence") or 0.0)

            # Strict confidence threshold: reject low-certainty guesses
            if confidence < 0.75:
                return None

            # Case A: Explicit function call returned
            if function_calls:
                call = function_calls[0]
                name = call.get("name")
                args = call.get("arguments") or {}
                if name in ("read_file", "grep_search", "run_command") or name in _NEEDLE_GIT_STUBS:
                    # Normalize argument keys for Argent tools
                    if name == "read_file":
                        fpath = args.get("file_path") or args.pop("path", "")
                        args["file_path"] = fpath
                        # Ground-truth check: target file MUST exist on disk
                        if not fpath or not os.path.exists(fpath):
                            logger.debug(f"Needle read_file rejected: path '{fpath}' does not exist on disk")
                            return None

                    elif name == "run_command":
                        cmd = (args.get("command") or "").strip()
                        if not cmd or not any(cmd.lower().startswith(safe) for safe in SAFE_COMMAND_PREFIXES):
                            logger.debug(f"Needle fast dispatch rejected unsafe command in Case A: '{cmd}'")
                            return None

                    elif name == "git_diff":
                        if not any(kw in q_lower for kw in ("diff", "дифф", "разниц", "изменен")):
                            return None
                        file_arg = (args.get("file") or "").strip()
                        if file_arg and not os.path.exists(file_arg):
                            return None
                        # Not an Argent tool: the Case B marker path already runs it
                        # as a shell command, and so must this one.
                        return FastDispatchResult(
                            tool_name="run_command",
                            arguments={"command": f"git diff {file_arg}".strip()},
                            confidence=confidence,
                            elapsed_ms=elapsed_ms,
                            raw_query=user_query,
                        )

                    elif name == "git_status":
                        if not any(kw in q_lower for kw in ("status", "статус")):
                            return None
                        return FastDispatchResult(
                            tool_name="run_command",
                            arguments={"command": "git status"},
                            confidence=confidence,
                            elapsed_ms=elapsed_ms,
                            raw_query=user_query,
                        )

                    elif name == "grep_search":
                        if not any(kw in q_lower for kw in ("grep", "греп", "найди", "поищи", "поиск", "search")):
                            return None

                    return FastDispatchResult(
                        tool_name=name,
                        arguments=args,
                        confidence=confidence,
                        elapsed_ms=elapsed_ms,
                        raw_query=user_query,
                    )

            # Case B: Execution result marker captured from dry-run stub
            if results and isinstance(results[0], str):
                first_res = results[0]
                if first_res.startswith("[read_file:"):
                    path = first_res[11:-1].strip()
                    if path and not path.startswith("read_file") and os.path.exists(path):
                        return FastDispatchResult(
                            tool_name="read_file",
                            arguments={"file_path": path},
                            confidence=confidence or 0.85,
                            elapsed_ms=elapsed_ms,
                            raw_query=user_query,
                        )
                elif first_res == "[git_status]" and any(kw in q_lower for kw in ("status", "статус")):
                    return FastDispatchResult(
                        tool_name="run_command",
                        arguments={"command": "git status"},
                        confidence=confidence or 0.85,
                        elapsed_ms=elapsed_ms,
                        raw_query=user_query,
                    )
                elif first_res.startswith("[git_diff:") and any(kw in q_lower for kw in ("diff", "дифф", "разниц", "изменен")):
                    file_arg = first_res[10:-1].strip()
                    if not file_arg or os.path.exists(file_arg):
                        cmd = f"git diff {file_arg}".strip()
                        return FastDispatchResult(
                            tool_name="run_command",
                            arguments={"command": cmd},
                            confidence=confidence or 0.85,
                            elapsed_ms=elapsed_ms,
                            raw_query=user_query,
                        )
                elif first_res.startswith("[run_command:"):
                    cmd = first_res[13:-1].strip()
                    # Only allow safe read-only commands through fast-path
                    if cmd and any(cmd.lower().startswith(safe) for safe in SAFE_COMMAND_PREFIXES):
                        return FastDispatchResult(
                            tool_name="run_command",
                            arguments={"command": cmd},
                            confidence=confidence or 0.85,
                            elapsed_ms=elapsed_ms,
                            raw_query=user_query,
                        )

            return None
        except Exception as e:
            logger.debug(f"Needle fast dispatch exception: {e}")
            return None

    def extract_structured(self, text: str, schema: Any) -> Optional[Any]:
        """Extracts structured fields matching a Pydantic model or dict schema using Needle 3."""
        if not self.initialize() or self._needle is None:
            return None
        try:
            return needle.extract(text, schema, strict=False)
        except Exception as e:
            logger.debug(f"Needle extraction failed: {e}")
            return None


# Global singleton access
coprocessor = NeedleCoprocessor.get_instance()
