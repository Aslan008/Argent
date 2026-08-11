"""Lightweight mutation testing for Windows (mutmut replacement).

Applies common mutations to a source file, runs the associated tests,
and reports which mutations survived (not caught by tests).

AST-aware: skips mutations inside docstrings, string literals, comments,
and return type annotations to avoid false positives.
"""
import ast
import io
import os
import re
import subprocess
import sys
import tokenize
from pathlib import Path

# --- Mutation operators ---
# Each: (regex pattern, replacement, description)
# We use simple token-level mutations that are safe to apply on Python source.

MUTATIONS = [
    # Arithmetic operator swaps
    (r'(\+)', '-', 'ADD_TO_SUB'),
    (r'(?<![\w])-', '+', 'SUB_TO_ADD'),  # careful: only minus, not part of identifier
    (r'(\*)', '/', 'MUL_TO_DIV'),
    (r'(?<!/)/(?!/)', '*', 'DIV_TO_MUL'),

    # Comparison flips
    (r'>=', '<=', 'GTE_TO_LTE'),
    (r'<=', '>=', 'LTE_TO_GTE'),
    (r'(?<![<>=!])>(?!=)', '<=', 'GT_TO_LTE'),
    (r'(?<![<>=!])<(?!=)', '>=', 'LT_TO_GTE'),
    (r'==', '!=', 'EQ_TO_NE'),
    (r'!=', '==', 'NE_TO_EQ'),

    # Boolean flips
    (r'\bTrue\b', 'False', 'TRUE_TO_FALSE'),
    (r'\bFalse\b', 'True', 'FALSE_TO_TRUE'),
    (r'\band\b', 'or', 'AND_TO_OR'),
    (r'\bor\b', 'and', 'OR_TO_AND'),

    # Constant mutations
    (r'\b0\b', '1', 'ZERO_TO_ONE'),
    (r'\b1\b', '0', 'ONE_TO_ZERO'),
    (r'\bNone\b', '0', 'NONE_TO_ZERO'),

    # Return value mutations
    (r'return\s+None', 'return 0', 'RET_NONE_TO_ZERO'),

    # String literal mutations
    (r'""', '"x"', 'EMPTY_STR_TO_X'),

    # Slice boundary mutations
    (r'\[-(\d+)\]', lambda m: f'[-{int(m.group(1))+1}]', 'SLICE_OFF_BY_ONE'),
    (r'\[(\d+)\]', lambda m: f'[{int(m.group(1))+1}]', 'INDEX_OFF_BY_ONE'),
]


def _pos_to_offset(source: str, pos: tuple[int, int]) -> int:
    """Convert (row, col) — 1-indexed row, 0-indexed col — to character offset."""
    row, col = pos
    offset = 0
    for i in range(row - 1):
        nl = source.find('\n', offset)
        offset = nl + 1
    return offset + col


def compute_skip_ranges(source: str) -> list[tuple[int, int]]:
    """Find character ranges that should NOT be mutated.

    Covers: string literals (incl. docstrings), comments, and return type
    annotations.  Mutations inside these are false positives — they change
    text that has no runtime effect.
    """
    skip: list[tuple[int, int]] = []

    # --- Strings & comments via tokenize ---
    try:
        tokens = tokenize.tokenize(io.BytesIO(source.encode("utf-8")).readline)
        for tok in tokens:
            if tok.type in (tokenize.STRING, tokenize.COMMENT):
                start = _pos_to_offset(source, tok.start)
                end = _pos_to_offset(source, tok.end)
                skip.append((start, end))
    except tokenize.TokenError:
        pass

    # --- Return annotations via ast ---
    # Skip both the -> arrow AND the annotation itself.
    # The AST node.returns starts at the type (e.g. 'str'), not at '->',
    # so we must manually find the arrow and add it to skip ranges.
    try:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.returns is not None:
                    ret_start = _pos_to_offset(source, (node.returns.lineno, node.returns.col_offset))
                    # Find the '->' arrow before the return annotation
                    arrow_pos = source.rfind('->', 0, ret_start)
                    if arrow_pos != -1:
                        skip.append((arrow_pos, arrow_pos + 2))
                    end = _pos_to_offset(source, (node.returns.end_lineno, node.returns.end_col_offset))
                    skip.append((ret_start, end))
    except SyntaxError:
        pass

    return skip


def _is_in_skip(pos: int, skip_ranges: list[tuple[int, int]]) -> bool:
    """True if *pos* falls inside any skip range."""
    for s, e in skip_ranges:
        if s <= pos < e:
            return True
    return False


def run_tests(test_files: list[str]) -> bool:
    """Run pytest on the given test files. Returns True if all pass."""
    cmd = [sys.executable, "-m", "pytest"] + test_files + ["-x", "-q", "--tb=no", "--no-header", "-p", "no:cacheprovider"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30,
                            cwd=os.getcwd())
    return result.returncode == 0


def mutate_module(module_path: str, test_files: list[str], max_mutations: int = 50):
    """Run mutation testing on a single module.

    Returns (killed, survived, total, details).
    """
    source = Path(module_path).read_text(encoding="utf-8")

    # Compute skip ranges once — AST-aware false-positive filtering
    skip_ranges = compute_skip_ranges(source)

    killed = 0
    survived = 0
    total = 0
    details = []

    # Generate mutations: try each operator on each match position
    for pattern, replacement, op_name in MUTATIONS:
        if total >= max_mutations:
            break

        matches = list(re.finditer(pattern, source))
        for i, m in enumerate(matches):
            if total >= max_mutations:
                break

            # Skip mutations inside strings, comments, and return annotations
            if _is_in_skip(m.start(), skip_ranges):
                continue

            # Apply mutation at this specific position
            if callable(replacement):
                new_text = replacement(m)
            else:
                new_text = replacement

            mutated = source[:m.start()] + new_text + source[m.end():]

            # Skip if mutation produces identical source (e.g. replacing '-' in a comment)
            if mutated == source:
                continue

            # Write mutated source
            Path(module_path).write_text(mutated, encoding="utf-8")

            total += 1
            desc = f"{op_name} @ pos {m.start()} (match: {m.group()!r} -> {new_text!r})"

            try:
                passed = run_tests(test_files)
            except subprocess.TimeoutExpired:
                passed = False  # timeout = likely infinite loop = mutation caught

            if passed:
                survived += 1
                details.append(f"  SURVIVED: {desc}")
            else:
                killed += 1
                details.append(f"  killed:   {desc}")

            # Restore original
            Path(module_path).write_text(source, encoding="utf-8")

    return killed, survived, total, details


def main():
    modules = [
        ("usage_tracker.py", ["tests/test_lmtrust_usage_tracker.py", "tests/test_usage_tracker.py"]),
        ("text_safety.py", ["tests/test_lmtrust_text_safety.py", "tests/test_text_safety.py"]),
        ("tool_recovery.py", ["tests/test_lmtrust_tool_recovery.py", "tests/test_tool_registry_consistency.py"]),
        ("atomic_io.py", ["tests/test_lmtrust_atomic_io.py", "tests/test_atomic_io.py"]),
    ]

    import builtins
    _print = builtins.print

    def print(*args, **kwargs):
        kwargs.setdefault('flush', True)
        _print(*args, **kwargs)

    print("=" * 70)
    print("LMTrust Mutation Testing — Custom (Windows mutmut replacement)")
    print("=" * 70)

    total_killed = 0
    total_survived = 0
    total_mutations = 0

    for module_path, test_files in modules:
        print(f"\n--- Mutating: {module_path} ---")
        print(f"    Tests: {', '.join(test_files)}")

        # Verify tests pass on unmutated code first
        if not run_tests(test_files):
            print(f"    ⚠ BASELINE FAILS — skipping {module_path}")
            continue

        killed, survived, total, details = mutate_module(module_path, test_files, max_mutations=30)

        kill_rate = (killed / total * 100) if total > 0 else 0
        total_killed += killed
        total_survived += survived
        total_mutations += total

        print(f"    Result: {killed}/{total} killed ({kill_rate:.0f}%), {survived} survived")

        if survived > 0:
            print(f"    Surviving mutations:")
            for d in details:
                if d.strip().startswith("SURVIVED"):
                    print(f"      {d.strip()}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    overall = (total_killed / total_mutations * 100) if total_mutations > 0 else 0
    print(f"  Total mutations: {total_mutations}")
    print(f"  Killed:          {total_killed}")
    print(f"  Survived:        {total_survived}")
    print(f"  Kill score:      {overall:.1f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()