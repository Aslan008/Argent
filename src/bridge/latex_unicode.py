"""LaTeX в структурный Unicode — то, что терминал действительно может показать.

Модель пишет формулы как `$E = mc^2$`, `$\\frac{a}{b}$`, `$$\\sum_{i=1}^{n} x_i$$`.
Rich (а за ним и TUI) рисует разметку Markdown, но формул не знает: `$...$` попадало
на экран сырым LaTeX, а `\\frac` местами съедался как управляющий символ.

Это подмножество, а не движок: греческие буквы, операторы, дроби, корни, степени и
индексы, `\\text{…}`. Неизвестная команда остаётся как есть — потерять формулу хуже,
чем показать её исходником. Содержимое блоков кода и инлайн-кода не трогается.
"""

from __future__ import annotations

import re
from typing import List, Tuple

# ── symbols ─────────────────────────────────────────────────────────────────

_SYMBOLS = {
    # греческие (строчные)
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε",
    "varepsilon": "ε", "zeta": "ζ", "eta": "η", "theta": "θ", "vartheta": "θ",
    "iota": "ι", "kappa": "κ", "lambda": "λ", "mu": "μ", "nu": "ν", "xi": "ξ",
    "pi": "π", "rho": "ρ", "sigma": "σ", "tau": "τ", "upsilon": "υ",
    "phi": "φ", "varphi": "φ", "chi": "χ", "psi": "ψ", "omega": "ω",
    # греческие (заглавные)
    "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ", "Lambda": "Λ", "Xi": "Ξ",
    "Pi": "Π", "Sigma": "Σ", "Upsilon": "Υ", "Phi": "Φ", "Psi": "Ψ",
    "Omega": "Ω",
    # операторы и отношения
    "sum": "∑", "prod": "∏", "int": "∫", "iint": "∬", "oint": "∮",
    "partial": "∂", "nabla": "∇", "infty": "∞", "pm": "±", "mp": "∓",
    "times": "×", "cdot": "·", "div": "÷", "ast": "∗", "star": "⋆",
    "approx": "≈", "neq": "≠", "ne": "≠", "le": "≤", "leq": "≤", "ge": "≥",
    "geq": "≥", "equiv": "≡", "sim": "∼", "simeq": "≃", "propto": "∝",
    "ll": "≪", "gg": "≫", "subset": "⊂", "subseteq": "⊆", "supset": "⊃",
    "supseteq": "⊇", "in": "∈", "notin": "∉", "ni": "∋", "cup": "∪", "cap": "∩",
    "setminus": "∖", "emptyset": "∅", "varnothing": "∅", "forall": "∀",
    "exists": "∃", "nexists": "∄", "neg": "¬", "land": "∧", "lor": "∨",
    "oplus": "⊕", "otimes": "⊗", "perp": "⊥", "parallel": "∥", "angle": "∠",
    "triangle": "△", "circ": "∘", "bullet": "•", "prime": "′",
    "to": "→", "rightarrow": "→", "leftarrow": "←", "leftrightarrow": "↔",
    "Rightarrow": "⇒", "Leftarrow": "⇐", "Leftrightarrow": "⇔", "mapsto": "↦",
    "implies": "⟹", "iff": "⟺", "uparrow": "↑", "downarrow": "↓",
    "ldots": "…", "cdots": "⋯", "vdots": "⋮", "ddots": "⋱", "dots": "…",
    "aleph": "ℵ", "hbar": "ℏ", "ell": "ℓ", "Re": "ℜ", "Im": "ℑ",
    "degree": "°", "dagger": "†", "ddagger": "‡", "checkmark": "✓",
    # функции остаются словами, но без обратной косой
    "sin": "sin", "cos": "cos", "tan": "tan", "cot": "cot", "sec": "sec",
    "csc": "csc", "arcsin": "arcsin", "arccos": "arccos", "arctan": "arctan",
    "sinh": "sinh", "cosh": "cosh", "tanh": "tanh", "log": "log", "ln": "ln",
    "exp": "exp", "lim": "lim", "max": "max", "min": "min", "sup": "sup",
    "inf": "inf", "det": "det", "dim": "dim", "gcd": "gcd", "mod": "mod",
    "bmod": "mod", "arg": "arg", "ker": "ker", "deg": "deg",
}

#: Пробельные команды: их место занимает обычный пробел (или ничего).
_SPACES = {",": " ", ";": " ", ":": " ", "!": "", " ": " ", "quad": "   ",
           "qquad": "      ", "enspace": " ", "thinspace": " "}

_ESCAPED = {"{": "{", "}": "}", "%": "%", "$": "$", "#": "#", "&": "&", "_": "_",
            "\\": "\n", ",": " ", ";": " ", ":": " ", "!": "", " ": " "}

# ── script tables ───────────────────────────────────────────────────────────

_SUP = dict(zip("0123456789+-=()abcdefghijklmnoprstuvwxyz",
                "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ᵃᵇᶜᵈᵉᶠᵍʰⁱʲᵏˡᵐⁿᵒᵖʳˢᵗᵘᵛʷˣʸᶻ"))
_SUP.update({"n": "ⁿ", "i": "ⁱ"})
_SUB = dict(zip("0123456789+-=()aehijklmnoprstuvx",
                "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₕᵢⱼₖₗₘₙₒₚᵣₛₜᵤᵥₓ"))
#: Знаки, которые в индексе приходится оставлять отдельным символом.
_SUB.update({"β": "ᵦ", "γ": "ᵧ", "ρ": "ᵨ", "φ": "ᵩ", "χ": "ᵪ"})

_STRUCTURAL = set("+-=<>*/")
_GROUP_OPS = set("+- ") | _STRUCTURAL


def _script(text: str, table: dict, marker: str) -> str:
    """`text` в виде степени/индекса, или `^(text)` когда Unicode такого не умеет."""
    if not text:
        return ""
    out: List[str] = []
    for ch in text:
        mapped = table.get(ch)
        if mapped is None:
            return f"{marker}({text})"
        out.append(mapped)
    return "".join(out)


def _needs_parens(text: str) -> bool:
    text = text.strip()
    if not text:
        return False
    # A fraction of two plain atoms reads better without brackets.
    return any(ch in _GROUP_OPS for ch in text) and not (text.startswith("(") and text.endswith(")"))


def _wrap(text: str) -> str:
    text = text.strip()
    return f"({text})" if _needs_parens(text) else text


def _group(expr: str, i: int) -> Tuple[str, int]:
    """Read `{...}` or a single atom starting at `i`; returns (content, next)."""
    n = len(expr)
    while i < n and expr[i] == " ":
        i += 1
    if i >= n:
        return "", i
    if expr[i] != "{":
        return expr[i], i + 1
    depth = 1
    j = i + 1
    while j < n and depth:
        if expr[j] == "{":
            depth += 1
        elif expr[j] == "}":
            depth -= 1
        j += 1
    return expr[i + 1:j - 1], j


def _optional(expr: str, i: int) -> Tuple[str, int]:
    """Read `[n]` (the index of a root), when it is there."""
    while i < len(expr) and expr[i] == " ":
        i += 1
    if i < len(expr) and expr[i] == "[":
        j = expr.find("]", i + 1)
        if j != -1:
            return expr[i + 1:j], j + 1
    return "", i


def _command(name: str, expr: str, i: int) -> Tuple[str, int]:
    """One `\\command` at `i` (already consumed) → (text, next index)."""
    if name in ("frac", "dfrac", "tfrac"):
        num, i = _group(expr, i)
        den, i = _group(expr, i)
        # The parts are formulas too: `\frac{\sqrt{x}}{2}` must not keep a raw `\sqrt`.
        return f"{_wrap(to_unicode(num))}/{_wrap(to_unicode(den))}", i
    if name == "sqrt":
        index, i = _optional(expr, i)
        body, i = _group(expr, i)
        body = to_unicode(body)
        if index:
            return f"{_script(to_unicode(index), _SUP, '^')}√({body})", i
        return f"√({body})", i
    if name in ("text", "mathrm", "mathit", "textbf", "operatorname", "mbox"):
        body, i = _group(expr, i)
        return body, i
    if name == "left" or name == "right":
        # \left( … \right) — the brackets themselves follow and are kept as they are.
        body, i = _group(expr, i) if i < len(expr) and expr[i] == "{" else ("", i)
        if body:
            return body, i
        if i < len(expr):
            return expr[i], i + 1
        return "", i
    if name in ("big", "Big", "bigg", "Bigg", "bigl", "bigr", "Bigl", "Bigr"):
        if i < len(expr):
            return expr[i], i + 1
        return "", i
    if name in _SPACES:
        return _SPACES[name], i
    if name in _SYMBOLS:
        return _SYMBOLS[name], i
    # An unknown command is left readable rather than swallowed: `\foo x`, not `\foox`.
    return f"\\{name} ", i


def to_unicode(expr: str) -> str:
    """One formula, without the `$` delimiters, as a structural Unicode string."""
    out: List[str] = []
    i = 0
    n = len(expr)
    while i < n:
        ch = expr[i]
        if ch == "\\":
            if i + 1 >= n:
                break
            nxt = expr[i + 1]
            if nxt.isalpha():
                j = i + 1
                while j < n and expr[j].isalpha():
                    j += 1
                text, i = _command(expr[i + 1:j], expr, j)
                out.append(text)
            else:
                out.append(_ESCAPED.get(nxt, nxt))
                i += 2
            continue
        if ch in "^_":
            i += 1
            arg, i = _group(expr, i)
            inner = to_unicode(arg)
            out.append(_script(inner, _SUP if ch == "^" else _SUB, ch))
            continue
        if ch == "{":
            arg, i = _group(expr, i)
            out.append(to_unicode(arg))
            continue
        out.append(ch)
        i += 1
    text = "".join(out).strip()
    # `\left( a \right)` leaves the brackets padded; a formula reads better without it.
    return re.sub(r"\s+\)", ")", re.sub(r"\(\s+", "(", text))


# ── finding formulas in Markdown ────────────────────────────────────────────

_INLINE_CODE = re.compile(r"`[^`\n]*`")


_BLOCK_MATH = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
_BRACKET_MATH = re.compile(r"\\\[(.+?)\\\]", re.DOTALL)
_PAREN_MATH = re.compile(r"\\\((.+?)\\\)", re.DOTALL)
_INLINE_MATH = re.compile(r"(?<!\\)\$(?!\$)([^$\n]+?)(?<!\\)\$")


_FRAC_START = re.compile(r"\\(?:d|t)?frac\s*\{")
# Operators that carry limits above and below on their own line. `\lim` is here
# because its subscript hangs underneath it, unlike an ordinary subscript.
# `\b` cannot end this pattern: after `\int` comes `_`, and `_` is a word
# character, so the boundary never matched and no limit was ever stacked.
_BIG_OP = re.compile(r"\\(iint|iiint|oint|int|sum|prod|lim)(?![A-Za-z])")
_BIG_OP_GLYPH = {
    "int": "∫", "iint": "∬", "iiint": "∭", "oint": "∮",
    "sum": "∑", "prod": "∏", "lim": "lim",
}
_MATRIX_START = re.compile(r"\\begin\{([pbv]matrix)\}")
# top, middle, bottom for the left and for the right edge of a matrix
_MATRIX_DELIM = {
    "pmatrix": ("⎛", "⎜", "⎝", "⎞", "⎟", "⎠"),
    "bmatrix": ("⎡", "⎢", "⎣", "⎤", "⎥", "⎦"),
    "vmatrix": ("│", "│", "│", "│", "│", "│"),
}


def _flush(atoms: List[tuple], buf: List[str]) -> None:
    if buf:
        atoms.append(("text", "".join(buf)))
        buf.clear()


def _matrix_atom(expr: str, m) -> Tuple[tuple, int]:
    """`\\begin{pmatrix} a & b \\\\ c & d \\end{pmatrix}` as a grid of rows."""
    kind = m.group(1)
    closer = "\\end{" + kind + "}"
    end = expr.find(closer, m.end())
    if end == -1:
        return (None, m.end())
    body = expr[m.end():end]
    grid: List[List[str]] = []
    for raw_row in re.split(r"\\\\", body):
        cells = [to_unicode(c.strip()) for c in raw_row.split("&")]
        if any(cells):
            grid.append(cells)
    if not grid:
        return (None, end + len(closer))

    cols = max(len(r) for r in grid)
    widths = [0] * cols
    for row in grid:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    top_l, mid_l, bot_l, top_r, mid_r, bot_r = _MATRIX_DELIM[kind]
    lines = []
    for row_i, row in enumerate(grid):
        inner = "  ".join(
            cell.ljust(widths[i]) for i, cell in enumerate(row)
        )
        if len(grid) == 1:
            left, right = top_l, top_r
        elif row_i == 0:
            left, right = top_l, top_r
        elif row_i == len(grid) - 1:
            left, right = bot_l, bot_r
        else:
            left, right = mid_l, mid_r
        lines.append(f"{left}{inner}{right}")
    return (("rows", lines), end + len(closer))


def _atom_block(atom: tuple) -> Tuple[List[str], int]:
    """(rows, baseline index) for one atom, every row padded to one width."""
    kind = atom[0]
    if kind == "text":
        raw = atom[1]
        text = to_unicode(raw)
        # `to_unicode` trims the ends, and that space is load-bearing here: it is
        # what keeps `∫` off the integrand and `=` off a matrix.
        if text and raw[:1].isspace():
            text = " " + text
        if text and raw[-1:].isspace():
            text = text + " "
        rows, base = [text], 0
    elif kind == "frac":
        num = to_unicode(" ".join(atom[1].split()))
        den = to_unicode(" ".join(atom[2].split()))
        width = max(len(num), len(den), 1)
        # A space either side, or the bar fuses with a neighbouring `=`/`+`.
        rows = [
            " " + num.center(width) + " ",
            " " + "─" * width + " ",
            " " + den.center(width) + " ",
        ]
        base = 1
    elif kind == "op":
        # The limits are formulas too: `\lim_{x \to 0}` must not print `\to`.
        glyph = atom[1]
        lower = to_unicode(atom[2])
        upper = to_unicode(atom[3])
        width = max(len(glyph), len(upper), len(lower))
        if upper and lower:
            rows, base = [upper.center(width), glyph.center(width), lower.center(width)], 1
        elif upper:
            rows, base = [upper.center(width), glyph.center(width)], 1
        elif lower:
            rows, base = [glyph.center(width), lower.center(width)], 0
        else:
            rows, base = [glyph], 0
    elif kind == "rows":
        # Round the baseline UP: a two-row matrix reads as growing downwards, so
        # the text beside it belongs on the first row, not the last.
        rows, base = list(atom[1]), (len(atom[1]) - 1) // 2
    else:
        rows, base = [""], 0
    width = max((len(r) for r in rows), default=0)
    return [r.center(width) for r in rows], base


def _atoms(expr: str) -> List[tuple]:
    """Split a formula into atoms that may need more than one row."""
    atoms: List[tuple] = []
    buf: List[str] = []
    i, n = 0, len(expr)
    while i < n:
        m = _FRAC_START.match(expr, i)
        if m:
            num, j = _group(expr, m.end() - 1)
            den, j = _group(expr, j)
            _flush(atoms, buf)
            atoms.append(("frac", num, den))
            i = j
            continue
        m = _MATRIX_START.match(expr, i)
        if m:
            atom, j = _matrix_atom(expr, m)
            if atom is not None:
                _flush(atoms, buf)
                atoms.append(atom)
            i = j
            continue
        m = _BIG_OP.match(expr, i)
        if m:
            glyph = _BIG_OP_GLYPH[m.group(1)]
            lower = upper = ""
            j = m.end()
            # Limits come in either order, braced or as a single character.
            for _ in range(2):
                k = j
                while k < n and expr[k] == " ":
                    k += 1
                if k >= n or expr[k] not in "_^":
                    break
                arg, j = _group(expr, k + 1)
                if expr[k] == "_":
                    lower = arg
                else:
                    upper = arg
            _flush(atoms, buf)
            atoms.append(("op", glyph, " ".join(lower.split()), " ".join(upper.split())))
            i = j
            continue
        buf.append(expr[i])
        i += 1
    _flush(atoms, buf)
    return atoms


def _stack_math(expr: str):
    """Lay a display formula out over several rows.

    `\\frac{a}{b}` came out as `a/b` — right inline, wrong on a line of its own,
    where LaTeX asked for a stacked fraction. Limits of `\\int`/`\\sum` belong
    above and below their sign for the same reason, and a matrix needs a grid.

    Everything is aligned on ONE baseline, so the rest of the formula stays level
    with the middle of a fraction instead of drifting to the top row. Returns
    None when no atom needs a second row, so plain formulas keep their one-line
    form and stay in the flow of the paragraph.
    """
    parts = [_atom_block(atom) for atom in _atoms(expr)]
    if not any(len(rows) > 1 for rows, _ in parts):
        return None
    baseline = max(base for _, base in parts)
    height = max(baseline - base + len(rows) for rows, base in parts)
    out: List[str] = []
    for row in range(height):
        line = ""
        for rows, base in parts:
            idx = row - (baseline - base)
            line += rows[idx] if 0 <= idx < len(rows) else " " * len(rows[0])
        out.append(line.rstrip())
    return "\n".join(out)


def _display(m) -> str:
    """A formula on its own line."""
    expr = " ".join(m.group(1).split())
    stacked = _stack_math(expr)
    if stacked is None:
        return to_unicode(expr)
    # Rich reflows a paragraph, so the rows must travel as a block to keep shape.
    return "```math\n" + stacked + "\n```"


def _convert_prose(chunk: str) -> str:
    """Formulas in one stretch of prose — no fenced code inside it."""
    chunk = _BLOCK_MATH.sub(_display, chunk)
    chunk = _BRACKET_MATH.sub(_display, chunk)
    chunk = _PAREN_MATH.sub(lambda m: to_unicode(" ".join(m.group(1).split())), chunk)
    lines: List[str] = []
    for line in chunk.split("\n"):
        pieces: List[str] = []
        last = 0
        for m in _INLINE_CODE.finditer(line):
            pieces.append(_INLINE_MATH.sub(
                lambda mm: to_unicode(mm.group(1).strip()), line[last:m.start()]))
            pieces.append(m.group(0))
            last = m.end()
        pieces.append(_INLINE_MATH.sub(
            lambda mm: to_unicode(mm.group(1).strip()), line[last:]))
        lines.append("".join(pieces))
    return "\n".join(lines)


_FENCED = re.compile(r"(?ms)^[ \t]*(?:```|~~~).*?(?:^[ \t]*(?:```|~~~)[ \t]*$|\Z)")


def render_math(text: str) -> str:
    """Every `$…$` / `$$…$$` / `\\(…\\)` / `\\[…\\]` outside code, as Unicode.

    Display math is matched over the WHOLE text, not row by row: `$$` on one line
    and the closing `$$` on another is the normal way to write it, and a per-row
    pass could never see the pair — which is why a standard display block reached
    the screen as raw LaTeX.
    """
    if "$" not in text and "\\(" not in text and "\\[" not in text:
        return text
    out: List[str] = []
    last = 0
    for m in _FENCED.finditer(text):
        out.append(_convert_prose(text[last:m.start()]))
        out.append(m.group(0))
        last = m.end()
    out.append(_convert_prose(text[last:]))
    # A block formula asks for its own rows; more than one blank row is noise.
    return re.sub(r"\n{3,}", "\n\n", "".join(out))
