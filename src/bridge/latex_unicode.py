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

_FENCE = re.compile(r"^\s*(```|~~~)")
_INLINE_CODE = re.compile(r"`[^`\n]*`")


def _outside_code(text: str, fn) -> str:
    """Apply `fn` to the parts of `text` that are not code."""
    lines = text.split("\n")
    out: List[str] = []
    in_fence = False
    for line in lines:
        if _FENCE.match(line):
            in_fence = not in_fence
            out.append(line)
            continue
        if in_fence:
            out.append(line)
            continue
        # Протекть инлайн-код: формулы внутри `...` не трогаем.
        pieces: List[str] = []
        last = 0
        for m in _INLINE_CODE.finditer(line):
            pieces.append(fn(line[last:m.start()]))
            pieces.append(m.group(0))
            last = m.end()
        pieces.append(fn(line[last:]))
        out.append("".join(pieces))
    return "\n".join(out)


_BLOCK_MATH = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
_BRACKET_MATH = re.compile(r"\\\[(.+?)\\\]", re.DOTALL)
_PAREN_MATH = re.compile(r"\\\((.+?)\\\)", re.DOTALL)
_INLINE_MATH = re.compile(r"(?<!\\)\$(?!\$)([^$\n]+?)(?<!\\)\$")


def _convert_blocks(text: str) -> str:
    def block(m) -> str:
        # No extra blank rows: the Markdown around the formula already has its own.
        return to_unicode(" ".join(m.group(1).split()))

    text = _BLOCK_MATH.sub(block, text)
    text = _BRACKET_MATH.sub(block, text)
    text = _PAREN_MATH.sub(lambda m: to_unicode(" ".join(m.group(1).split())), text)
    text = _INLINE_MATH.sub(lambda m: to_unicode(m.group(1).strip()), text)
    return text


def render_math(text: str) -> str:
    """Every `$…$` / `$$…$$` / `\\(…\\)` / `\\[…\\]` outside code, as Unicode."""
    if "$" not in text and "\\(" not in text and "\\[" not in text:
        return text
    out = _outside_code(text, _convert_blocks)
    # A block formula asks for its own rows; more than one blank row is noise.
    return re.sub(r"\n{3,}", "\n\n", out)
