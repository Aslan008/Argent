"""Формулы: LaTeX → структурный Unicode, и ничего не теряется.

Rich рисует Markdown, но формул не знает: `$\\frac{a}{b}$` доходило до экрана
сырым LaTeX, а `\\f` в `\\frac` местами съедался как управляющий символ. Здесь
проверяется и преобразование, и то, что код, инлайн-код и неизвестные команды
не портятся.
"""

import pytest

from src.bridge.latex_unicode import render_math, to_unicode


@pytest.mark.parametrize("latex, want", [
    ("E = mc^2", "E = mc²"),
    ("x^{n+1}", "xⁿ⁺¹"),
    ("x_i", "xᵢ"),
    ("a_{i,j}", "a_(i,j)"),          # запятой в нижнем индексе Unicode нет
    (r"\alpha + \beta = \gamma", "α + β = γ"),
    (r"\Delta x \to 0", "Δ x → 0"),
    (r"\frac{a}{b}", "a/b"),
    (r"\frac{x+1}{2}", "(x+1)/2"),
    (r"\frac{1}{N}", "1/N"),
    (r"\sqrt{x}", "√(x)"),
    (r"\sqrt[3]{x}", "³√(x)"),
    (r"\sum_{i=1}^{n} x_i", "∑ᵢ₌₁ⁿ xᵢ"),
    (r"\int_0^1 f(x)\,dx", "∫₀¹ f(x) dx"),
    (r"\lim_{n \to \infty} a_n", "lim_(n → ∞) aₙ"),
    (r"\text{если } x > 0", "если  x > 0"),
    (r"\left( a + b \right)", "(a + b)"),
    (r"a \le b \ne c \approx d", "a ≤ b ≠ c ≈ d"),
    (r"\foo{x}", r"\foo x"),
])
def test_formulas_become_structure(latex, want):
    assert to_unicode(latex) == want


def test_delimiters_are_removed_and_code_is_untouched():
    assert render_math("Энергия $E = mc^2$ растёт.") == "Энергия E = mc² растёт."
    assert render_math("Формула: $\\alpha$.") == "Формула: α."
    # Блочная формула не добавляет лишних пустых строк.
    assert render_math("Итог:\n$$\\sum_{i=1}^{n} x_i$$\nконец") == "Итог:\n∑ᵢ₌₁ⁿ xᵢ\nконец"


def test_inline_code_and_fences_are_not_touched():
    assert render_math("`$x^2$`") == "`$x^2$`"
    fenced = "```python\nprice = '$5'\n```"
    assert render_math(fenced) == fenced
    mixed = "текст $x^2$ и `$x^2$` и $y_1$"
    assert render_math(mixed) == "текст x² и `$x^2$` и y₁"


def test_a_broken_formula_is_not_destroyed():
    # Нет закрывающего $: это не формула, и текст остаётся как был.
    assert render_math("стоит $5 и всё") == "стоит $5 и всё"
    # Экранированный доллар — просто доллар.
    assert render_math(r"цена \$5") == r"цена \$5"


def test_plain_text_is_returned_as_is():
    for text in ["", "просто текст", "a > b и c < d", "```\ncode\n```"]:
        assert render_math(text) == text


def test_the_frac_control_character_does_not_eat_the_formula():
    # `\f` — form feed: до этой правки Rich показывал `rac{1}{N}`.
    out = render_math(r"$\sigma = \sqrt{\frac{1}{N}\sum (x_i - \mu)^2}$")
    assert "rac" not in out
    assert "σ" in out and "√" in out and "∑" in out and "μ" in out
