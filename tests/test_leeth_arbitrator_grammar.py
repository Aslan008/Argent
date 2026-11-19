"""tests/test_leeth_arbitrator_grammar.py — солвер должен судить ту же грамматику, что исполняет инструмент.

Почему это существует: на живом пути диспетчера из 153 спорных случаев солвер был прав в 124 (81 %), а все 29
ошибок — один класс: арифметическая проверка принимала только цифры и знаки, поэтому 'sqrt(144)' объявлялось
противоречием, хотя calculate отвечает 12. Точность солвера, который судит чужой язык, — фикция: он «ловит»
всё, что содержит букву, и мешает тем сильнее, чем полезнее инструмент.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from src.leeth.arbitrator import CDCLArbitrator


@pytest.fixture(scope="module")
def arb():
    return CDCLArbitrator()


def _verdict(arb, param, query=None):
    # the query must contain the param: "param_in_query" is a separate, legitimate constraint
    return arb.arbitrate(tool="calculate", need_tool=True, param_text=param,
                         query=query if query is not None else "посчитай " + param.strip(),
                         log_shadow=False)


@pytest.mark.parametrize("param", [
    "128 * 64",
    "sqrt(144)",          # the tool answers 12 — 29 live conflicts came from exactly this
    " sqrt(144)",         # leading space must not change the verdict
    "20 % 3",
    "2^10",               # the tool normalises ^ to **
    "2x + 3",             # the tool normalises 2x to 2*x
    "1/0",                # syntactically fine: a runtime failure, not a malformed proposal
])
def test_arithmetic_the_tool_would_accept_is_not_a_conflict(arb, param):
    res = _verdict(arb, param)
    assert res.error is None
    assert res.sat, f"calculate({param!r}) заблокирован, хотя инструмент его исполняет: core={res.core}"


@pytest.mark.parametrize("param", [
    "sqrt(144",           # unbalanced bracket — 85 live conflicts, all genuine
    "корень из 144",      # prose, no numbers-with-operators structure
    "shutdown /s /t 0",
    "os.system('dir')",   # an attribute call: nothing to do with arithmetic
    "config.py",
])
def test_malformed_or_foreign_params_stay_a_conflict(arb, param):
    res = _verdict(arb, param)
    assert res.error is None
    assert not res.sat, f"calculate({param!r}) прошёл проверку, хотя это не арифметика"
