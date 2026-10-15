"""
tests/test_lmtrust_chained_commands_deep.py
LMTrust Deep Blind Spot & Neuro-Symbolic Audit for Chained / Compound Shell Commands:
- Layer 0: Smoke & Determinism
- Layer 1: Contract & Parameter Guarantees (D7)
- Layer 2: Boundary (Empty tokens, semicolons only, mixed operators) (D1, D3)
- Layer 2b: Quoting & Escape Preservation (D3, D8)
- Layer 4: Adversarial Injections, Unclosed Quotes & Subcommand Hiding (D8)
- Layer 4b: Session Grant Key Security & Non-Leakage (D8)
- Layer 7: Compositional Multi-Chains & Redirects (D4, D6)
- Layer 10: Mathematical Invariant of Risk Monotonicity (D10)
"""

import pytest

from approval import (
    assess_command_risk,
    is_destructive_command,
    command_grant_key,
    _split_unquoted_chain_commands,
    _has_unquoted_chain_operators,
    _has_unclosed_quotes,
)


class TestL0Smoke:
    """Layer 0: Smoke & determinism for split and assessment primitives."""

    def test_split_unquoted_chain_commands_returns_list(self):
        res = _split_unquoted_chain_commands("echo 1; echo 2")
        assert isinstance(res, list)
        assert len(res) == 2

    def test_split_deterministic_on_repeated_runs(self):
        cmd = "python -u -c \"import a\" ; python -u -c \"import b\""
        res1 = _split_unquoted_chain_commands(cmd)
        res2 = _split_unquoted_chain_commands(cmd)
        assert res1 == res2


class TestL1Contract:
    """Layer 1: Contract guarantees: no empty tokens in output, stripped strings."""

    def test_split_never_contains_empty_strings(self):
        for cmd in (";;;", "   ;   ;   ", "echo 1;;echo 2;", "\n\ncmd1;\ncmd2\n"):
            parts = _split_unquoted_chain_commands(cmd)
            for p in parts:
                assert p.strip() != "", f"Found empty or whitespace part in {parts}"
                assert p == p.strip(), f"Part not stripped: '{p}'"


class TestL2Boundary:
    """Layer 2: Boundary values: whitespace, consecutive delimiters, single tokens."""

    @pytest.mark.parametrize("empty_input", ["", "   ", "\t", "\n", ";", ";;;", "&&", "||"])
    def test_empty_and_delimiter_only_inputs(self, empty_input):
        assert _split_unquoted_chain_commands(empty_input) == []

    def test_single_command_without_operators(self):
        cmd = "python app.py --arg1 val1"
        assert _split_unquoted_chain_commands(cmd) == [cmd]
        assert not _has_unquoted_chain_operators(cmd)

    def test_trailing_and_leading_delimiters(self):
        cmd = "; echo first ; echo second ;"
        parts = _split_unquoted_chain_commands(cmd)
        assert parts == ["echo first", "echo second"]
        assert _has_unquoted_chain_operators(cmd)

    def test_newline_delimiters_split_correctly(self):
        cmd = "git status\npython -u -c \"print('ok')\"\nls -la"
        parts = _split_unquoted_chain_commands(cmd)
        assert parts == ["git status", 'python -u -c "print(\'ok\')"', "ls -la"]


class TestL2bQuotingAndEscaping:
    """Layer 2b: Quoting edge cases: single/double nested, escaped quotes, semicolons in strings."""

    def test_semicolon_in_double_quotes_not_split(self):
        cmd = 'python -c "a = \';\'; b = 123; print(a, b)"'
        parts = _split_unquoted_chain_commands(cmd)
        assert len(parts) == 1
        assert parts[0] == cmd
        assert not _has_unquoted_chain_operators(cmd)

    def test_semicolon_in_single_quotes_not_split(self):
        cmd = "node -e 'console.log(\";;;\");'"
        parts = _split_unquoted_chain_commands(cmd)
        assert len(parts) == 1
        assert parts[0] == cmd
        assert not _has_unquoted_chain_operators(cmd)

    def test_escaped_quotes_handled_correctly(self):
        cmd = 'echo "escaped \\" quote ; still inside" ; echo outside'
        parts = _split_unquoted_chain_commands(cmd)
        assert len(parts) == 2
        assert parts[1] == "echo outside"

    def test_nested_quotes_mixed(self):
        cmd = '''python -c "s = 'nested \\"double\\" in single'; print(s)" ; echo done'''
        parts = _split_unquoted_chain_commands(cmd)
        assert len(parts) == 2
        assert parts[1] == "echo done"


class TestL4AdversarialSecurity:
    """Layer 4: Adversarial attacks: hiding malicious commands in benign chains."""

    def test_destructive_hidden_at_end_of_benign_chain(self):
        cmd = "git status ; echo hello ; python -u -c \"print('ok')\" ; rm notes.txt"
        level, reasons = assess_command_risk(cmd)
        assert level == "warn"
        assert is_destructive_command(cmd) is True

    def test_catastrophic_hidden_in_middle_of_chain(self):
        cmd = "echo 1 && python -u -c \"print(2)\" && rm -rf / && echo 3"
        level, reasons = assess_command_risk(cmd)
        assert level == "block"
        assert is_destructive_command(cmd) is True

    def test_dangerous_inline_python_in_chain(self):
        cmd = 'python -u -c "print(1)" ; python -c "import os; os.remove(\'db.sqlite\')"'
        level, reasons = assess_command_risk(cmd)
        assert level == "warn"
        assert any("runs inline interpreter" in r or "deletes" in r for r in reasons)
        assert is_destructive_command(cmd) is True

    def test_unclosed_quotes_flagged_as_warn(self):
        # Benign command with unclosed quote -> flagged as warn
        cmd_unclosed_benign = 'echo "hello incomplete prompt'
        assert _has_unclosed_quotes(cmd_unclosed_benign) is True
        level, reasons = assess_command_risk(cmd_unclosed_benign)
        assert level == "warn"
        assert "unclosed quotation mark" in reasons[0]

        # Adversarial attempt to hide rm -rf / inside unclosed quote -> blocked at catastrophic scanner
        cmd_unclosed_adversarial = 'echo "hello ; rm -rf /'
        assert _has_unclosed_quotes(cmd_unclosed_adversarial) is True
        level_adv, reasons_adv = assess_command_risk(cmd_unclosed_adversarial)
        assert level_adv == "block"
        assert any("recursive force-delete" in r for r in reasons_adv)

    def test_redirections_not_confused_with_background_ampersand(self):
        cmd = "python script.py 2>&1"
        assert not _has_unquoted_chain_operators(cmd)
        parts = _split_unquoted_chain_commands(cmd)
        assert len(parts) == 1


class TestL4bSessionGrantKeySecurity:
    """Layer 4b: Security invariant: compound commands MUST NEVER receive session grants."""

    @pytest.mark.parametrize("compound_cmd", [
        "echo hello ; rm -rf /",
        "python safe.py && del secret.txt",
        "ls | grep foo",
        "cmd1 ;",
        "; cmd2",
        ";;;",
        "cmd1\ncmd2",
    ])
    def test_compound_commands_never_grant_key(self, compound_cmd):
        key = command_grant_key(compound_cmd)
        assert key is None, f"Compound command '{compound_cmd}' leaked grant key: '{key}'"


class TestL7CompositionalChains:
    """Layer 7: Composition of multiple tools, PowerShell call operators, and multi-step verification."""

    def test_powershell_call_operator_in_chain(self):
        cmd = '& "C:\\Program Files\\Python314\\python.exe" -u -c "print(1)" ; echo done'
        parts = _split_unquoted_chain_commands(cmd)
        assert len(parts) == 2
        level, reasons = assess_command_risk(cmd)
        assert level == "safe"
        assert reasons == []

    def test_ten_chained_safe_commands_all_safe(self):
        chain = " && ".join([f"python -u -c \"import sys; print({i})\"" for i in range(10)])
        parts = _split_unquoted_chain_commands(chain)
        assert len(parts) == 10
        level, reasons = assess_command_risk(chain)
        assert level == "safe"
        assert reasons == []
        assert is_destructive_command(chain) is False


class TestL10RiskMonotonicityInvariant:
    """Layer 10: Formal proof that risk(chain) == max(risk(sub) for sub in chain)."""

    def test_monotonicity_across_safe_warn_block_combinations(self):
        # 1. safe + safe == safe
        cmd_ss = 'echo 1 ; python -u -c "print(2)"'
        assert assess_command_risk(cmd_ss)[0] == "safe"

        # 2. safe + warn == warn
        cmd_sw = 'echo 1 ; rm notes.txt'
        assert assess_command_risk(cmd_sw)[0] == "warn"

        # 3. safe + block == block
        cmd_sb = 'echo 1 ; format C:'
        assert assess_command_risk(cmd_sb)[0] == "block"

        # 4. warn + block == block
        cmd_wb = 'rm notes.txt ; format C:'
        assert assess_command_risk(cmd_wb)[0] == "block"

        # 5. block + safe == block
        cmd_bs = 'format C: ; echo 1'
        assert assess_command_risk(cmd_bs)[0] == "block"
