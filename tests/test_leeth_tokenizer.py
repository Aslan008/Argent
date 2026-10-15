"""
tests/test_leeth_tokenizer.py — Verification tests for Leeth multilingual code & text tokenizer.
"""

import tempfile
import pytest
from src.leeth.tokenizer import LeethTokenizer, SPECIAL_TOKENS


def test_special_tokens_initialization():
    tok = LeethTokenizer()
    for st in SPECIAL_TOKENS:
        assert st in tok.vocab
        assert tok.vocab[st] == tok.encode(st, add_special_tokens=False)[0]


def test_tokenizer_training_and_serialization():
    corpus = [
        "using System; namespace App { class Program { static void Main() {} } }",
        "#include <iostream> int main() { std::cout << 'Hello'; return 0; }",
        "fn main() { println!('Rust test running'); }",
        "def test_agent_execution(): assert agent.run() is True",
        "const path = require('path'); export default function run() {}",
        "прочитай конфигурационный файл config.json и выведи статус",
        "запусти тесты pytest tests/test_agent.py --verbose",
        "ошибка компиляции CS0246: тип или пространство имен не найдено",
        "error[E0382]: use of moved value: `agent`",
    ]

    tok = LeethTokenizer()
    tok.train_from_corpus(corpus, target_vocab_size=300, min_freq=1)
    assert tok.vocab_size > len(SPECIAL_TOKENS)

    sample = "прочитай файл config.json и запусти cargo test"
    encoded = tok.encode(sample, add_special_tokens=True)
    assert encoded[0] == tok.cls_token_id
    assert encoded[-1] == tok.sep_token_id

    decoded = tok.decode(encoded, skip_special_tokens=True)
    assert "прочитай" in decoded
    assert "cargo" in decoded
    assert "config.json" in decoded

    # Serialization test
    with tempfile.TemporaryDirectory() as tmpdir:
        tok.save(tmpdir)
        loaded = LeethTokenizer.load(tmpdir)
        assert loaded.vocab_size == tok.vocab_size
        assert loaded.merges == tok.merges
        assert loaded.encode(sample) == encoded


def test_multilingual_code_roundtrip():
    corpus = [
        "namespace Argent.Core { public async Task<bool> RunAsync() { return await Task.FromResult(true); } }",
        "#include <vector> std::unique_ptr<Agent> agent = std::make_unique<Agent>();",
        "pub fn execute_command(cmd: &str) -> Result<String, AgentError> { Ok(cmd.to_string()) }",
        "def test_run(): pass",
        "export const handler = async (req: Request) => { return res.json({ ok: true }); };",
        "прочитай файл src/main.rs и запусти тесты cargo test",
    ]

    tok = LeethTokenizer()
    tok.train_from_corpus(corpus, target_vocab_size=500, min_freq=1)

    for text in corpus:
        encoded = tok.encode(text, add_special_tokens=False)
        decoded = tok.decode(encoded, skip_special_tokens=True)
        # Verify all key identifiers and words survived round-trip
        for word in ["namespace", "RunAsync", "vector", "std", "execute_command", "cargo", "прочитай"]:
            if word in text:
                assert word in decoded
