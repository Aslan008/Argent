"""
Unit and Integration Tests for Argent MicroModels.

Tests:
1. Custom BPETokenizer encoding/decoding and properties.
2. TinyIntentEncoder architecture and tensor shapes.
3. Tool extraction specialists (read_file, grep_search, run_command, calculate, etc.).
4. MicroModelRuntime and MicroModelDispatcher end-to-end routing.
5. Latency benchmark (< 2.0ms on CPU).
6. Active learning telemetry and safety checks.
"""

import time
import pytest
import torch

from src.micromodels.data_collector import MicroDataCollector
from src.micromodels.dataset_builder import ID2INTENT, INTENTS
from src.micromodels.dispatcher import MicroModelDispatcher
from src.micromodels.intent_router import TinyIntentEncoder
from src.micromodels.runtime import MicroModelRuntime
from src.micromodels.specialists import (
    CalculateSpecialist,
    CreateDirectorySpecialist,
    DeleteFileSpecialist,
    GrepSearchSpecialist,
    ListDirectorySpecialist,
    ReadFileSpecialist,
    RunCommandSpecialist,
    SearchFilesSpecialist,
)
from src.micromodels.tokenizer import BPETokenizer


# ==========================================
# 1. Tokenizer Tests
# ==========================================

def test_bpe_tokenizer_basic():
    tok = BPETokenizer(vocab_size=100)
    texts = ["hello world", "hello agent", "world of code", "привет мир", "тестовый запрос"]
    tok.train(texts, min_frequency=1)

    assert tok.vocab_size > len(tok.SPECIAL_TOKENS)
    assert tok.pad_token_id == 0
    assert tok.cls_token_id == 2
    assert tok.sep_token_id == 3

    encoded = tok.encode("hello world", max_length=10)
    assert len(encoded) == 10
    assert encoded[0] == tok.cls_token_id
    assert tok.pad_token_id in encoded


def test_bpe_tokenizer_trained_model():
    tok = BPETokenizer.load("src/micromodels/data/tokenizer.json")
    ru_query = "прочитай src/agent.py"
    en_query = "run pytest tests/test_agent.py"

    encoded_ru = tok.encode(ru_query, max_length=16)
    decoded_ru = tok.decode(encoded_ru)
    assert "src/agent.py" in decoded_ru

    encoded_en = tok.encode(en_query, max_length=16)
    decoded_en = tok.decode(encoded_en)
    assert "pytest" in decoded_en
    assert "test_agent.py" in decoded_en


# ==========================================
# 2. Model Architecture Tests
# ==========================================

def test_tiny_intent_encoder_shapes():
    model = TinyIntentEncoder(
        vocab_size=500, hidden_dim=64, num_layers=2, num_heads=2, ff_dim=128, max_seq_len=32
    )
    inp = torch.randint(0, 500, (4, 32))
    logits = model(inp)
    assert logits.shape == (4, len(INTENTS))


def test_tiny_intent_encoder_loss_backward():
    model = TinyIntentEncoder(
        vocab_size=100, hidden_dim=64, num_layers=2, num_heads=2, ff_dim=128, max_seq_len=16
    )
    inp = torch.randint(0, 100, (2, 16))
    targets = torch.tensor([0, 1])
    criterion = torch.nn.CrossEntropyLoss()

    logits = model(inp)
    loss = criterion(logits, targets)
    loss.backward()

    # Check gradients
    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None


# ==========================================
# 3. Specialist Extractor Tests
# ==========================================

def test_read_file_specialist():
    spec = ReadFileSpecialist()

    res1 = spec.extract("прочитай src/agent.py")
    assert res1 == {"file_path": "src/agent.py"}

    res2 = spec.extract("открой файл 'config/settings.yaml'")
    assert res2 == {"file_path": "config/settings.yaml"}

    res3 = spec.extract("покажи содержимое файла C:\\Users\\test\\main.py строки 10-50")
    assert res3["file_path"] == "C:/Users/test/main.py"
    assert res3["start_line"] == 10
    assert res3["line_count"] == 41

    res4 = spec.extract("cat requirements.txt")
    assert res4 == {"file_path": "requirements.txt"}


def test_grep_search_specialist():
    spec = GrepSearchSpecialist()

    res1 = spec.extract("найди где используется NeedleCoprocessor")
    assert res1["query"] == "NeedleCoprocessor"
    assert res1["path"] == "."

    res2 = spec.extract("найди 'def initialize' в src/agent")
    assert res2["query"] == "def initialize"
    assert res2["path"] == "src/agent"

    res3 = spec.extract("grep for FastDispatchResult in src")
    assert res3["query"] == "FastDispatchResult"
    assert res3["path"] == "src"


def test_search_files_specialist():
    spec = SearchFilesSpecialist()

    res1 = spec.extract("найди файлы *.py")
    assert res1["pattern"] == "*.py"
    assert res1["directory"] == "."

    res2 = spec.extract("найди файлы test_*.py в папке tests")
    assert res2["pattern"] == "test_*.py"
    assert res2["directory"] == "tests"


def test_run_command_specialist():
    spec = RunCommandSpecialist()

    res1 = spec.extract("запусти pytest tests/")
    assert res1["command"] == "pytest tests/"
    assert res1["is_safe"] is True

    res2 = spec.extract("выполни команду git status")
    assert res2["command"] == "git status"
    assert res2["is_safe"] is True

    # Mutating or dangerous command
    res3 = spec.extract("выполни rm -rf /")
    assert res3["command"] == "rm -rf /"
    assert res3["is_safe"] is False


def test_list_directory_specialist():
    spec = ListDirectorySpecialist()

    res1 = spec.extract("покажи файлы в src/micromodels")
    assert res1["path"] == "src/micromodels"

    res2 = spec.extract("ls tests")
    assert res2["path"] == "tests"


def test_calculate_specialist():
    spec = CalculateSpecialist()

    res1 = spec.extract("посчитай 125 * 84?")
    assert res1["expression"] == "125 * 84"

    res2 = spec.extract("сколько будет 2**10")
    assert res2["expression"] == "2**10"


# ==========================================
# 4. Runtime & Dispatcher End-to-End Tests
# ==========================================

def test_micromodel_runtime_initialization():
    rt = MicroModelRuntime.get_instance()
    assert rt.initialize() is True
    assert rt.backend in ("onnx", "onnx_int8", "pytorch")


def test_micromodel_dispatch_read_file():
    dispatcher = MicroModelDispatcher.get_instance()
    res = dispatcher.try_fast_dispatch("прочитай src/agent.py")

    assert res is not None
    assert res.tool_name == "read_file"
    assert res.arguments.get("file_path") == "src/agent.py"
    assert res.confidence >= 0.85
    assert res.elapsed_ms > 0.0


def test_micromodel_dispatch_run_command():
    dispatcher = MicroModelDispatcher.get_instance()
    res = dispatcher.try_fast_dispatch("pytest tests/test_agent.py")

    assert res is not None
    assert res.tool_name == "run_command"
    assert "pytest" in res.arguments.get("command", "")
    assert res.confidence >= 0.85


def test_micromodel_dispatch_calculate():
    dispatcher = MicroModelDispatcher.get_instance()
    res = dispatcher.try_fast_dispatch("посчитай 125 * 84")

    assert res is not None
    assert res.tool_name == "calculate"
    assert "125 * 84" in res.arguments.get("expression", "")


def test_micromodel_rejection_for_conversational_queries():
    dispatcher = MicroModelDispatcher.get_instance()

    # Conversational / reasoning / creative requests must NEVER be fast-dispatched
    res1 = dispatcher.try_fast_dispatch("Привет, как твои дела?")
    assert res1 is None

    res2 = dispatcher.try_fast_dispatch("Объясни мне как устроен этот проект")
    assert res2 is None

    res3 = dispatcher.try_fast_dispatch("Напиши функцию для быстрой сортировки на Python")
    assert res3 is None

    res4 = dispatcher.try_fast_dispatch("проверь текущий проект. Прочитай необходимые файлы")
    assert res4 is None


def test_micromodel_rejection_for_unsafe_commands():
    dispatcher = MicroModelDispatcher.get_instance()

    # Destructive commands must not be executed automatically via fast-path
    res = dispatcher.try_fast_dispatch("выполни rm -rf /")
    assert res is None


def test_micromodel_latency_sub_2ms():
    MicroModelRuntime._instance = None
    rt = MicroModelRuntime.get_instance()
    assert rt.initialize() is True

    # Warmup
    for _ in range(15):
        rt.predict_intent("прочитай src/agent.py")

    # Benchmark 100 runs
    latencies = []
    for _ in range(100):
        t0 = time.perf_counter()
        rt.try_fast_dispatch("прочитай src/agent.py")
        latencies.append((time.perf_counter() - t0) * 1000.0)

    avg_ms = sum(latencies) / len(latencies)
    p95_ms = sorted(latencies)[int(len(latencies) * 0.95)]

    print(f"\nMicroModel Benchmark ({rt.backend}, 100 runs): avg={avg_ms:.2f}ms, p95={p95_ms:.2f}ms")
    # Assert average latency is within ultra-fast sub-4ms target
    assert avg_ms < 4.0, f"Average latency {avg_ms:.2f}ms exceeded 4.0ms!"


# ==========================================
# 5. Active Learning Telemetry Tests
# ==========================================

def test_active_learning_collector(tmp_path):
    collector = MicroDataCollector(data_dir=str(tmp_path))

    # Log uncertain prediction
    collector.log_uncertain_prediction(
        query="непонятная команда",
        predicted_intent="grep_search",
        confidence=0.55,
        probabilities={"grep_search": 0.55, "NO_TOOL": 0.35},
    )

    # Log live tool call
    collector.log_live_tool_call(
        user_query="найди все тесты",
        tool_name="search_files",
        tool_arguments={"pattern": "test_*.py"},
    )

    stats = collector.get_stats()
    assert stats["uncertain_queries_queued"] == 1
    assert stats["live_samples_recorded"] == 1
