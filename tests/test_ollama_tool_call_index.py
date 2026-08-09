"""Ollama emits tool calls COMPLETE, not as deltas — so the index must be
global across the response.

Observed with GLM 5.2 through Ollama cloud, in agent.log:

    Failed to parse tool args for read_fileread_file
    Failed to parse tool args for run_commandread_fileread_file

enumerate() restarted at 0 in every chunk, and the consumer appends deltas per
index, so every call landed in slot 0: the NAMES concatenated and so did the
JSON arguments, which then failed to parse and were replaced with {}. The tool
ran with no arguments and the model was told nothing. Any model emitting more
than one tool call per turn hit this.
"""

import json

import pytest

from providers import OllamaProvider


class _FakeOllama:
    """Streams the chunk shape Ollama actually produces: one complete tool call
    per chunk, arguments already decoded into a dict."""

    def __init__(self, chunks):
        self._chunks = chunks

    def chat(self, **kwargs):
        return iter(self._chunks)


def _call(name, **args):
    return {"function": {"name": name, "arguments": args}}


def _provider(chunks):
    p = OllamaProvider.__new__(OllamaProvider)
    p._ollama = _FakeOllama(chunks)
    return p


def _accumulate(deltas):
    """The consumer's accumulation, mirrored from agent.py."""
    slots = []
    for d in deltas:
        while len(slots) <= d["index"]:
            slots.append({"name": "", "arguments": ""})
        slots[d["index"]]["name"] += d["function_name_delta"]
        slots[d["index"]]["arguments"] += d["function_arguments_delta"]
    return slots


def _deltas(provider):
    out = []
    for chunk in provider.stream_chat(model="m", messages=[]):
        out.extend(chunk.get("tool_call_deltas", []))
    return out


class TestIndexing:
    def test_two_calls_in_separate_chunks_stay_separate(self):
        """The reported failure: read_file + read_file -> "read_fileread_file"."""
        p = _provider([
            {"message": {"tool_calls": [_call("read_file", file_path="a.py")]}},
            {"message": {"tool_calls": [_call("read_file", file_path="b.py")]}},
        ])
        slots = _accumulate(_deltas(p))
        assert [s["name"] for s in slots] == ["read_file", "read_file"]
        assert [json.loads(s["arguments"])["file_path"] for s in slots] == ["a.py", "b.py"]

    def test_three_different_tools(self):
        """run_command + read_file + read_file -> "run_commandread_fileread_file"."""
        p = _provider([
            {"message": {"tool_calls": [_call("run_command", command="ls")]}},
            {"message": {"tool_calls": [_call("read_file", file_path="a.py")]}},
            {"message": {"tool_calls": [_call("read_file", file_path="b.py")]}},
        ])
        assert [s["name"] for s in _accumulate(_deltas(p))] == [
            "run_command", "read_file", "read_file"]

    def test_several_calls_in_one_chunk_are_also_distinct(self):
        p = _provider([
            {"message": {"tool_calls": [_call("read_file", file_path="a.py"),
                                        _call("read_file", file_path="b.py")]}},
        ])
        assert [d["index"] for d in _deltas(p)] == [0, 1]

    def test_the_arguments_remain_parseable(self):
        """Concatenated JSON is what produced "Extra data: line 1 column 95"."""
        p = _provider([
            {"message": {"tool_calls": [_call("read_file", file_path="a.py")]}},
            {"message": {"tool_calls": [_call("read_file", file_path="b.py")]}},
        ])
        for slot in _accumulate(_deltas(p)):
            json.loads(slot["arguments"])          # must not raise

    def test_content_chunks_between_calls_do_not_shift_the_index(self):
        p = _provider([
            {"message": {"tool_calls": [_call("read_file", file_path="a.py")]}},
            {"message": {"content": "думаю…"}},
            {"message": {"tool_calls": [_call("read_file", file_path="b.py")]}},
        ])
        assert [d["index"] for d in _deltas(p)] == [0, 1]

    def test_a_single_call_is_unaffected(self):
        p = _provider([{"message": {"tool_calls": [_call("read_file", file_path="a.py")]}}])
        assert [d["index"] for d in _deltas(p)] == [0]

    def test_no_tool_calls(self):
        p = _provider([{"message": {"content": "просто ответ"}}])
        assert _deltas(p) == []


class TestTimeout:
    def test_the_client_has_a_read_timeout(self):
        """A dropped connection produced 948 seconds of spinner ended by Ctrl+C;
        nothing distinguishes "still working" from "the socket is gone" except
        silence, and forever is the wrong amount of silence to accept."""
        p = OllamaProvider()
        timeout = getattr(getattr(p._ollama, "_client", None), "timeout", None)
        assert timeout is not None and timeout.read and timeout.read >= 30

    def test_it_is_a_read_timeout_not_a_total_one(self, monkeypatch):
        """It measures the GAP between tokens, so a slow but living generation
        is untouched — a total limit would kill long legitimate answers."""
        p = OllamaProvider()
        timeout = p._ollama._client.timeout
        assert timeout.connect < timeout.read
