"""The checkpoint list is cached, and the cache never lies.

Drawing one toolbar line used to cost a `for-each-ref` plus a `merge-base` per entry
— ~340 ms measured on a real repository — and the toolbar asks every second. The
cache is what makes that free; these tests are what keep it honest.
"""

import os
import subprocess

import pytest

from src.agent import checkpoints as cp


@pytest.fixture
def repo(tmp_path):
    """A throwaway repository with one commit."""
    def run(*args):
        return subprocess.run(["git", *args], cwd=tmp_path, capture_output=True, text=True)
    run("init", "-q")
    run("config", "user.email", "t@example.com")
    run("config", "user.name", "T")
    (tmp_path / "a.txt").write_text("one\n", encoding="utf-8")
    run("add", "-A")
    run("commit", "-q", "-m", "initial")
    old = os.getcwd()
    os.chdir(tmp_path)
    try:
        yield tmp_path
    finally:
        os.chdir(old)


def test_a_second_call_does_not_run_git_again(repo, monkeypatch):
    calls = []
    real = cp._scan_checkpoints
    monkeypatch.setattr(cp, "_scan_checkpoints", lambda limit: calls.append(limit) or real(limit))

    cp.invalidate_checkpoint_cache()
    first = cp.list_checkpoints(10)
    second = cp.list_checkpoints(10)

    assert calls == [10], "второй вызов должен браться из кэша"
    assert first == second


def test_a_new_checkpoint_shows_up_at_once(repo):
    cp.invalidate_checkpoint_cache()
    assert cp.list_checkpoints(10) == []

    (repo / "a.txt").write_text("two\n", encoding="utf-8")
    cp.create_checkpoint("before: edit")

    labels = [c["label"] for c in cp.list_checkpoints(10)]
    assert labels == ["before: edit"], "кэш не сбросился после создания чекпоинта"


def test_another_repository_never_shares_the_answer(repo, tmp_path_factory):
    cp.invalidate_checkpoint_cache()
    (repo / "a.txt").write_text("two\n", encoding="utf-8")
    cp.create_checkpoint("before: edit")
    assert [c["label"] for c in cp.list_checkpoints(10)] == ["before: edit"]

    other = tmp_path_factory.mktemp("other")
    old = os.getcwd()
    os.chdir(other)
    try:
        subprocess.run(["git", "init", "-q"], cwd=other, capture_output=True)
        # Not the same tree: an empty answer, not the neighbour's list.
        assert cp.list_checkpoints(10) == []
    finally:
        os.chdir(old)


def test_the_caller_cannot_corrupt_the_cache(repo):
    cp.invalidate_checkpoint_cache()
    (repo / "a.txt").write_text("two\n", encoding="utf-8")
    cp.create_checkpoint("before: edit")

    first = cp.list_checkpoints(10)
    first[0]["label"] = "tampered"
    assert cp.list_checkpoints(10)[0]["label"] == "before: edit"


def test_a_smaller_limit_reads_from_a_bigger_cache(repo):
    cp.invalidate_checkpoint_cache()
    (repo / "a.txt").write_text("two\n", encoding="utf-8")
    cp.create_checkpoint("one")
    (repo / "a.txt").write_text("three\n", encoding="utf-8")
    cp.create_checkpoint("two")

    big = cp.list_checkpoints(10)
    small = cp.list_checkpoints(1)
    assert len(big) >= 2
    assert small == big[:1]
