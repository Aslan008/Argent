"""/review: collecting the diff, asking the reviewer, reading its answer back."""

import json
import subprocess

import pytest

from src.agent import review
from src.agent.review import (
    Finding, ReviewError, build_task, collect_diff, drop_unknown_files, parse_findings, run_review, split_target,
)


def git(repo, *args):
    done = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    return done.stdout


@pytest.fixture
def repo(tmp_path):
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.email", "t@argent.local")
    git(tmp_path, "config", "user.name", "T")
    (tmp_path / "app.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    (tmp_path / "util.py").write_text("X = 1\n", encoding="utf-8")
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-q", "-m", "initial")
    return tmp_path


class TestCollectDiff:
    def test_uncommitted_means_staged_unstaged_and_new_files(self, repo):
        (repo / "app.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
        (repo / "util.py").write_text("X = 2\n", encoding="utf-8")
        git(repo, "add", "util.py")
        (repo / "fresh.py").write_text("print('new')\n", encoding="utf-8")
        bundle = collect_diff("", repo)
        assert set(bundle.files) == {"app.py", "util.py", "fresh.py"}
        assert "return a - b" in bundle.text and "+print('new')" in bundle.text and "+X = 2" in bundle.text

    def test_staged_only(self, repo):
        (repo / "app.py").write_text("changed\n", encoding="utf-8")
        (repo / "util.py").write_text("X = 3\n", encoding="utf-8")
        git(repo, "add", "util.py")
        bundle = collect_diff("staged", repo)
        assert bundle.files == ["util.py"] and "app.py" not in bundle.text

    def test_a_commit_and_a_range(self, repo):
        (repo / "util.py").write_text("X = 9\n", encoding="utf-8")
        git(repo, "commit", "-qam", "bump x")
        assert collect_diff("HEAD", repo).files == ["util.py"]
        assert "bump x" in collect_diff("HEAD", repo).text
        assert collect_diff("HEAD~1..HEAD", repo).files == ["util.py"]

    def test_a_path_limits_the_review_to_it_including_new_files_under_it(self, repo):
        (repo / "pkg").mkdir()
        (repo / "pkg" / "mod.py").write_text("Y = 1\n", encoding="utf-8")
        (repo / "app.py").write_text("changed\n", encoding="utf-8")
        bundle = collect_diff("pkg", repo)
        assert bundle.files == ["pkg/mod.py"]

    def test_nothing_to_review_is_an_error_with_a_reason(self, repo):
        with pytest.raises(ReviewError, match="Nothing to review"):
            collect_diff("", repo)

    def test_outside_a_repository(self, tmp_path):
        with pytest.raises(ReviewError, match="not a git repository"):
            collect_diff("", tmp_path)

    def test_lockfiles_are_left_out(self, repo):
        (repo / "package-lock.json").write_text('{"a": 1}\n', encoding="utf-8")
        (repo / "app.py").write_text("changed\n", encoding="utf-8")
        git(repo, "add", "-A")
        assert "package-lock" not in collect_diff("staged", repo).text

    def test_a_big_diff_keeps_the_small_files_whole_and_names_the_big_ones(self, repo):
        (repo / "big.py").write_text("\n".join(f"line_{i} = {i}" for i in range(4000)) + "\n", encoding="utf-8")
        (repo / "util.py").write_text("X = 2\n", encoding="utf-8")
        bundle = collect_diff("", repo, budget=5000)
        assert "+X = 2" in bundle.text and "line_3999" not in bundle.text
        assert bundle.omitted == ["big.py"] and "big.py" in bundle.files
        assert "big.py" in build_task(bundle) and "too large to include" in build_task(bundle)

    def test_binary_new_files_are_skipped(self, repo):
        (repo / "blob.bin").write_bytes(b"\x00\x01\x02")
        (repo / "app.py").write_text("changed\n", encoding="utf-8")
        assert "blob.bin" not in collect_diff("", repo).files


class TestSplitTarget:
    def test_recognises_staged_ranges_paths_and_revisions(self, repo):
        (repo / "sub").mkdir()
        assert split_target("staged only the parser", repo) == ("staged", "only the parser")
        assert split_target("main..HEAD", repo) == ("main..HEAD", "")
        assert split_target("sub the new code", repo) == ("sub", "the new code")
        assert split_target("HEAD", repo) == ("HEAD", "")

    def test_anything_else_is_the_focus(self, repo):
        assert split_target("the SQL escaping", repo) == ("", "the SQL escaping")
        assert split_target("", repo) == ("", "")


class TestParseFindings:
    def good(self):
        return [
            {"file": "app.py", "line": 2, "severity": "low", "summary": "minor", "scenario": ""},
            {"file": "app.py", "line": "7", "severity": "CRITICAL", "summary": "bad", "scenario": "x=None"},
        ]

    def test_reads_a_bare_array_sorted_worst_first(self):
        found = parse_findings(json.dumps(self.good()))
        assert [f.severity for f in found] == ["critical", "low"] and found[0].line == 7

    def test_reads_a_fenced_block_with_chatter_around_it(self):
        text = "Here is my review:\n```json\n" + json.dumps(self.good()) + "\n```\nHope it helps."
        assert len(parse_findings(text)) == 2

    def test_reads_an_array_buried_in_prose(self):
        assert len(parse_findings("Findings: " + json.dumps(self.good()) + " — end")) == 2

    def test_an_empty_list_is_a_clean_review_not_a_failure(self):
        assert parse_findings("[]") == []

    def test_unreadable_answers_are_none_not_an_empty_review(self):
        assert parse_findings("I looked and it seems fine.") is None
        assert parse_findings("[not json]") is None

    def test_odd_values_are_normalised(self):
        found = parse_findings(json.dumps([{"file": "a\\b.py", "line": "n/a", "severity": "scary", "summary": "  s  x "},
                                           {"summary": ""}, "junk"]))
        assert len(found) == 1
        f = found[0]
        assert (f.file, f.line, f.severity, f.summary) == ("a/b.py", 0, "medium", "s x")

    def test_findings_about_files_that_are_not_there_are_dropped(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        found = [Finding("app.py", 1, "high", "real"), Finding("ghost.py", 3, "high", "invented"),
                 Finding("", 0, "low", "general")]
        kept, dropped = drop_unknown_files(found, ["app.py"])
        assert [f.summary for f in kept] == ["real", "general"] and dropped == 1


class TestRunReview:
    def test_the_task_carries_the_diff_the_contract_and_the_focus(self, repo):
        (repo / "app.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
        seen = {}

        def reviewer(task):
            seen["task"] = task
            return json.dumps([{"file": "app.py", "line": 2, "severity": "high",
                                "summary": "add subtracts", "scenario": "add(1, 2) returns -1"}])

        result = run_review("", "arithmetic", cwd=repo, runner=reviewer)
        assert "return a - b" in seen["task"] and "arithmetic" in seen["task"] and "VERIFY" in seen["task"]
        assert result.readable and [f.summary for f in result.findings] == ["add subtracts"]
        text = review.render(result)
        assert "app.py:2" in text and "add(1, 2) returns -1" in text and "high" in text

    def test_an_unreadable_answer_is_shown_as_is(self, repo):
        (repo / "app.py").write_text("x\n", encoding="utf-8")
        result = run_review("", cwd=repo, runner=lambda task: "no idea, looks fine")
        assert not result.readable and "no idea, looks fine" in review.render(result)

    def test_a_clean_review_says_so(self, repo):
        (repo / "app.py").write_text("x\n", encoding="utf-8")
        result = run_review("", cwd=repo, runner=lambda task: "[]")
        assert result.readable and "Существенных проблем не найдено" in review.render(result)

    def test_invented_files_are_reported_as_dropped(self, repo):
        (repo / "app.py").write_text("x\n", encoding="utf-8")
        reply = json.dumps([{"file": "nowhere.py", "line": 1, "severity": "high", "summary": "ghost"}])
        result = run_review("", cwd=repo, runner=lambda task: reply)
        assert result.findings == [] and result.dropped == 1 and "Отброшено" in review.render(result)

    def test_the_review_is_saved_and_the_fix_request_lists_the_findings(self, repo):
        (repo / "app.py").write_text("x\n", encoding="utf-8")
        reply = json.dumps([{"file": "app.py", "line": 1, "severity": "high", "summary": "oops", "scenario": "always"}])
        result = run_review("", cwd=repo, runner=lambda task: reply)
        path = review.save(result, repo)
        assert path.endswith(".md") and "oops" in (repo / ".argent" / "artifacts").glob("review-*.md").__next__().read_text(encoding="utf-8")
        request = review.fix_request(result)
        assert "app.py:1" in request and "oops" in request and "always" in request

    def test_the_reviewer_gets_only_tools_that_read(self):
        for forbidden in ("write_file", "replace_in_file", "run_command", "delete_file"):
            assert forbidden not in review.READONLY_TOOLS
