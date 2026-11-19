import subprocess
import sys
from pathlib import Path

from argent.bench.case import Case, Check
from argent.bench.report import to_markdown
from argent.bench.runner import run_case


class _WriterAgent:
    def __init__(self, text: str):
        self.text = text

    def process_user_input(self, task):
        Path("out.txt").write_text(self.text, encoding="utf-8")
        yield {"type": "content_stream", "content": "done"}


def _case(check_text: str = "ok") -> Case:
    cmd = (
        f'"{sys.executable}" -c "from pathlib import Path; '
        f'assert Path(\'out.txt\').read_text(encoding=\'utf-8\') == \'{check_text}\'"'
    )
    return Case(id="write", task="write out.txt", checks=[Check(cmd=cmd)])


def test_a_passing_case_runs_its_agent_and_check(tmp_path):
    result = run_case(
        _case(),
        repo=tmp_path,
        worktree=tmp_path,
        agent_factory=lambda: _WriterAgent("ok"),
    )
    assert result.passed is True
    assert result.events >= 2  # content + done
    assert result.checks[0].passed is True
    assert "write" in to_markdown([result])


def test_a_failing_check_marks_the_case_failed(tmp_path):
    result = run_case(
        _case("expected"),
        repo=tmp_path,
        worktree=tmp_path,
        agent_factory=lambda: _WriterAgent("wrong"),
    )
    assert result.passed is False
    assert result.checks[0].passed is False


def test_a_failing_setup_stops_before_the_agent(tmp_path):
    case = _case()
    case.setup = ["exit 3"]
    result = run_case(
        case,
        repo=tmp_path,
        worktree=tmp_path,
        agent_factory=lambda: _WriterAgent("ok"),
    )
    assert result.passed is False
    assert "setup failed" in result.setup_error
    assert not (tmp_path / "out.txt").exists()


def test_run_case_uses_an_isolated_git_worktree(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "bench@example.test"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Bench"], cwd=repo, check=True)
    (repo / "base.txt").write_text("base", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=repo, check=True)

    result = run_case(
        _case(),
        repo=repo,
        agent_factory=lambda: _WriterAgent("ok"),
    )
    assert result.passed is True
    assert result.worktree != str(repo)
    assert not (repo / "out.txt").exists()
    assert (repo / "base.txt").read_text(encoding="utf-8") == "base"
    assert not Path(result.worktree).exists()

