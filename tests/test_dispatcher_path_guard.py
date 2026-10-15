"""The dispatcher must not run a "file", a "command" or an "expression" that is really a sentence.

The extract head sometimes hands back the request itself ("открой его", "проверь сеть"), a phrase ("папке проекта"), LaTeX, or a span that
runs into the next line of the routing state.  Dispatching a tool with such a string only fails and costs a turn, so the argument builder
refuses it and the request goes to the LLM.
"""

import pytest

from src.leeth.dispatcher import LeethDispatcher

D = LeethDispatcher()


@pytest.mark.parametrize("tool", ["read_file", "list_directory"])
@pytest.mark.parametrize("param", [
    "открой его", "покажи его", "этот файл", "папке проекта", "открой тот файл ещё раз", "what is in it",
    'этот файл"\nWorking directory: C:/Users/user/Desktop/Argent', "src/app.py\nWorking directory: C:/x",
])
def test_a_sentence_is_not_a_path(tool, param):
    assert D._build_arguments(tool, param) is None


@pytest.mark.parametrize("param", [
    "src/app.py", "README", "Makefile", "C:\\Users\\user\\UnityProjects\\First Proj\\Assets\\Scripts", "docs/my file.md", "~/work/api", ".gitignore",
    "config.yaml", "..", "C:/work/app", "My Notes.txt",
])
def test_real_paths_still_go_through(param):
    assert D._build_arguments("read_file", param) == {"file_path": param}
    assert D._build_arguments("list_directory", param) == {"dir_path": param}


def test_other_tools_are_untouched():
    assert D._build_arguments("grep_search", "поиск по тексту") == {"pattern": "поиск по тексту", "directory": "."}
    assert D._build_arguments("run_command", "git status") == {"command": "git status"}
    assert D._build_arguments("calculate", "2 + 2") == {"expression": "2 + 2"}
    assert D._build_arguments("browser_open", "example.com") == {"url": "https://example.com"}


@pytest.mark.parametrize("param", ["проверь сеть", "пересобери zip архив", "покажи статус гит", "  запусти сервер", "Проверь V7"])
def test_a_russian_phrase_is_not_a_shell_command(param):
    assert D._build_arguments("run_command", param) is None


@pytest.mark.parametrize("param", ["git status", "pytest -x", "cd C:\\Users\\user\\Desktop\\Agem.io", "./run.sh", "python main.py --имя", "C:\\tools\\a.exe /c", "npm run dev"])
def test_real_commands_still_go_through(param):
    assert D._build_arguments("run_command", param) == {"command": param}


@pytest.mark.parametrize("param", ["((((100 \\cdot 2 - 50) : 5 + \\sqrt{16}) \\cdot 3 - 20) : 2 + 55 \\cdot (10 - 2^3)\\)", "\\frac{3}{4} + 2", "x^{2} + 1"])
def test_latex_is_not_an_expression_the_calculator_takes(param):
    assert D._build_arguments("calculate", param) is None


@pytest.mark.parametrize("param", ["2 + 2 * 2", "123*47", "15% от 240", "sqrt(16) + 3", "(3+4)*5", "2**10"])
def test_plain_arithmetic_still_goes_through(param):
    assert D._build_arguments("calculate", param) == {"expression": param}
