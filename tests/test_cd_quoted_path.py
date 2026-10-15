"""Regression: /cd must accept paths wrapped in quotes.

Bug: the /cd handler took the rest of the line verbatim, so a quoted path
(/cd "C:\\Users\\user\\Desktop\\SiteCite") kept its quotes as part of the
path and Path().resolve() failed with "directory does not exist".
"""

from pathlib import Path

from src.cli.cd_path import clean_cd_path


def test_double_quoted_path():
    path = clean_cd_path('"C:\\Users\\user\\Desktop\\SiteCite"')
    assert path == "C:\\Users\\user\\Desktop\\SiteCite"
    assert Path(path).is_absolute()


def test_single_quoted_path():
    path = clean_cd_path("'C:\\Users\\user\\Desktop\\SiteCite'")
    assert path == "C:\\Users\\user\\Desktop\\SiteCite"


def test_quoted_path_with_spaces():
    path = clean_cd_path('"C:\\Users\\My Projects\\Site"')
    assert path == "C:\\Users\\My Projects\\Site"


def test_unquoted_path_unchanged():
    assert clean_cd_path("C:\\Users\\user\\Desktop") == "C:\\Users\\user\\Desktop"


def test_unquoted_path_with_spaces_unchanged():
    # Spaces are legal in Windows paths: an unquoted multi-word path still resolves.
    assert clean_cd_path("C:\\Users\\My Projects\\Site") == "C:\\Users\\My Projects\\Site"


def test_relative_path_unchanged():
    assert clean_cd_path("src") == "src"


def test_mismatched_quotes_left_alone():
    assert clean_cd_path('"C:\\Users\\name\'') == '"C:\\Users\\name\''


def test_lone_quote_left_alone():
    assert clean_cd_path('"') == '"'


def test_empty_string():
    assert clean_cd_path("") == ""
    assert clean_cd_path("   ") == ""
    assert clean_cd_path("''") == ""
