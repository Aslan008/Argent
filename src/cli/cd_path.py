"""Parse the /cd argument: strip the enclosing quotes from a quoted path.

On Windows a path with spaces is usually pasted in wrapped in quotes
(/cd "C:\\Users\\name\\My Project"). The CLI takes the rest of the line
verbatim, so the quotes end up as part of the path and Path().resolve()
fails with "directory does not exist". Stripping the matching outer quotes
makes such paths work.
"""

QUOTES = "\"'"


def clean_cd_path(raw: str) -> str:
    """Strip a matching pair of enclosing quotes from a /cd argument.

    Handles both double ("...") and single ('...') quotes. A path that is
    not wrapped in a matching pair is returned as-is: spaces are legal in
    Windows paths, so an unquoted multi-word path still resolves fine.
    """
    text = raw.strip()
    if len(text) >= 2 and text[0] in QUOTES and text[-1] == text[0]:
        return text[1:-1]
    return text
