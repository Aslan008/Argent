"""The streaming markdown segmenter behind append-only rendering."""

from src.cli.stream_markdown import MarkdownStreamSplitter, _last_safe_boundary


class TestBoundary:
    def test_complete_paragraphs_commit_fully(self):
        t = "A\n\nB\n\n"
        assert _last_safe_boundary(t) == len(t)

    def test_incomplete_trailing_paragraph_stays(self):
        assert _last_safe_boundary("A\n\nB") == len("A\n\n")

    def test_nothing_safe_without_a_blank_line(self):
        assert _last_safe_boundary("just typing") == 0

    def test_open_fence_blocks_commit(self):
        assert _last_safe_boundary("```py\ncode\n") == 0

    def test_blank_line_inside_fence_is_not_a_boundary(self):
        t = "```\n\n```\n\n"
        assert _last_safe_boundary(t) == len(t)  # only after the fence closes

    def test_closed_fence_is_a_boundary(self):
        t = "```py\ncode\n```\n\nNext"
        assert _last_safe_boundary(t) == len("```py\ncode\n```\n\n")


class TestSplitterStreaming:
    def test_commits_block_on_blank_line(self):
        s = MarkdownStreamSplitter()
        assert s.feed("Hello ") == ""
        assert s.feed("world.") == ""        # no boundary yet
        assert s.feed("\n\n") == "Hello world.\n\n"
        assert s.pending() == ""

    def test_keeps_incomplete_tail_pending(self):
        s = MarkdownStreamSplitter()
        s.feed("Para one.\n\nPara two")
        assert s.pending() == "Para two"

    def test_code_block_commits_only_when_closed(self):
        s = MarkdownStreamSplitter()
        assert s.feed("```python\n") == ""
        assert s.feed("x = 1\n") == ""        # still open
        committed = s.feed("```\n\n")
        assert "```python" in committed and "x = 1" in committed

    def test_finalize_flushes_remainder(self):
        s = MarkdownStreamSplitter()
        s.feed("trailing text with no blank line")
        assert s.finalize() == "trailing text with no blank line"
        assert s.pending() == ""

    def test_reconstructs_full_text(self):
        s = MarkdownStreamSplitter()
        chunks = ["# Title\n\n", "Some ", "prose.\n\n", "```\ncode\n```\n\n", "tail"]
        out = ""
        for ch in chunks:
            out += s.feed(ch)
        out += s.finalize()
        assert out == "".join(chunks)


class TestReplace:
    def test_replace_keeps_committed_prefix(self):
        s = MarkdownStreamSplitter()
        s.feed("Here.\n\n")            # committed
        s.feed('{"tool": "x"}')        # pending raw JSON
        s.replace("Here.\n\n")         # JSON stripped; cleaned == committed prefix
        assert s.pending() == ""
        assert s.finalize() == ""

    def test_replace_whole_when_nothing_committed(self):
        s = MarkdownStreamSplitter()
        s.feed('{"tool": "read_file"}')   # single line, nothing committed
        s.replace("")                     # JSON-only turn -> cleaned empty
        assert s.pending() == ""

    def test_replace_non_prefix_drops_pending(self):
        s = MarkdownStreamSplitter()
        s.feed("Committed.\n\n")
        s.feed("pending")
        s.replace("totally different")    # not a prefix of committed
        assert s.pending() == ""          # best-effort: keep committed, drop tail
