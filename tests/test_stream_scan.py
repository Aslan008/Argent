"""The raw-tool marker scan must cost O(chunk), not O(buffer) per token.

Rescanning the whole accumulated answer on every chunk made a long reply
quadratic (measured: 12-193 ms for a 54 KB answer, depending on chunk size).
The fix scans only the tail that a marker could straddle.
"""

from argent.agent import _RAW_TOOL_LOOKBACK, _raw_tool_marker_started


def feed(chunks):
    """Replay `chunks` through the production scan; return the final flag."""
    buffer = ""
    building = False
    for chunk in chunks:
        previous = len(buffer)
        buffer += chunk
        if not building and _raw_tool_marker_started(buffer, previous):
            building = True
    return building


class TestRawToolMarker:
    def test_detects_fenced_json_in_one_chunk(self):
        assert feed(['```json\n{"name": "read_file"}']) is True

    def test_detects_fence_split_across_chunks(self):
        # Streaming splits the fence itself, so the scan has to look back.
        assert feed(["ok\n```", "js", 'on\n{"name": "x"}']) is True

    def test_detects_multiline_name_marker_split(self):
        assert feed(['see: {\n  ', '"na', 'me": "write_file"}']) is True

    def test_plain_prose_is_not_a_tool_call(self):
        assert feed(["Просто текст. ", "Ещё текст. ", '{"a": 1}']) is False

    def test_marker_before_the_window_is_not_rescanned(self):
        # Answering False here is what keeps the per-token cost constant.
        buffer = "```json" + "x" * 500
        assert _raw_tool_marker_started(buffer, len(buffer)) is False

    def test_lookback_covers_the_longest_marker(self):
        assert _RAW_TOOL_LOOKBACK == len('{\n  "name":') - 1
