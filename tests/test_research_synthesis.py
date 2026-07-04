"""Pure synthesis helpers: LLM query-list parsing + numbered source citations."""

from src.research.synthesis import parse_query_list, number_sources


class TestParseQueryList:
    def test_plain_json_array(self):
        assert parse_query_list('["a", "b", "c"]') == ["a", "b", "c"]

    def test_fenced_json_block(self):
        raw = '```json\n["x", "y"]\n```'
        assert parse_query_list(raw) == ["x", "y"]

    def test_array_with_trailing_prose_via_fence_strip(self):
        assert parse_query_list('```\n["only"]\n```') == ["only"]

    def test_limit_applied(self):
        assert parse_query_list('["1","2","3","4","5","6"]', limit=3) == ["1", "2", "3"]

    def test_empty_input(self):
        assert parse_query_list("") == []
        assert parse_query_list(None) == []

    def test_non_list_json(self):
        assert parse_query_list('{"a": 1}') == []

    def test_empty_array(self):
        assert parse_query_list("[]") == []

    def test_garbage(self):
        assert parse_query_list("not json at all") == []

    def test_coerces_non_strings(self):
        assert parse_query_list("[1, 2]") == ["1", "2"]


class TestNumberSources:
    def test_numbers_unique_urls_in_appearance_order(self):
        chunks = ["c1", "c2", "c3"]
        smap = {"c1": "http://a", "c2": "http://b", "c3": "http://a"}
        annotated, sources = number_sources(chunks, smap)
        assert "[1] c1" in annotated and "[2] c2" in annotated and "[1] c3" in annotated
        assert sources == ["[1] http://a", "[2] http://b"]

    def test_chunk_without_source_is_unnumbered(self):
        chunks = ["c1", "orphan"]
        smap = {"c1": "http://a"}
        annotated, sources = number_sources(chunks, smap)
        assert "[1] c1" in annotated
        # the orphan keeps no marker
        assert "\norphan" in annotated or annotated.endswith("orphan")
        assert sources == ["[1] http://a"]

    def test_empty(self):
        annotated, sources = number_sources([], {})
        assert annotated == "" and sources == []

    def test_separator_between_chunks(self):
        annotated, _ = number_sources(["a", "b"], {"a": "http://x", "b": "http://x"})
        assert "\n---\n" in annotated
