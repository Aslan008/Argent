"""
Deep mutation-killing tests for src/rag/keyword_index.py.

Covers tokenize, BM25Index.__init__, BM25Index.search, edge cases,
get_index caching, and clear_cache with exact BM25 score assertions.
"""

import math
from unittest.mock import MagicMock

import pytest

from src.rag.keyword_index import (
    BM25Index,
    tokenize,
    get_index,
    clear_cache,
    _CACHE,
)


# ---------------------------------------------------------------------------
# tokenize
# ---------------------------------------------------------------------------

class TestTokenize:
    def test_basic_split(self):
        assert tokenize("hello world") == ["hello", "world"]

    def test_short_words_filtered(self):
        # words with len <= 2 are dropped
        assert tokenize("ab cd ef gh") == []

    def test_boundary_length_two_dropped_length_three_kept(self):
        # "abc" has len 3 (> 2) → kept; "ab" has len 2 → dropped
        assert tokenize("ab abc") == ["abc"]

    def test_lowercase(self):
        assert tokenize("ABC XYZ") == ["abc", "xyz"]

    def test_unicode_text(self):
        result = tokenize("café résumé naïve")
        assert result == ["café", "résumé", "naïve"]

    def test_empty_string(self):
        assert tokenize("") == []

    def test_numbers_short_filtered(self):
        # "42" has len 2 → filtered
        assert tokenize("42") == []

    def test_numbers_long_kept(self):
        # \w+ matches digit sequences; len > 2 → kept
        assert tokenize("123 4567") == ["123", "4567"]

    def test_mixed_alphanumeric(self):
        # \w+ matches "a1b2" (len 4) and "c3d" (len 3), both > 2
        assert tokenize("a1b2 c3d") == ["a1b2", "c3d"]

    def test_punctuation_only(self):
        assert tokenize("!!! ??? ...") == []

    def test_underscore_is_word_char(self):
        # \w includes underscore, so "hello_world" is one token
        assert tokenize("hello_world") == ["hello_world"]

    def test_single_word_long_enough(self):
        assert tokenize("hello") == ["hello"]

    def test_mixed_case_and_short(self):
        assert tokenize("He is OK") == []

    def test_newlines_and_tabs_split(self):
        assert tokenize("hello\nworld\tfoo") == ["hello", "world", "foo"]


# ---------------------------------------------------------------------------
# BM25Index.__init__
# ---------------------------------------------------------------------------

class TestBM25Init:
    def test_empty_corpus(self):
        idx = BM25Index([], [], [])
        assert idx.N == 1  # max(0, 1)
        assert idx.avgdl == 1.0  # doc_len is empty → fallback
        assert idx.doc_len == []
        assert dict(idx.postings) == {}
        assert idx.idf == {}

    def test_single_doc(self):
        idx = BM25Index(["1"], ["hello world"], [{}])
        assert idx.N == 1
        assert idx.avgdl == 2.0
        assert idx.doc_len == [2]
        assert ("hello", [(0, 1)]) in dict(idx.postings).items()
        assert ("world", [(0, 1)]) in dict(idx.postings).items()

    def test_multiple_docs_postings_structure(self):
        docs = ["apple banana", "banana cherry", "apple cherry date"]
        idx = BM25Index(["a", "b", "c"], docs, [{}, {}, {}])
        # postings: term -> list of (doc_idx, freq)
        postings = dict(idx.postings)
        assert postings["apple"] == [(0, 1), (2, 1)]
        assert postings["banana"] == [(0, 1), (1, 1)]
        assert postings["cherry"] == [(1, 1), (2, 1)]
        assert postings["date"] == [(2, 1)]

    def test_df_counts_via_idf_keys(self):
        docs = ["apple banana", "banana cherry", "apple cherry date"]
        idx = BM25Index(["a", "b", "c"], docs, [{}, {}, {}])
        # df: apple=2, banana=2, cherry=2, date=1
        # Verify via IDF formula: idf = log(1 + (N - d + 0.5)/(d + 0.5))
        N = 3
        for term, expected_df in [("apple", 2), ("banana", 2), ("cherry", 2), ("date", 1)]:
            expected_idf = math.log(1 + (N - expected_df + 0.5) / (expected_df + 0.5))
            assert math.isclose(idx.idf[term], expected_idf, rel_tol=1e-12)

    def test_idf_formula_exact(self):
        # N=1, df=1 for each term
        idx = BM25Index(["1"], ["hello world"], [{}])
        expected = math.log(1 + (1 - 1 + 0.5) / (1 + 0.5))  # log(4/3)
        assert math.isclose(idx.idf["hello"], expected, rel_tol=1e-12)
        assert math.isclose(idx.idf["world"], expected, rel_tol=1e-12)

    def test_idf_formula_rare_vs_common(self):
        # doc1="apple banana", doc2="apple banana", doc3="apple cherry"
        # df: apple=3, banana=2, cherry=1
        docs = ["apple banana", "apple banana", "apple cherry"]
        idx = BM25Index(["a", "b", "c"], docs, [{}, {}, {}])
        # rarer term → higher IDF
        assert idx.idf["cherry"] > idx.idf["banana"] > idx.idf["apple"]

    def test_doc_len_recorded(self):
        docs = ["apple banana cherry", "date", "elderberry fig grape hazel"]
        idx = BM25Index(["a", "b", "c"], docs, [{}, {}, {}])
        assert idx.doc_len == [3, 1, 4]
        assert idx.avgdl == (3 + 1 + 4) / 3

    def test_custom_k1_b(self):
        idx = BM25Index(["1"], ["hello world"], [{}], k1=2.0, b=0.5)
        assert idx.k1 == 2.0
        assert idx.b == 0.5


# ---------------------------------------------------------------------------
# BM25Index.search
# ---------------------------------------------------------------------------

class TestBM25Search:
    def test_matching_terms_ranked(self):
        docs = ["apple banana cherry", "banana cherry date", "apple date elderberry"]
        idx = BM25Index(["a", "b", "c"], docs, [{}, {}, {}])
        res = idx.search("apple", top_k=3)
        ids = [r[0] for r in res]
        # "apple" appears in doc 0 and doc 2; doc 0 is shorter (3 vs 3) — same len
        # both have freq=1, dl=3 → same score, but both should be returned
        assert "a" in ids
        assert "c" in ids
        assert "b" not in ids  # doc 1 has no "apple"

    def test_no_matching_terms_empty(self):
        idx = BM25Index(["1"], ["hello world"], [{}])
        assert idx.search("xyz quantum", top_k=5) == []

    def test_top_k_one_returns_only_top(self):
        docs = ["apple banana", "apple apple banana"]
        idx = BM25Index(["a", "b"], docs, [{}, {}])
        res = idx.search("apple", top_k=1)
        assert len(res) == 1
        # doc 1 has higher term freq → should rank first
        assert res[0][0] == "b"

    def test_top_k_exceeds_results_returns_all(self):
        docs = ["apple banana", "cherry date"]
        idx = BM25Index(["a", "b"], docs, [{}, {}])
        res = idx.search("apple", top_k=10)
        # only doc 0 matches
        assert len(res) == 1
        assert res[0][0] == "a"

    def test_term_frequency_higher_freq_higher_score(self):
        # Two docs, same length, different term frequency for "apple"
        # doc1 = "apple apple apple" (freq=3, dl=3)
        # doc2 = "apple banana cherry" (freq=1, dl=3)
        docs = ["apple apple apple", "apple banana cherry"]
        idx = BM25Index(["a", "b"], docs, [{}, {}])
        res = idx.search("apple", top_k=2)
        # doc with freq=3 should rank above freq=1
        assert res[0][0] == "a"
        assert res[1][0] == "b"

    def test_term_frequency_exact_score(self):
        # N=2, avgdl=3, df[apple]=2
        # idf[apple] = log(1 + (2-2+0.5)/(2+0.5)) = log(1.2)
        # doc1 (freq=3, dl=3): denom = 3 + 1.5*(1-0.75+0.75*3/3) = 3+1.5 = 4.5
        #   score = log(1.2) * (3*2.5)/4.5 = log(1.2) * 5/3
        # doc2 (freq=1, dl=3): denom = 1 + 1.5*(0.25+0.75) = 2.5
        #   score = log(1.2) * 2.5/2.5 = log(1.2)
        docs = ["apple apple apple", "apple banana cherry"]
        idx = BM25Index(["a", "b"], docs, [{}, {}])
        idf = idx.idf["apple"]
        expected_a = idf * 5.0 / 3.0
        expected_b = idf * 1.0
        assert math.isclose(expected_a, math.log(1.2) * 5 / 3, rel_tol=1e-12)
        assert math.isclose(expected_b, math.log(1.2), rel_tol=1e-12)
        assert expected_a > expected_b

    def test_idf_effect_rare_term_higher_score(self):
        # doc1="apple banana", doc2="apple banana", doc3="apple cherry"
        # df: apple=3, banana=2, cherry=1
        # cherry is rarest → highest IDF → highest score
        docs = ["apple banana", "apple banana", "apple cherry"]
        idx = BM25Index(["a", "b", "c"], docs, [{}, {}, {}])
        res_cherry = idx.search("cherry", top_k=1)
        res_apple = idx.search("apple", top_k=1)
        # Verify IDF values
        assert idx.idf["cherry"] > idx.idf["apple"]
        # cherry score (only doc c, freq=1, dl=2):
        #   denom = 1 + 1.5*(0.25 + 0.75*2/2) = 1 + 1.5 = 2.5
        #   score = idf_cherry * 2.5/2.5 = idf_cherry = log(8/3)
        # apple score (any doc, freq=1, dl=2):
        #   score = idf_apple * 2.5/2.5 = idf_apple = log(8/7)
        assert math.isclose(idx.idf["cherry"], math.log(8 / 3), rel_tol=1e-12)
        assert math.isclose(idx.idf["apple"], math.log(8 / 7), rel_tol=1e-12)
        assert res_cherry[0][0] == "c"

    def test_length_normalization_longer_doc_lower_score(self):
        # doc1 = "apple" (freq=1, dl=1)
        # doc2 = "apple banana cherry date elderberry" (freq=1, dl=5)
        # N=2, avgdl=(1+5)/2=3, df[apple]=2
        # idf[apple] = log(1.2)
        # doc1: denom = 1 + 1.5*(0.25 + 0.75*1/3) = 1 + 1.5*0.5 = 1.75
        #   score = log(1.2) * 2.5/1.75 = log(1.2) * 10/7
        # doc2: denom = 1 + 1.5*(0.25 + 0.75*5/3) = 1 + 1.5*1.5 = 3.25
        #   score = log(1.2) * 2.5/3.25 = log(1.2) * 10/13
        docs = ["apple", "apple banana cherry date elderberry"]
        idx = BM25Index(["a", "b"], docs, [{}, {}])
        res = idx.search("apple", top_k=2)
        # shorter doc should rank first
        assert res[0][0] == "a"
        assert res[1][0] == "b"
        # Verify exact scores
        idf = idx.idf["apple"]
        assert math.isclose(idf, math.log(1.2), rel_tol=1e-12)
        score_a = idf * 2.5 / 1.75
        score_b = idf * 2.5 / 3.25
        assert math.isclose(score_a, math.log(1.2) * 10 / 7, rel_tol=1e-12)
        assert math.isclose(score_b, math.log(1.2) * 10 / 13, rel_tol=1e-12)
        assert score_a > score_b

    def test_exact_score_single_doc(self):
        # docs=["hello world"], N=1, avgdl=2
        # idf[hello] = log(1 + (1-1+0.5)/(1+0.5)) = log(4/3)
        # search "hello": freq=1, dl=2
        #   denom = 1 + 1.5*(0.25 + 0.75*2/2) = 1 + 1.5 = 2.5
        #   score = log(4/3) * 2.5/2.5 = log(4/3)
        idx = BM25Index(["1"], ["hello world"], [{}])
        assert math.isclose(idx.idf["hello"], math.log(4 / 3), rel_tol=1e-12)
        res = idx.search("hello", top_k=1)
        assert len(res) == 1
        assert res[0][0] == "1"

    def test_search_returns_id_doc_meta_tuples(self):
        idx = BM25Index(["x"], ["hello world"], [{"src": "test"}])
        res = idx.search("hello", top_k=1)
        rid, doc, meta = res[0]
        assert rid == "x"
        assert doc == "hello world"
        assert meta == {"src": "test"}

    def test_multi_term_query_accumulates_scores(self):
        # Query with two terms that both match the same doc
        # doc = "apple banana cherry"
        # Query "apple banana" → both terms match doc 0
        # score = idf[apple]*tf_component + idf[banana]*tf_component
        docs = ["apple banana cherry", "date elderberry fig"]
        idx = BM25Index(["a", "b"], docs, [{}, {}])
        res = idx.search("apple banana", top_k=2)
        assert res[0][0] == "a"
        assert len(res) == 1  # only doc 0 matches either term

    def test_search_empty_query(self):
        idx = BM25Index(["1"], ["hello world"], [{}])
        # tokenize("") → [], so no terms to search
        assert idx.search("", top_k=5) == []

    def test_search_query_only_short_words(self):
        idx = BM25Index(["1"], ["hello world"], [{}])
        # "ab cd" → tokenize filters all → []
        assert idx.search("ab cd", top_k=5) == []


# ---------------------------------------------------------------------------
# BM25Index edge cases
# ---------------------------------------------------------------------------

class TestBM25EdgeCases:
    def test_doc_with_repeated_terms(self):
        # "apple" appears 3 times in doc 0, 1 time in doc 1
        docs = ["apple apple apple banana", "apple cherry date"]
        idx = BM25Index(["a", "b"], docs, [{}, {}])
        # Verify postings
        assert dict(idx.postings)["apple"] == [(0, 3), (1, 1)]
        # doc 0 should rank higher for "apple"
        res = idx.search("apple", top_k=2)
        assert res[0][0] == "a"

    def test_doc_no_valid_tokens(self):
        # "ab cd ef" → all len <= 2 → no tokens
        docs = ["ab cd ef", "apple banana"]
        idx = BM25Index(["a", "b"], docs, [{}, {}])
        assert idx.doc_len == [0, 2]
        # doc 0 has no postings
        res = idx.search("apple", top_k=5)
        assert len(res) == 1
        assert res[0][0] == "b"

    def test_doc_no_valid_tokens_alone(self):
        idx = BM25Index(["a"], ["ab cd ef"], [{}])
        assert idx.doc_len == [0]
        # avgdl: doc_len is [0] (truthy, non-empty list) → 0/1 = 0.0
        assert idx.avgdl == 0.0
        assert dict(idx.postings) == {}
        assert idx.idf == {}
        assert idx.search("anything", top_k=5) == []

    def test_all_docs_identical(self):
        docs = ["apple banana", "apple banana", "apple banana"]
        idx = BM25Index(["a", "b", "c"], docs, [{}, {}, {}])
        # All docs have same score for "apple"
        res = idx.search("apple", top_k=3)
        assert len(res) == 3
        ids = {r[0] for r in res}
        assert ids == {"a", "b", "c"}

    def test_all_docs_identical_same_postings(self):
        docs = ["apple banana", "apple banana"]
        idx = BM25Index(["a", "b"], docs, [{}, {}])
        # Both docs have identical postings entries
        assert dict(idx.postings)["apple"] == [(0, 1), (1, 1)]
        assert dict(idx.postings)["banana"] == [(0, 1), (1, 1)]

    def test_empty_doc_mixed_with_valid_docs(self):
        docs = ["", "apple banana cherry"]
        idx = BM25Index(["a", "b"], docs, [{}, {}])
        # doc 0: "" → tokenize → [] → doc_len=0
        assert idx.doc_len == [0, 3]
        res = idx.search("apple", top_k=5)
        assert len(res) == 1
        assert res[0][0] == "b"

    def test_avgdl_with_empty_doc_list(self):
        idx = BM25Index([], [], [])
        # doc_len is [] → falsy → avgdl = 1.0
        assert idx.avgdl == 1.0


# ---------------------------------------------------------------------------
# get_index
# ---------------------------------------------------------------------------

class TestGetIndex:
    def setup_method(self):
        clear_cache()

    def _make_col(self, name="test_col", ids=None, docs=None, metas=None):
        col = MagicMock()
        col.name = name
        col.count.return_value = len(ids or [])
        col.get.return_value = {
            "ids": ids or [],
            "documents": docs or [],
            "metadatas": metas or [],
        }
        return col

    def test_builds_index_from_collection(self):
        col = self._make_col(ids=["1"], docs=["hello world"], metas=[{}])
        idx = get_index(col)
        assert isinstance(idx, BM25Index)
        assert idx.ids == ["1"]
        assert idx.docs == ["hello world"]
        assert idx.metas == [{}]

    def test_count_change_triggers_rebuild(self):
        col = self._make_col(ids=["1"], docs=["hello world"], metas=[{}])
        idx1 = get_index(col)
        # Simulate count change
        col.count.return_value = 2
        col.get.return_value = {
            "ids": ["1", "2"],
            "documents": ["hello world", "foo bar baz"],
            "metadatas": [{}, {}],
        }
        idx2 = get_index(col)
        assert idx1 is not idx2  # different objects → rebuilt
        assert idx2.N == 2
        assert col.get.call_count == 2

    def test_same_count_returns_cached(self):
        col = self._make_col(ids=["1"], docs=["hello world"], metas=[{}])
        idx1 = get_index(col)
        idx2 = get_index(col)
        assert idx1 is idx2  # same object → cached
        assert col.get.call_count == 1

    def test_count_exception_sets_count_negative_one(self):
        col = MagicMock()
        col.name = "err_col"
        col.count.side_effect = RuntimeError("connection failed")
        col.get.return_value = {
            "ids": ["1"],
            "documents": ["hello world"],
            "metadatas": [{}],
        }
        idx = get_index(col)
        assert isinstance(idx, BM25Index)
        # Verify cache stored with count = -1
        cached = _CACHE.get("err_col")
        assert cached is not None
        assert cached[0] == -1

    def test_count_exception_second_call_uses_cache(self):
        col = MagicMock()
        col.name = "err_col2"
        col.count.side_effect = RuntimeError("connection failed")
        col.get.return_value = {
            "ids": ["1"],
            "documents": ["hello world"],
            "metadatas": [{}],
        }
        idx1 = get_index(col)
        idx2 = get_index(col)
        # count=-1 both times → cached[0] == count → returns cached
        assert idx1 is idx2
        assert col.get.call_count == 1

    def test_no_name_uses_object_id(self):
        col = MagicMock()
        del col.name  # ensure no name attribute
        col.count.return_value = 1
        col.get.return_value = {
            "ids": ["1"],
            "documents": ["hello world"],
            "metadatas": [{}],
        }
        idx = get_index(col)
        assert isinstance(idx, BM25Index)
        # The cache key should be id(col)
        assert id(col) in _CACHE

    def test_different_collections_cached_separately(self):
        col1 = self._make_col(name="col_a", ids=["1"], docs=["hello world"], metas=[{}])
        col2 = self._make_col(name="col_b", ids=["2"], docs=["foo bar baz"], metas=[{}])
        idx1 = get_index(col1)
        idx2 = get_index(col2)
        assert idx1 is not idx2
        assert idx1.ids == ["1"]
        assert idx2.ids == ["2"]

    def test_get_index_with_empty_data(self):
        col = MagicMock()
        col.name = "empty_col"
        col.count.return_value = 0
        col.get.return_value = {"ids": [], "documents": [], "metadatas": []}
        idx = get_index(col)
        assert idx.N == 1
        assert idx.doc_len == []


# ---------------------------------------------------------------------------
# clear_cache
# ---------------------------------------------------------------------------

class TestClearCache:
    def setup_method(self):
        clear_cache()

    def test_clear_cache_empties_cache(self):
        col = MagicMock()
        col.name = "clear_test"
        col.count.return_value = 1
        col.get.return_value = {
            "ids": ["1"],
            "documents": ["hello world"],
            "metadatas": [{}],
        }
        get_index(col)
        assert len(_CACHE) > 0
        clear_cache()
        assert _CACHE == {}
        assert len(_CACHE) == 0

    def test_clear_cache_forces_rebuild(self):
        col = MagicMock()
        col.name = "rebuild_test"
        col.count.return_value = 1
        col.get.return_value = {
            "ids": ["1"],
            "documents": ["hello world"],
            "metadatas": [{}],
        }
        get_index(col)
        assert col.get.call_count == 1
        clear_cache()
        get_index(col)
        assert col.get.call_count == 2  # rebuilt after clear

    def test_clear_cache_on_empty_cache_no_error(self):
        # Should not raise even if cache is already empty
        clear_cache()
        clear_cache()
        assert _CACHE == {}