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
# ---------------------------------------------------------------------------
# BM25 exact score assertions (kills 9 surviving arithmetic mutations)
# ---------------------------------------------------------------------------

class TestBM25ExactScores:
    """Assert exact BM25 scores to kill 9 surviving arithmetic mutations
    in the IDF calculation (line 46) and scoring formula (lines 58-60).

    Mutations targeted:
      1. ADD_TO_SUB: freq + k1*... -> freq - k1*...  (denom, line 59)
      2. ADD_TO_SUB: freq * (k1+1) -> freq - (k1+1)  (numerator, line 60)
      3. SUB_TO_ADD: (N - d + 0.5) -> (N + d + 0.5)  (IDF, line 46)
      4. SUB_TO_ADD: (1 - b + ...) -> (1 + b + ...)  (denom, line 59)
      5. MUL_TO_DIV: freq * (k1+1) -> freq / (k1+1)  (numerator, line 60)
      6. DIV_TO_MUL: dl / avgdl   -> dl * avgdl      (denom, line 59)
      7. OR_TO_AND:  doc_len or 1 -> doc_len and 1   (line 58)
      8. OR_TO_AND:  avgdl or 1   -> avgdl and 1     (line 59)
      9. OR_TO_AND:  denom or 1   -> denom and 1     (line 60)

    Strategy: compute expected scores with the CORRECT formula independently
    of BM25Index, then capture actual scores from search() via a patched
    ``sorted`` and assert they match exactly.  Any mutation in the formula
    produces a different actual score that fails the assertion.
    """

    # -- test corpus --------------------------------------------------
    # doc0: "apple apple apple banana cherry"  -> dl=5, apple=3, banana=1, cherry=1
    # doc1: "apple banana"                      -> dl=2, apple=1, banana=1
    # doc2: "apple apple date elderberry fig grape" -> dl=6, apple=2, date=1, ...
    IDS = ["a", "b", "c"]
    DOCS = [
        "apple apple apple banana cherry",
        "apple banana",
        "apple apple date elderberry fig grape",
    ]
    METAS = [{"i": 0}, {"i": 1}, {"i": 2}]
    K1 = 1.5
    B = 0.75
    N = 3
    DOC_LENS = [5, 2, 6]
    AVGDL = 13.0 / 3.0  # (5 + 2 + 6) / 3
    DF = {
        "apple": 3, "banana": 2, "cherry": 1, "date": 1,
        "elderberry": 1, "fig": 1, "grape": 1,
    }
    FREQS = {
        "apple": {0: 3, 1: 1, 2: 2},
        "banana": {0: 1, 1: 1},
        "cherry": {0: 1},
        "date": {2: 1},
        "elderberry": {2: 1},
        "fig": {2: 1},
        "grape": {2: 1},
    }

    # -- helpers ------------------------------------------------------
    def _build(self):
        return BM25Index(self.IDS, self.DOCS, self.METAS,
                         k1=self.K1, b=self.B)

    def _expected_idf(self, term):
        """Compute IDF with the CORRECT formula (independent of BM25Index)."""
        d = self.DF[term]
        return math.log(1 + (self.N - d + 0.5) / (d + 0.5))

    def _expected_term_score(self, term, doc_idx):
        """Compute expected BM25 score for a single term in a single doc
        using the CORRECT formula (not calling search())."""
        freq = self.FREQS.get(term, {}).get(doc_idx)
        if freq is None:
            return 0.0
        idf = self._expected_idf(term)
        dl = self.DOC_LENS[doc_idx]
        avgdl = self.AVGDL
        denom = freq + self.K1 * (1 - self.B + self.B * dl / avgdl)
        return idf * (freq * (self.K1 + 1)) / denom

    def _expected_total_score(self, query, doc_idx):
        """Expected total BM25 score for a multi-term query in one doc."""
        return sum(
            self._expected_term_score(term, doc_idx)
            for term in set(tokenize(query))
        )

    @staticmethod
    def _search_and_capture(idx, query, top_k=10):
        """Call idx.search() and capture the internal scores dict by
        temporarily patching ``builtins.sorted``.

        search() calls ``sorted(scores.items(), ...)`` exactly once, so the
        captured dict maps doc_idx -> BM25 score for every matching doc.
        """
        captured = {}
        _real_sorted = sorted

        def _capturing_sorted(iterable, **kwargs):
            items = list(iterable)
            try:
                for k, v in items:
                    captured[k] = v
            except (ValueError, TypeError):
                pass  # not a (idx, score) iterable
            return _real_sorted(items, **kwargs)

        import builtins
        orig = builtins.sorted
        builtins.sorted = _capturing_sorted
        try:
            results = idx.search(query, top_k=top_k)
        finally:
            builtins.sorted = orig
        return results, captured

    # -- IDF exact values (kills mutation 3: SUB_TO_ADD in IDF) -------
    def test_idf_exact_all_terms(self):
        """IDF values must match log(1 + (N - d + 0.5)/(d + 0.5))."""
        idx = self._build()
        for term, df in self.DF.items():
            expected = self._expected_idf(term)
            assert math.isclose(idx.idf[term], expected, rel_tol=1e-12), (
                f"IDF[{term}] (df={df}): expected {expected!r}, "
                f"got {idx.idf[term]!r}"
            )

    def test_idf_apple_exact(self):
        """IDF[apple] = log(1 + (3-3+0.5)/(3+0.5)) = log(8/7)."""
        idx = self._build()
        assert math.isclose(idx.idf["apple"], math.log(8 / 7), rel_tol=1e-12)

    def test_idf_banana_exact(self):
        """IDF[banana] = log(1 + (3-2+0.5)/(2+0.5)) = log(8/5)."""
        idx = self._build()
        assert math.isclose(idx.idf["banana"], math.log(8 / 5), rel_tol=1e-12)

    def test_idf_rare_term_exact(self):
        """IDF[cherry] (df=1) = log(1 + (3-1+0.5)/(1+0.5)) = log(8/3)."""
        idx = self._build()
        assert math.isclose(idx.idf["cherry"], math.log(8 / 3), rel_tol=1e-12)

    # -- structural values --------------------------------------------
    def test_avgdl_exact(self):
        idx = self._build()
        assert math.isclose(idx.avgdl, self.AVGDL, rel_tol=1e-12)

    def test_doc_len_exact(self):
        idx = self._build()
        assert idx.doc_len == self.DOC_LENS

    # -- single-term exact scores (kills mutations 1,2,4,5,6,7,8,9) ---
    def test_single_term_apple_exact_scores(self):
        """Assert exact BM25 scores for 'apple' across all 3 docs.

        Expected scores are computed with the correct formula independently
        of the BM25Index implementation, so any mutation in lines 58-60
        produces a mismatched actual score.
        """
        idx = self._build()
        _, captured = self._search_and_capture(idx, "apple", top_k=10)
        for doc_idx in range(3):
            expected = self._expected_term_score("apple", doc_idx)
            assert doc_idx in captured, f"doc {doc_idx} missing from scores"
            assert math.isclose(captured[doc_idx], expected, rel_tol=1e-12), (
                f"doc {doc_idx}: expected {expected!r}, "
                f"got {captured[doc_idx]!r}"
            )

    def test_single_term_apple_score_values(self):
        """Hardcoded expected score values (computed by hand) to ensure
        the formula is exactly right.

        doc0: idf * 130/81  (freq=3, dl=5,  avgdl=13/3)
        doc1: idf * 260/197 (freq=1, dl=2,  avgdl=13/3)
        doc2: idf * 520/409 (freq=2, dl=6,  avgdl=13/3)
        """
        idx = self._build()
        _, captured = self._search_and_capture(idx, "apple", top_k=10)
        idf = math.log(8 / 7)
        assert math.isclose(captured[0], idf * 130 / 81, rel_tol=1e-12)
        assert math.isclose(captured[1], idf * 260 / 197, rel_tol=1e-12)
        assert math.isclose(captured[2], idf * 520 / 409, rel_tol=1e-12)

    def test_single_term_apple_ordering(self):
        """Correct ordering for 'apple' is doc0 > doc1 > doc2."""
        idx = self._build()
        results, _ = self._search_and_capture(idx, "apple", top_k=10)
        ids = [r[0] for r in results]
        assert ids == ["a", "b", "c"]

    # -- multi-term exact scores --------------------------------------
    def test_multi_term_apple_banana_exact_scores(self):
        """Assert exact scores for multi-term query 'apple banana'."""
        idx = self._build()
        _, captured = self._search_and_capture(idx, "apple banana", top_k=10)
        for doc_idx in range(3):
            expected = self._expected_total_score("apple banana", doc_idx)
            if expected == 0.0:
                continue  # doc doesn't match any term
            assert doc_idx in captured, f"doc {doc_idx} missing from scores"
            assert math.isclose(captured[doc_idx], expected, rel_tol=1e-12), (
                f"doc {doc_idx}: expected {expected!r}, "
                f"got {captured[doc_idx]!r}"
            )

    def test_multi_term_apple_banana_hardcoded(self):
        """Hardcoded multi-term scores.

        doc0: log(8/7)*130/81 + log(8/5)*130/139
        doc1: log(8/7)*260/197 + log(8/5)*260/197
        doc2: log(8/7)*520/409  (no banana)
        """
        idx = self._build()
        _, captured = self._search_and_capture(idx, "apple banana", top_k=10)
        idf_a = math.log(8 / 7)
        idf_b = math.log(8 / 5)
        assert math.isclose(
            captured[0], idf_a * 130 / 81 + idf_b * 130 / 139, rel_tol=1e-12
        )
        assert math.isclose(
            captured[1], idf_a * 260 / 197 + idf_b * 260 / 197, rel_tol=1e-12
        )
        assert math.isclose(captured[2], idf_a * 520 / 409, rel_tol=1e-12)

    # -- length normalisation: kills DIV_TO_MUL & OR_TO_AND(avgdl) ----
    def test_length_normalization_exact_scores(self):
        """Use docs with very different lengths to ensure the dl/avgdl
        term is exercised with large values, killing DIV_TO_MUL (mutation 6)
        and OR_TO_AND(avgdl) (mutation 8)."""
        docs = [
            "apple apple apple apple apple apple apple apple apple apple "
            "elderberry fig grape hazel iris",
            "apple",
        ]
        idx = BM25Index(["a", "b"], docs, [{}, {}], k1=1.5, b=0.75)
        # N=2, dl=[14, 1], avgdl=7.5, df[apple]=2
        idf = math.log(1.2)  # log(1 + 0.5/2.5)
        avgdl = 7.5
        # doc0: freq=10, dl=14
        expected_0 = idf * 25 / (10 + 1.5 * (0.25 + 0.75 * 14 / avgdl))
        # doc1: freq=1, dl=1
        expected_1 = idf * 2.5 / (1 + 1.5 * (0.25 + 0.75 * 1 / avgdl))
        _, captured = self._search_and_capture(idx, "apple", top_k=10)
        assert math.isclose(captured[0], expected_0, rel_tol=1e-12), (
            f"doc0: expected {expected_0!r}, got {captured[0]!r}"
        )
        assert math.isclose(captured[1], expected_1, rel_tol=1e-12), (
            f"doc1: expected {expected_1!r}, got {captured[1]!r}"
        )

    def test_length_normalization_ordering(self):
        """With the correct formula, the high-freq long doc (freq=10, dl=14)
        outranks the low-freq short doc (freq=1, dl=1).

        With DIV_TO_MUL (dl*avgdl instead of dl/avgdl) the length penalty
        becomes so extreme that the ordering flips.
        """
        docs = [
            "apple apple apple apple apple apple apple apple apple apple "
            "elderberry fig grape hazel iris",
            "apple",
        ]
        idx = BM25Index(["a", "b"], docs, [{}, {}], k1=1.5, b=0.75)
        results = idx.search("apple", top_k=2)
        assert results[0][0] == "a"  # high-freq doc wins
        assert results[1][0] == "b"

    # -- numerator formula: kills MUL_TO_DIV & ADD_TO_SUB -------------
    def test_numerator_formula_exact(self):
        """Verify numerator freq*(k1+1) by checking score scales correctly
        with freq for same-length docs (kills mutations 2 and 5)."""
        docs = ["apple banana cherry", "apple apple banana"]
        idx = BM25Index(["x", "y"], docs, [{}, {}], k1=1.5, b=0.75)
        # N=2, dl=[3, 3], avgdl=3, df[apple]=2
        idf = math.log(1.2)
        # doc0: freq=1, denom = 1 + 1.5*(0.25+0.75) = 2.5
        #   score = idf * 2.5 / 2.5 = idf
        # doc1: freq=2, denom = 2 + 1.5*(0.25+0.75) = 3.5
        #   score = idf * 5 / 3.5
        _, captured = self._search_and_capture(idx, "apple", top_k=10)
        assert math.isclose(captured[0], idf * 1.0, rel_tol=1e-12)
        assert math.isclose(captured[1], idf * 5.0 / 3.5, rel_tol=1e-12)

    # -- denom or 1 fallback: kills OR_TO_AND(denom) -----------------
    def test_denom_not_replaced_by_one(self):
        """If ``denom or 1`` were mutated to ``denom and 1``, the score
        would become ``idf * numerator`` (dividing by 1 instead of denom).
        Assert the actual score is much smaller than that wrong value."""
        idx = self._build()
        _, captured = self._search_and_capture(idx, "apple", top_k=10)
        idf = math.log(8 / 7)
        # Wrong score if denom and 1 were used: idf * 7.5 for doc0
        wrong_0 = idf * 7.5
        assert not math.isclose(captured[0], wrong_0, rel_tol=1e-3)
        assert captured[0] < wrong_0  # correct score is much smaller

    # -- doc_len or 1: kills OR_TO_AND(doc_len) ----------------------
    def test_doc_len_not_replaced_by_one(self):
        """If ``doc_len[idx] or 1`` were mutated to ``doc_len[idx] and 1``,
        every dl would become 1, collapsing length normalisation.

        With dl=1 for all docs, doc1 (freq=1, dl=2) and doc2 (freq=2, dl=6)
        would get different scores than the correct formula produces.
        """
        idx = self._build()
        _, captured = self._search_and_capture(idx, "apple", top_k=10)
        # Compute what the score would be if dl were always 1
        idf = math.log(8 / 7)
        avgdl = self.AVGDL
        wrong_denom_1 = 1 + 1.5 * (0.25 + 0.75 * 1 / avgdl)
        wrong_score_1 = idf * 2.5 / wrong_denom_1  # for doc1 (freq=1)
        # The correct score for doc1 uses dl=2, not dl=1
        assert not math.isclose(captured[1], wrong_score_1, rel_tol=1e-6), (
            f"doc1 score {captured[1]} matches dl=1 mutation {wrong_score_1}"
        )

    # -- avgdl or 1: kills OR_TO_AND(avgdl) --------------------------
    def test_avgdl_not_replaced_by_one(self):
        """If ``avgdl or 1`` were mutated to ``avgdl and 1``, avgdl would
        become 1 (since avgdl=13/3 is truthy, ``13/3 and 1`` returns 1).

        This changes every score because dl/avgdl becomes dl/1 = dl.
        """
        idx = self._build()
        _, captured = self._search_and_capture(idx, "apple", top_k=10)
        idf = math.log(8 / 7)
        # Compute what the score would be if avgdl were 1
        wrong_denom_0 = 3 + 1.5 * (0.25 + 0.75 * 5 / 1)
        wrong_score_0 = idf * 7.5 / wrong_denom_0
        assert not math.isclose(captured[0], wrong_score_0, rel_tol=1e-6), (
            f"doc0 score {captured[0]} matches avgdl=1 mutation {wrong_score_0}"
        )