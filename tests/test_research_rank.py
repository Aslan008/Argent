"""Structure-aware chunking + cross-encoder reranking.

The reranker's scorer is injected, so no model is downloaded or loaded here.
"""

from src.research import rerank as rr
from src.research.chunking import chunk_document
from src.research.rerank import rerank


class TestChunkDocument:
    def test_empty(self):
        assert chunk_document("") == []
        assert chunk_document("   \n  ") == []

    def test_short_single_chunk(self):
        assert chunk_document("hello world") == ["hello world"]

    def test_packs_paragraphs_with_overlap(self):
        p1, p2, p3 = "x" * 400, "y" * 400, "z" * 400
        chunks = chunk_document(f"{p1}\n\n{p2}\n\n{p3}", target_size=1000, overlap_units=1)
        assert len(chunks) == 2
        # p2 is the overlap unit shared between the two chunks.
        assert p2 in chunks[0] and p2 in chunks[1]
        assert p1 in chunks[0] and p3 in chunks[1]

    def test_never_exceeds_target_by_packing(self):
        paras = "\n\n".join("w" * 300 for _ in range(5))
        chunks = chunk_document(paras, target_size=1000)
        # No packed chunk should blow well past the target (one 300-unit slack).
        assert all(len(c) <= 1000 + 300 for c in chunks)

    def test_oversized_unit_hard_split(self):
        chunks = chunk_document("w" * 2500, target_size=1000, overlap_units=1)
        assert len(chunks) == 3
        assert all(len(c) <= 1000 for c in chunks)
        assert sum(len(c) for c in chunks) == 2500  # no duplication from overlap


class TestRerank:
    def test_empty(self):
        assert rerank("q", []) == []

    def test_orders_by_score_desc(self):
        scorer = lambda pairs: [0.1, 0.9, 0.5]  # b > c > a
        assert rerank("q", ["a", "b", "c"], top_n=3, _scorer=scorer) == ["b", "c", "a"]

    def test_respects_top_n(self):
        scorer = lambda pairs: [1, 2, 3, 4]
        assert rerank("q", ["a", "b", "c", "d"], top_n=2, _scorer=scorer) == ["d", "c"]

    def test_scorer_receives_query_chunk_pairs(self):
        seen = {}

        def scorer(pairs):
            seen["pairs"] = pairs
            return [1, 2]
        rerank("myq", ["x", "y"], _scorer=scorer)
        assert seen["pairs"] == [("myq", "x"), ("myq", "y")]

    def test_falls_back_to_bi_encoder(self, monkeypatch):
        def bad_scorer(pairs):
            raise RuntimeError("model gone")
        monkeypatch.setattr(rr, "_bi_encoder_rank", lambda q, c, n: ["BI-RESULT"])
        assert rerank("q", ["a", "b"], _scorer=bad_scorer) == ["BI-RESULT"]

    def test_falls_back_to_input_order(self, monkeypatch):
        def bad(*a, **k):
            raise RuntimeError("x")
        monkeypatch.setattr(rr, "_bi_encoder_rank", bad)
        assert rerank("q", ["a", "b", "c"], top_n=2, _scorer=bad) == ["a", "b"]
