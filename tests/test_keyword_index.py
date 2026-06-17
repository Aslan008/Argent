from src.rag.keyword_index import BM25Index, tokenize, get_index, clear_cache, _CACHE


DOCS = [
    "Unity API: Rigidbody.AddForce  Adds a force to the rigidbody along a vector.",
    "Unity API: Rigidbody.AddTorque  Adds a torque to the rigidbody around an axis.",
    "Unity API: Transform.Translate  Moves the transform by a translation vector.",
    "Unity API: Collider.OnCollisionEnter  Called when this collider starts touching another.",
]
IDS = ["a", "b", "c", "d"]
METAS = [{"symbol": s} for s in ("Rigidbody.AddForce", "Rigidbody.AddTorque",
                                 "Transform.Translate", "Collider.OnCollisionEnter")]


class TestTokenize:
    def test_drops_short_tokens_and_lowercases(self):
        assert tokenize("The AddForce a Vector3") == ["the", "addforce", "vector3"]


class TestBM25:
    def test_ranks_relevant_doc_first(self):
        idx = BM25Index(IDS, DOCS, METAS)
        res = idx.search("how does AddForce work", top_k=2)
        assert res[0][0] == "a"  # AddForce doc ranks first

    def test_torque_query_ranks_torque(self):
        idx = BM25Index(IDS, DOCS, METAS)
        res = idx.search("add torque around axis", top_k=1)
        assert res[0][2]["symbol"] == "Rigidbody.AddTorque"

    def test_no_match_returns_empty(self):
        idx = BM25Index(IDS, DOCS, METAS)
        assert idx.search("photosynthesis quantum chromodynamics", top_k=5) == []

    def test_returns_id_doc_meta_tuples(self):
        idx = BM25Index(IDS, DOCS, METAS)
        rid, doc, meta = idx.search("translate transform", top_k=1)[0]
        assert rid == "c" and "Translate" in doc and meta["symbol"] == "Transform.Translate"


class FakeCol:
    def __init__(self, name, ids, docs, metas):
        self.name = name
        self._data = {"ids": ids, "documents": docs, "metadatas": metas}
        self.get_calls = 0

    def count(self):
        return len(self._data["ids"])

    def get(self, include=None):
        self.get_calls += 1
        return self._data


class TestCache:
    def setup_method(self):
        clear_cache()

    def test_index_built_once_and_cached(self):
        col = FakeCol("kb_unity", IDS, DOCS, METAS)
        get_index(col)
        get_index(col)
        get_index(col)
        assert col.get_calls == 1  # built once, then served from cache

    def test_count_change_rebuilds(self):
        col = FakeCol("kb_x", IDS, DOCS, METAS)
        get_index(col)
        col._data["ids"] = IDS + ["e"]
        col._data["documents"] = DOCS + ["new doc about lighting"]
        col._data["metadatas"] = METAS + [{"symbol": "Light"}]
        get_index(col)
        assert col.get_calls == 2  # rebuilt because count changed

    def test_clear_cache_forces_rebuild(self):
        col = FakeCol("kb_y", IDS, DOCS, METAS)
        get_index(col)
        clear_cache()
        get_index(col)
        assert col.get_calls == 2
