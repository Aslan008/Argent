import pytest

pytest.importorskip("bs4")

from src.rag.unity_docs import is_unity_doc, clean_unity_html, chunk_unity_doc


# A representative Unity ScriptReference page (trimmed) with the real boilerplate.
API_PAGE = """<!DOCTYPE html><html><head>
<meta name="author" content="Unity Technologies" />
<link rel="canonical" href="https://docs.unity3d.com/6000.4/Documentation/ScriptReference/Rigidbody.AddForce.html" />
<title>Unity - Scripting API: Rigidbody.AddForce</title>
<script>var x = 1;</script><style>.a{}</style>
</head><body>
<div class="header"><div class="menu"><div class="search-form"><form><input/></form></div></div></div>
<nav class="breadcrumb">Home > Rigidbody</nav>
<div class="content"><div class="section">
<h1>Rigidbody.AddForce</h1>
<div class="feedback">Leave feedback</div>
<div class="suggest">Suggest a change Success! Thank you for helping us improve the quality of Unity Documentation. Close</div>
<p>public void AddForce(Vector3 force, ForceMode mode = ForceMode.Force);</p>
<h2>Parameters</h2>
<p>force: Force vector in world coordinates.</p>
<h2>Description</h2>
<p>Adds a force to the Rigidbody. Force is applied continuously along the direction of the force vector.</p>
</div></div>
<footer class="footer">Copyright Unity</footer>
</body></html>"""

MANUAL_PAGE = """<html><head><meta name="author" content="Unity Technologies"/>
<title>Unity - Manual: Rigidbody physics</title></head><body>
<div class="menu">nav noise</div>
<div class="section"><h1>Introduction to rigid body physics</h1>
<p>A rigid body is a physical body that does not deform under forces.</p></div></body></html>"""


class TestDetection:
    def test_unity_page_detected(self):
        assert is_unity_doc(API_PAGE)

    def test_non_unity_not_detected(self):
        assert not is_unity_doc("<html><body><h1>My blog</h1></body></html>")


class TestCleaner:
    def test_api_symbol_extracted_tight(self):
        sym, _ = clean_unity_html(API_PAGE)
        assert sym == "Rigidbody.AddForce"

    def test_manual_title_kept_readable(self):
        sym, _ = clean_unity_html(MANUAL_PAGE)
        assert sym == "Introduction to rigid body physics"

    def test_boilerplate_stripped(self):
        _, text = clean_unity_html(API_PAGE)
        assert "Leave feedback" not in text
        assert "Thank you for helping us improve" not in text
        assert "nav noise" not in text
        assert "Copyright Unity" not in text
        assert "Home > Rigidbody" not in text

    def test_real_content_kept(self):
        _, text = clean_unity_html(API_PAGE)
        assert "AddForce(Vector3 force" in text
        assert "Force vector in world coordinates" in text
        assert "Adds a force to the Rigidbody" in text


class TestChunking:
    def test_symbol_prefixed(self):
        chunks = chunk_unity_doc(API_PAGE, "ScriptReference/Rigidbody.AddForce.html")
        assert chunks
        assert chunks[0][0].startswith("Unity API: Rigidbody.AddForce")
        assert chunks[0][1]["symbol"] == "Rigidbody.AddForce"

    def test_large_page_splits(self):
        big = API_PAGE.replace("Adds a force to the Rigidbody.",
                               "Adds a force. " + "Detail sentence. " * 400)
        chunks = chunk_unity_doc(big, "x.html", max_chars=800)
        assert len(chunks) > 1
        # Every chunk carries the symbol prefix for retrieval.
        assert all(c.startswith("Unity API: Rigidbody.AddForce") for c, _ in chunks)

    def test_empty_html_no_chunks(self):
        assert chunk_unity_doc("<html><body></body></html>", "e.html") == []
