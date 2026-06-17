import builtins

import rag_engine


class TestReadPdf:
    def test_missing_file_returns_none(self):
        # Never raises, never returns garbage — just None to skip the file.
        assert rag_engine._read_pdf("C:/no/such/file.pdf") is None

    def test_no_parser_returns_none(self, monkeypatch):
        real_import = builtins.__import__

        def no_pdf(name, *args, **kwargs):
            if name in ("pypdf", "PyPDF2"):
                raise ImportError(name)
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_pdf)
        assert rag_engine._read_pdf("anything.pdf") is None
