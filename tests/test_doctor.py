import pytest

import doctor


@pytest.fixture(autouse=True)
def stub_network_checks(monkeypatch):
    # Hermetic unit tests: the provider check performs real network I/O.
    checks = [
        (name, fn) if name != "Provider" else (name, lambda: (doctor.OK, "stubbed"))
        for name, fn in doctor.CHECKS
    ]
    monkeypatch.setattr(doctor, "CHECKS", checks)


class TestRunDiagnostics:
    def test_never_raises_and_covers_all_checks(self):
        results = doctor.run_diagnostics()
        assert len(results) == len(doctor.CHECKS)
        for name, status, detail in results:
            assert status in (doctor.OK, doctor.WARN, doctor.FAIL)
            assert isinstance(detail, str) and detail

    def test_crashing_check_reports_fail_instead_of_raising(self, monkeypatch):
        def boom():
            raise RuntimeError("kaput")
        monkeypatch.setattr(doctor, "CHECKS", [("Boom", boom)])
        results = doctor.run_diagnostics()
        assert results == [("Boom", doctor.FAIL, "check crashed: kaput")]

    def test_details_are_ascii_safe(self):
        # Legacy Windows consoles (cp1251) must be able to render the table.
        for name, status, detail in doctor.run_diagnostics():
            detail.encode("cp1251", errors="strict")


class TestBundledChromium:
    """The Chromium revision is pinned to the installed Playwright; a stale download must be reported in words."""

    def _make(self, tmp_path, monkeypatch, installed):
        import browser_detect
        monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(tmp_path))
        needed = browser_detect.bundled_chromium_status()["needed"]
        for folder in installed(needed):
            (tmp_path / folder).mkdir()
            (tmp_path / folder / "INSTALLATION_COMPLETE").write_text("")
        return browser_detect.bundled_chromium_status()

    def test_missing_revision_is_listed_with_what_is_installed(self, tmp_path, monkeypatch):
        pytest.importorskip("playwright")
        st = self._make(tmp_path, monkeypatch, lambda needed: ["chromium-1"])
        assert st["known"] and st["missing"]
        assert "chromium-1" in st["installed"]

    def test_complete_install_has_nothing_missing(self, tmp_path, monkeypatch):
        pytest.importorskip("playwright")
        st = self._make(tmp_path, monkeypatch, lambda needed: [
            f"{n.replace('-', '_') if 'shell' in n else n}-{r}" for n, r in needed.items()])
        assert st["known"] and st["missing"] == []

    def test_doctor_warns_and_names_the_command(self, tmp_path, monkeypatch):
        pytest.importorskip("playwright")
        monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(tmp_path))
        status, detail = doctor._check_browser()
        assert status == doctor.WARN
        assert "playwright install chromium" in detail
        detail.encode("cp1251", errors="strict")
