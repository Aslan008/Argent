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
