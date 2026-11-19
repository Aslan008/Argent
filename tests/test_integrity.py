import os

import pytest

from argent.integrity import integrity_label, label_for_rid


def test_known_integrity_rids():
    assert label_for_rid(0x1000) == "Low"
    assert label_for_rid(0x2000) == "Medium"
    assert label_for_rid(0x3000) == "High"
    assert label_for_rid(0x9999).startswith("Unknown")


@pytest.mark.skipif(os.name != "nt", reason="Windows token only")
def test_the_current_process_reports_a_label_or_unknown():
    assert integrity_label() in {"Low", "Medium", "High", "System", "Untrusted", None}
