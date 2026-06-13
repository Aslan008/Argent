from usage_tracker import SessionUsage, _short


class TestShort:
    def test_under_thousand(self):
        assert _short(999) == "999"

    def test_thousands(self):
        assert _short(1234) == "1.2k"
        assert _short(2000) == "2k"

    def test_millions(self):
        assert _short(1_500_000) == "1.5M"


class TestSessionUsage:
    def test_accumulates(self):
        u = SessionUsage()
        u.add({"prompt": 100, "completion": 50, "cost": 0.001})
        u.add({"prompt": 200, "completion": 25, "cost": 0.002})
        assert u.prompt == 300
        assert u.completion == 75
        assert round(u.cost, 4) == 0.003
        assert u.requests == 2

    def test_handles_missing_cost(self):
        u = SessionUsage()
        u.add({"prompt": 10, "completion": 5})
        assert u.cost == 0.0

    def test_reset(self):
        u = SessionUsage()
        u.add({"prompt": 10, "completion": 5, "cost": 0.5})
        u.reset()
        assert u.prompt == 0 and u.cost == 0.0 and u.requests == 0

    def test_format_last_tokens_only(self):
        s = SessionUsage.format_last({"prompt": 1200, "completion": 567})
        assert "1.2k" in s and "567" in s
        assert "$" not in s  # no cost shown when zero

    def test_format_last_with_cost(self):
        s = SessionUsage.format_last({"prompt": 100, "completion": 50, "cost": 0.0021})
        assert "$0.0021" in s

    def test_format_is_ascii_safe(self):
        # Rendered through Rich on legacy cp1251 consoles — must be encodable.
        s = SessionUsage.format_last({"prompt": 1200, "completion": 567, "cost": 0.0021})
        s.encode("cp1251", errors="strict")
        SessionUsage().format_session().encode("cp1251", errors="strict")

    def test_format_session(self):
        u = SessionUsage()
        u.add({"prompt": 1000, "completion": 1000, "cost": 0.05})
        s = u.format_session()
        assert "2k" in s
        assert "$0.05" in s
