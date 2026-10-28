from unittest.mock import MagicMock, patch

from argent.providers import DeepSeekProvider, OpenRouterProvider
from argent.usage_tracker import SessionUsage


class TestSessionBalance:
    def test_balance_alone_is_shown_before_any_request(self):
        u = SessionUsage()
        u.set_balance(3.85, "USD")
        assert u.format_session() == "bal $3.85"

    def test_cost_and_balance_can_both_be_shown(self):
        u = SessionUsage()
        u.add({"prompt": 500, "completion": 300, "cost": 0.02})
        u.set_balance(3.85, "USD")
        assert "sum 800 tok" in u.format_session()
        assert "$0.02" in u.format_session()
        assert "bal $3.85" in u.format_session()

    def test_balance_drop_is_account_spend(self):
        """Account spend is shown again, but as approximate."""
        u = SessionUsage()
        u.set_balance(3.85, "USD", provider="deepseek")
        u.set_balance(3.80, "USD", provider="deepseek")
        s = u.format_session()
        assert "spent~ $0.05" in s
        assert "bal $3.80" in s

    def test_balance_switch_does_not_leave_stale_spend(self):
        u = SessionUsage()
        u.set_balance(3.85, "USD", provider="deepseek")
        u.set_balance(10.00, "USD", provider="openrouter")
        assert u.balance == 10.00
        assert "spent~" not in u.format_session()

    def test_non_usd_currency_is_labeled(self):
        u = SessionUsage()
        u.set_balance(12.5, "CNY")
        assert "bal CNY 12.50" in u.format_session()

    def test_clear_balance_for_provider_switch(self):
        u = SessionUsage()
        u.set_balance(3.85, "USD", provider="deepseek")
        u.clear_balance("openrouter")
        assert u.balance is None
        assert "bal" not in u.format_session()

    def test_default_deepseek_model_is_current(self):
        p = DeepSeekProvider(api_key="sk-test", base_url="https://api.deepseek.com")
        assert p.default_model() == "deepseek-flash"


class TestDeepSeekBalance:
    def test_parses_usd_balance(self):
        p = DeepSeekProvider(api_key="sk-test", base_url="https://api.deepseek.com")
        response = MagicMock()
        response.ok = True
        response.json.return_value = {
            "is_available": True,
            "balance_infos": [
                {"currency": "CNY", "total_balance": "28.00"},
                {"currency": "USD", "total_balance": "3.85"},
            ],
        }
        with patch("requests.get", return_value=response) as get:
            assert p.get_balance() == (3.85, "USD")
            assert "user/balance" in get.call_args.args[0]

    def test_http_error_is_silent(self):
        p = DeepSeekProvider(api_key="sk-test", base_url="https://api.deepseek.com")
        response = MagicMock()
        response.ok = False
        with patch("requests.get", return_value=response):
            assert p.get_balance() is None

    def test_without_key_is_silent(self):
        p = DeepSeekProvider(api_key=None, base_url="https://api.deepseek.com")
        assert p.get_balance() is None


class TestOpenRouterBalance:
    def test_remaining_is_credits_minus_usage(self):
        p = OpenRouterProvider(api_key="sk-or-test", base_url="https://openrouter.ai/api/v1")
        response = MagicMock()
        response.ok = True
        response.json.return_value = {
            "data": {"total_credits": 10.0, "total_usage": 2.5}
        }
        with patch("requests.get", return_value=response) as get:
            assert p.get_balance() == (7.5, "USD")
            assert get.call_args.args[0].endswith("/credits")
