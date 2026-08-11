# Пример 4: Внешний API — HTTP-клиент с retry/fallback

## Исходный код

```python
import requests

def fetch_player_data(player_id, timeout=5, retries=3):
    for attempt in range(retries):
        try:
            response = requests.get(
                f"https://api.game.com/players/{player_id}",
                timeout=timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException:
            if attempt == retries - 1:
                return None  # fallback
    return None
```

## Дерево решений: Внешний API → L0, L1, L4b, L7, L8 × D4, D5, D6

> **⚠️ Исключение из AP1:** Для внешних API моки транспортного слоя —
> необходимость. Мы мокаем `requests.get` (транспорт), но тестируем реальную
> логику retry/fallback (количество попыток, возвращаемое значение, обработку
> ошибок). L6 No-Mock E2E неприменим к внешним сервисам.

## Tests

```python
import pytest
from unittest.mock import patch, MagicMock
import requests

# L0: Smoke
class TestSmoke:
    def test_function_exists(self):
        assert callable(fetch_player_data)

# L1: Contract
class TestContract:
    def test_returns_data_on_success(self):
        with patch('requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"id": 123, "hp": 100}
            mock_get.return_value = mock_response

            result = fetch_player_data(123)
            assert result == {"id": 123, "hp": 100}

# L4b: Fallback
class TestFallback:
    def test_returns_none_after_all_retries(self):
        """Все попытки провалились → fallback возвращает None."""
        with patch('requests.get') as mock_get:
            mock_get.side_effect = requests.ConnectionError()
            result = fetch_player_data(123, retries=3)
            assert result is None
            assert mock_get.call_count == 3  # ровно 3 попытки

    def test_retries_on_timeout_then_succeeds(self):
        """Timeout → retry → успех."""
        with patch('requests.get') as mock_get:
            timeout_resp = requests.Timeout()
            success = MagicMock()
            success.status_code = 200
            success.json.return_value = {"id": 123}
            mock_get.side_effect = [timeout_resp, timeout_resp, success]

            result = fetch_player_data(123, retries=3)
            assert result == {"id": 123}
            assert mock_get.call_count == 3

    def test_succeeds_on_first_try(self):
        """Успех с первой попытки — без лишних retry."""
        with patch('requests.get') as mock_get:
            success = MagicMock()
            success.status_code = 200
            success.json.return_value = {"id": 123}
            mock_get.return_value = success

            result = fetch_player_data(123, retries=3)
            assert result == {"id": 123}
            assert mock_get.call_count == 1  # не было retry

# L7: Compositional Failure
class TestCompositional:
    def test_timeout_then_connection_error(self):
        """Два разных сбоя подряд — fallback срабатывает."""
        with patch('requests.get') as mock_get:
            mock_get.side_effect = [
                requests.Timeout(),
                requests.ConnectionError(),
                requests.ConnectionError(),
            ]
            result = fetch_player_data(123, retries=3)
            assert result is None

    def test_http_error_then_timeout(self):
        """HTTP 500 → timeout → timeout — fallback."""
        with patch('requests.get') as mock_get:
            http_error = MagicMock()
            http_error.raise_for_status.side_effect = requests.HTTPError("500")
            mock_get.side_effect = [
                http_error,
                requests.Timeout(),
                requests.Timeout(),
            ]
            result = fetch_player_data(123, retries=3)
            assert result is None

# L8: Temporal
class TestTemporal:
    def test_retry_count_respected(self):
        """Ровно N попыток, не больше."""
        with patch('requests.get') as mock_get:
            mock_get.side_effect = requests.ConnectionError()
            fetch_player_data(123, retries=5)
            assert mock_get.call_count == 5

    def test_zero_retries(self):
        """0 retry — ни одной попытки, сразу None."""
        with patch('requests.get') as mock_get:
            mock_get.side_effect = requests.ConnectionError()
            result = fetch_player_data(123, retries=0)
            assert result is None
            assert mock_get.call_count == 0

    def test_one_retry(self):
        """1 retry — одна попытка, затем None."""
        with patch('requests.get') as mock_get:
            mock_get.side_effect = requests.ConnectionError()
            result = fetch_player_data(123, retries=1)
            assert result is None
            assert mock_get.call_count == 1
```