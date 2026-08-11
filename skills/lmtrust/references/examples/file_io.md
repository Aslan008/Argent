# Пример 3: Файловый I/O — `save_game` / `load_game`

## Исходный код

```python
import json

def save_game(player, filepath):
    data = {"hp": player.hp, "shield": player.shield, "state": player.state}
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f)

def load_game(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data
```

## Дерево решений: Файловый I/O → L0, L1, L2, L6, L4b × D3, D5, Known Coordinates

## Tests

```python
import os
import json
import tempfile
import pytest

# L0: Smoke
class TestSmoke:
    def test_save_creates_file(self):
        player = Player(hp=100, shield=30)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as f:
            path = f.name
        try:
            save_game(player, path)
            assert os.path.exists(path)
        finally:
            if os.path.exists(path):
                os.unlink(path)

# L1: Contract + Round-trip
class TestContract:
    def test_save_load_identity(self):
        """save → load возвращает те же данные."""
        player = Player(hp=75, shield=30)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as f:
            path = f.name
        try:
            save_game(player, path)
            data = load_game(path)
            assert data["hp"] == 75
            assert data["shield"] == 30
            assert data["state"] == "alive"
        finally:
            if os.path.exists(path):
                os.unlink(path)

# L2: Boundary (8 условий)
class TestEdgeCases:
    def test_empty_file(self):
        """Пустой файл — должно упасть явно, не молча."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            f.write('')
            path = f.name
        try:
            with pytest.raises(json.JSONDecodeError):
                load_game(path)
        finally:
            os.unlink(path)

    def test_crlf_line_endings(self):
        """Windows CRLF не должен ломать парсинг (AP6)."""
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.json') as f:
            f.write(b'{"hp": 100, "shield": 50, "state": "alive"}\r\n')
            path = f.name
        try:
            data = load_game(path)
            assert data["hp"] == 100
        finally:
            os.unlink(path)

    def test_bom_prefix(self):
        """BOM в начале файла — не должен ломать парсинг."""
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.json') as f:
            f.write(b'\xef\xbb\xbf{"hp": 100, "shield": 50, "state": "alive"}')
            path = f.name
        try:
            data = load_game(path)
            assert data["hp"] == 100
        finally:
            os.unlink(path)

    def test_single_field(self):
        """Минимальный валидный JSON — один ключ."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            json.dump({"hp": 100}, f)
            path = f.name
        try:
            data = load_game(path)
            assert "hp" in data
        finally:
            os.unlink(path)

    def test_malformed_json(self):
        """Невалидный JSON — явная ошибка, не молчаливый None."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            f.write('{broken json')
            path = f.name
        try:
            with pytest.raises(json.JSONDecodeError):
                load_game(path)
        finally:
            os.unlink(path)

# L6: Integration (No-Mock E2E)
class TestIntegration:
    def test_full_save_load_cycle(self):
        """Полный цикл save → load → восстановление Player — без моков."""
        player = Player(hp=80, shield=25)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as f:
            path = f.name
        try:
            save_game(player, path)
            data = load_game(path)
            restored = Player(hp=data["hp"], shield=data["shield"])
            assert restored.hp == player.hp
            assert restored.shield == player.shield
        finally:
            if os.path.exists(path):
                os.unlink(path)

# L4b: Fallback
class TestFallback:
    def test_load_nonexistent_file(self):
        """Несуществующий файл — явная ошибка, не молчаливый None."""
        with pytest.raises(FileNotFoundError):
            load_game("/nonexistent/path/file.json")

    def test_save_to_nonexistent_dir(self):
        """Сохранение в несуществующую директорию — явная ошибка."""
        player = Player(hp=100)
        with pytest.raises((FileNotFoundError, OSError)):
            save_game(player, "/nonexistent/dir/file.json")
```