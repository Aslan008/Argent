"""Ctrl+C должен останавливать команду, которая уже выполняется.

Раньше отмена ставила флаг, который читал только цикл агента между
инструментами. Уже запущенный `run_command` его не видел, и остановка во время
сборки означала «дождись сборки». Здесь проверяется, что команда замечает
отмену сама и возвращается быстро, а не через свой таймаут.
"""

import threading
import time

import pytest

from argent.tools import command_ops


@pytest.fixture(autouse=True)
def _reset_generation():
    # Тесты не должны зависеть от порядка запуска: поколение только растёт.
    yield


def test_generation_only_grows():
    before = command_ops.cancel_generation()
    after = command_ops.bump_cancel_generation()
    assert after == before + 1
    assert command_ops.cancel_generation() == after


def test_a_running_command_stops_when_the_turn_is_cancelled(monkeypatch):
    # Подтверждение — интерактивный вопрос; в тесте отвечать некому, и без
    # подмены команда просто не запустится (это не отказ отмены, а окружение).
    monkeypatch.setattr(command_ops, "request_approval", lambda *a, **k: True)

    # Спит 30 секунд: без отмены тест не успел бы вернуться.
    command = 'python -c "import time; time.sleep(30)"'
    result = {}

    def run():
        started = time.time()
        result["output"] = command_ops.run_command(command, timeout=25)
        result["elapsed"] = time.time() - started

    worker = threading.Thread(target=run, daemon=True)
    worker.start()

    # Дать процессу запуститься, затем попросить остановку — как Ctrl+C.
    time.sleep(1.0)
    command_ops.bump_cancel_generation()

    worker.join(timeout=20)
    assert not worker.is_alive(), "команда не остановилась по отмене"

    assert result["elapsed"] < 15, f"остановка заняла {result['elapsed']:.1f}s"
    assert "[STOPPED BY USER]" in result["output"]
