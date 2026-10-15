# -*- coding: utf-8 -*-
"""scripts/_llm_urgency_probe.py — проверка urgency на рукописных кейсах (13 шт)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.leeth import leeth
from src.leeth.primitives import ScoreQuestion
from _multidomain_battery import Q_URGENCY, URGENCY_RUBRIC, prod_envelope

leeth.initialize(force_device="cuda")

cases = [
    ("Срочно! Сервис лежит, клиенты не могут оформить заказ, теряем деньги каждую минуту.", 3),
    ("У нас аудит завтра, нужны все документы сегодня.", 3),
    ("Не работает кнопка экспорта, но это не срочно, посмотрите когда будет время.", 1),
    ("Подскажите, как добавить нового сотрудника в аккаунт? Не горит.", 1),
    ("Утечка данных клиентов, логи показывают подозрительную активность. Срочно!", 3),
    ("Пришлите, пожалуйста, прайс на следующую неделю.", 1),
    ("Ключевой клиент расторгает договор из-за сбоя — нужен звонок руководства сейчас.", 3),
    ("Напомните, пожалуйста, до конца недели о продлении договора.", 2),
    ("Сервер баз данных не отвечает, производство остановлено. Реагируйте немедленно!", 3),
    ("Уточните, пожалуйста, статус нашего тикета — ждём ответа со вторника.", 2),
    ("Инцидент безопасности: подозрительный вход в аккаунт администратора. Реагируйте сейчас.", 3),
    ("Согласуйте, пожалуйста, дату технического окна на следующей неделе.", 2),
    ("Срочно: у нас аудит ФНС завтра, нужны все документы по счетам сегодня.", 3),
]
q = ScoreQuestion(id="urgency", question=Q_URGENCY, levels=3, rubric=URGENCY_RUBRIC)
ok = 0
for text, gold in cases:
    b = leeth.decide(prod_envelope(text), [q])
    r = b.results["urgency"]
    level = int(round(r.score / 100 * 3)) or 1
    mark = "OK " if level == gold else "FAIL"
    ok += level == gold
    print(f"{mark} gold={gold} level={level}/3 score={r.score:.1f} conf={r.confidence:.2f} | {text[:50]}")
print(f"итог: {ok}/{len(cases)}")