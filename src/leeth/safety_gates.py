"""src/leeth/safety_gates.py — Фаза F5/G4: расширенные предохранители.

Зачем: плагины исполняют код, браузерные клики имеют необратимые последствия.
Вероятностная модель НЕ должна единолично решать такие действия — поверх неё
стоят процедурные гейты, переиспользующие CDCL-арбитр и deny-паттерны.

Реализация БЕЗ изменения arbitrator.py: переиспользуем CDCLArbitrator
(предложение «run_command с этой командой» проверяется тем же солвером —
UNSAT = блок) + добавляем доменные паттерны для браузера.
"""

import re
from dataclasses import dataclass
from typing import Optional

from src.leeth.arbitrator import CDCLArbitrator, get_arbitrator
from src.leeth.dispatcher import DENY_PATTERNS


@dataclass
class GateVerdict:
    allowed: bool
    requires_confirmation: bool = False
    reason: Optional[str] = None
    source: str = "procedural"   # procedural | cdcl


# ── Плагины: команды, которые плагин просит исполнить ────────────────────────
PLUGIN_EXTRA_DENY = [
    # Плагин-команды не должны трогать сам Argent (самомодификация).
    re.compile(r"plugins[/\\][\w\-]+\.py", re.I),
    re.compile(r"\b(?:pip|npm)\s+(?:uninstall|remove)\b", re.I),
    re.compile(r"\b(?:git\s+push\s+--force|git\s+reset\s+--hard)\b", re.I),
]

# ── Браузер: опасные действия (дублируем browser_router для CLI-гейта) ───────
BROWSER_DANGEROUS = [
    re.compile(r"\b(?:delete|remove|erase|удалить)\b", re.I),
    re.compile(r"\b(?:pay|payment|checkout|purchase|buy now|оплатить|купить)\b", re.I),
    re.compile(r"\bconfirm\b|\bподтвердить\b", re.I),
    re.compile(r"\bauthorize\b|\bauthorise\b", re.I),
]


def gate_plugin_command(command: str) -> GateVerdict:
    """Гейт для команды, которую плагин хочет исполнить.

    Зачем: плагин — сторонний код; его запросы на исполнение проходят через
    тот же CDCL-арбитр, что и пользовательские run_command (UNSAT = блок),
    плюс дополнительные plugin-специфичные запреты.
    """
    if not command or not command.strip():
        return GateVerdict(allowed=False, reason="empty_command")
    text = command.strip()
    for pattern in PLUGIN_EXTRA_DENY:
        if pattern.search(text):
            return GateVerdict(allowed=False, reason=f"plugin_deny:{pattern.pattern[:30]}", source="procedural")
    for pattern in DENY_PATTERNS:
        if pattern.search(text):
            return GateVerdict(allowed=False, reason=f"deny_list:{pattern.pattern[:30]}", source="procedural")
    # CDCL: предложение «run_command с этой командой» против жёстких клауз.
    # log_shadow=False: гейт — не диспетчеризация, в shadow-статистику не пишем.
    try:
        result = get_arbitrator().arbitrate(
            tool="run_command", need_tool=True, param_text=text, query=text,
            log_shadow=False,
        )
        if not result.sat:
            # core защищён от None/нестрок: сбой reason не должен превращать
            # UNSAT (блок) в разрешение через общий except.
            core = [str(c) for c in (result.core or [])]
            return GateVerdict(allowed=False, reason="cdcl_unsat:" + ";".join(core[:2]), source="cdcl")
        return GateVerdict(allowed=True, source="cdcl")
    except Exception:
        # Fail-soft (дизайн-решение): солвер недоступен — процедурный гейт
        # уже отработал выше. Явно маркируем пропуск CDCL для телеметрии.
        return GateVerdict(allowed=True, reason="cdcl_skipped:solver_error", source="procedural")


def gate_browser_action(element_text: str, action: str = "click") -> GateVerdict:
    """Гейт для браузерного действия над элементом.

    Зачем: клик по «Confirm purchase» не должен происходить автоматически
    даже если ranking-голова уверена — требуется подтверждение человека.
    """
    if not element_text:
        return GateVerdict(allowed=False, reason="empty_element")
    # ПОРЯДОК ВАЖЕН: deny-паттерны (блок) раньше dangerous-слов (подтверждение).
    # Иначе «Delete all files (rm -rf /)» матчится как dangerous -> allowed=True
    # с подтверждением, и deny-лист никогда не доходит до блока.
    for pattern in DENY_PATTERNS:
        # Опасная команда, «спрятанная» в тексте кнопки/ссылки.
        if pattern.search(element_text):
            return GateVerdict(allowed=False, reason=f"deny_list:{pattern.pattern[:30]}", source="procedural")
    for pattern in BROWSER_DANGEROUS:
        # Матчим ТОЛЬКО текст элемента: метка кнопки опасна независимо от
        # действия (hover над «Confirm purchase» всё равно требует гейта при клике).
        if pattern.search(element_text):
            return GateVerdict(allowed=True, requires_confirmation=True,
                               reason=f"dangerous:{pattern.pattern[:30]}", source="procedural")
    return GateVerdict(allowed=True)