"""Answer hints written into a safety STATE by the teacher - the label said in words instead of shown by the situation.

Found 2026-10-19: every long R001 record ended with a narrator's sentence on what the action would do ("...permanently destroying
all staging data", "...without modifying any system logs", "under security rule 1 this is always dangerous"); normal batches of
rounds 2-3 described the action with verdict words ("safe planned cleanup", "permanently dropping...") in ~10 % of rows, rounds
4-5 in 1-2 %. A model trained on them reads the verdict, not the action. The ingest refuses such safety records.

Nouns are allowed: "журнал безопасности", "security log", "политика" are names of objects the action touches.
"""

from __future__ import annotations

import re
from typing import Optional

_VERDICT = re.compile(
    r"\b(опасн\w*|небезопасн\w*|безопасн(ый|ая|ое|ые|ого|ому|ым|ыми|ом|ой|ых|о)|безвредн\w*|деструктивн\w*|разрушительн\w*|"
    r"безвозвратн\w*|необратим\w*|небезпечн\w*|безпечн(ий|а|е|о|ого|ому)|"
    r"dangerous|unsafe|harmless|destructive|non-destructive|risky|irreversibl\w*|irrecoverabl\w*|permanently|safely|"
    r"gefährlich\w*|harmlos\w*|unumkehrbar\w*|peligros\w*|inofensiv\w*|irreversible|dangereu\w*|inoffensi\w*|"
    r"pericolos\w*|innocu\w*|perigos\w*|niebezpieczn\w*|nieodwracaln\w*|bezpowrotn\w*|tehlikeli|zararsız|güvenli|"
    r"қауіпті|қауіпсіз)\b", re.I)
_READ_ONLY = re.compile(r"(read[- ]only|только (на )?чтени|только для чтения|лише читанн|somente leitura|lecture seule|solo lectura|"
                        r"nur lesend|sola lettura|tylko do odczytu|salt okunur)", re.I)
_CONSEQUENCE = re.compile(
    r"(without (modify|alter|chang|affect|delet|touch|remov)\w*|sans (modifi|alt[ée]r|supprim)\w*|sin (modific|alterar|borrar)\w*|"
    r"sem (modific|alterar|apagar)\w*|ohne (zu )?(ändern|löschen|veränder)\w*|bez (zmian|modyfik|usuwan)\w*|"
    r"не (изменяя|удаляя|затрагивая|влияя|очищая|меняя)|не вносит изменени\w*|без изменени\w*|"
    r"data loss|loss of (all )?data|потер\w* (всех )?данн\w*|утрат\w* данн\w*|utrat\w* danych|perte de données|pérdida de datos|"
    r"perda de dados|datenverlust)", re.I)
_RULE = re.compile(r"(правил\w*\s*№?\s*\d|\brule\s*#?\s*\d|\bregel\s*\d|règle\s*(de sécurité\s*)?\d|regla\s*(de seguridad\s*)?\d|"
                   r"regola\s*(di sicurezza\s*)?\d|reguł\w*\s*\d|kural\w*\s*\d|правил\w* \d)", re.I)
_NARRATION = re.compile(r"(агент\w* (сам|по своей инициативе|без запроса|без спроса)|по решению агента|без спроса|сам решил|"
                        r"on its own|without being asked|unprompted|ohne Aufforderung|de su propia iniciativa|de sa propre initiative|"
                        r"kendi kendine|за рамки (просьбы|запроса)|сверх просьбы|не просил\w*|did not ask|didn't ask|never asked)", re.I)


_QUOTES = "\"'«»“”„‘’「」"
_USER_LINE = re.compile(r"\s*(user|пользователь|користувач|usuario|utilisateur|benutzer|użytkownik|kullanıcı|пайдаланушы|қолданушы|"
                        r"utente|usuário|ユーザー|用户)[^:：\n]{0,40}[:：]", re.I)


def _quoted_or_in_token(text: str, start: int) -> bool:
    """A match inside a flag or a name (--delete-permanently, read_only.cfg) is part of the command; a match inside quotes or on the
    user's line ("User: verify it without modifying any records") is the user's own words. Neither is the narrator's verdict."""
    if start and text[start - 1] in "-_./=":
        return True
    line_start = text.rfind("\n", 0, start) + 1
    if _USER_LINE.match(text, line_start):
        return True
    return sum(text[line_start:start].count(q) for q in _QUOTES) % 2 == 1


def verdict_in_state(state: str) -> Optional[str]:
    """The first hint found in a safety state, or None."""
    for pat in (_VERDICT, _READ_ONLY, _CONSEQUENCE, _RULE, _NARRATION):
        for m in pat.finditer(state):
            if not _quoted_or_in_token(state, m.start()):
                return m.group(0)
    return None


# The ONE line that differs in a fact twin pair ("в факте", "факт в середине контекста") must be a fact about the situation - what
# the file is, who uses it, whether a copy exists - not a norm or a forecast. Round 17 wrote the verdict there instead: "deletion is
# strictly forbidden by the audit standard", "the reset was allowed by the administrator", "removing it will trigger cascading
# removal of cloud-init", "Removal will not impact any system daemons" (fact pairs with such a line: rounds 15-16 2 of 72, round 17
# 21 of 36). Checked on the differing lines only: a whole state may well hold a policy name or the user's "можно".
_NORM = re.compile(
    r"(запрещ\w*|запрет\w*|разреш\w*|недопустим\w*|нельзя|наруш\w*|привед[её]т|вызовет|сломает|прерв[её]т|повлеч[её]т|"
    r"заборон\w*|дозвол\w*|поруш\w*|призведе|"
    r"forbid\w*|prohibit\w*|\ballow(ed|s)?\b|permit(ted|s)?\b|violat\w*|\bbreak(s|ing)?\b|\bpolicy|политик\w*|політик\w*|"
    r"\bwill (not )?(trigger|cause|break|destroy|disrupt|interrupt|impact|affect|harm|delete|wipe|lose|crash|drop|take down|corrupt)\w*|"
    r"verboten|erlaubt|interdit\w*|autoris\w*|prohibid\w*|permitid\w*|proibid\w*|vietat\w*|consentit\w*|zakazan\w*|dozwolon\w*|"
    r"yasak\w*|izin ver\w*|тыйым|рұқсат|禁止|严禁|允许|将导致|会导致|不会影响|許可|引き起こ)", re.I)


def norm_in_fact(line: str) -> Optional[str]:
    """A norm or forecast in a fact twin's differing line (or a verdict word), or None."""
    m = _NORM.search(line)
    return m.group(0) if m else verdict_in_state(line)
