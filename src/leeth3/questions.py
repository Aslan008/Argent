"""The safety question carries no information: Argent always asks the same thing (src/leeth2/data/argent.py SAFETY_QUESTION).

A teacher that writes its own question per example puts the answer into it ("may the agent irreversibly destroy the master key...",
"...when the user only asked..."). On round 2-4 batches the question ALONE predicted the label in 78-85 % of cases (round 1: 50 %),
so a model would learn to read the question instead of the action. The ingest therefore replaces every safety question with one of
these neutral phrasings in the example's language, chosen deterministically from the record id (a re-ingest is byte-identical).
"""

from __future__ import annotations

import hashlib
from typing import Dict, List

from src.leeth2.data.argent import SAFETY_QUESTION
from src.leeth3.records import QUESTION_FREE_DOMAINS

SAFETY_QUESTIONS: Dict[str, List[str]] = {
    "ru": [SAFETY_QUESTION,
           "Можно ли выполнить это, не спрашивая пользователя?",
           "Безопасно ли выполнить это действие сразу, без подтверждения?",
           "Может ли агент выполнить это без вопроса пользователю?",
           "Допустимо ли выполнить это немедленно, не дожидаясь пользователя?",
           "Можно ли запускать это без подтверждения пользователя?"],
    "en": ["Is it safe to run this without asking the user?",
           "Can the agent do this right away without confirmation?",
           "Is this action safe to execute immediately?",
           "May the agent proceed with this without checking with the user?",
           "Is it fine to run this without the user's sign-off?"],
    "de": ["Darf das ohne Rückfrage beim Benutzer ausgeführt werden?",
           "Ist es sicher, das sofort ohne Bestätigung auszuführen?",
           "Kann der Agent das ausführen, ohne den Benutzer zu fragen?"],
    "es": ["¿Se puede ejecutar esto sin preguntar al usuario?",
           "¿Es seguro hacer esto de inmediato, sin confirmación?",
           "¿Puede el agente realizar esto sin consultar al usuario?"],
    "fr": ["Peut-on exécuter cela sans demander à l'utilisateur ?",
           "Est-il sûr de faire cela tout de suite, sans confirmation ?",
           "L'agent peut-il effectuer cela sans consulter l'utilisateur ?"],
    "pt": ["É possível executar isso sem perguntar ao usuário?",
           "É seguro fazer isso imediatamente, sem confirmação?",
           "O agente pode realizar isso sem consultar o usuário?"],
    "it": ["Si può eseguire questo senza chiedere all'utente?",
           "È sicuro farlo subito, senza conferma?",
           "L'agente può procedere senza consultare l'utente?"],
    "pl": ["Czy można to wykonać bez pytania użytkownika?",
           "Czy bezpiecznie jest zrobić to od razu, bez potwierdzenia?",
           "Czy agent może to wykonać bez konsultacji z użytkownikiem?"],
    "tr": ["Bu, kullanıcıya sormadan çalıştırılabilir mi?",
           "Bunu onay almadan hemen yapmak güvenli mi?",
           "Ajan bunu kullanıcıya danışmadan yapabilir mi?"],
    "uk": ["Чи можна виконати це, не питаючи користувача?",
           "Чи безпечно виконати цю дію одразу, без підтвердження?",
           "Чи може агент зробити це без запитання користувачеві?"],
    "kk": ["Мұны пайдаланушыдан сұрамай орындауға бола ма?",
           "Бұл әрекетті растаусыз бірден орындау қауіпсіз бе?",
           "Агент мұны пайдаланушымен ақылдаспай орындай ала ма?"],
    "zh": ["不询问用户就可以执行这个操作吗?",
           "无需确认立即执行这个操作安全吗?",
           "代理可以不经用户同意直接执行吗?"],
    "ja": ["ユーザーに確認せずにこれを実行してもよいですか?",
           "確認なしでこの操作をすぐに実行しても安全ですか?",
           "エージェントはユーザーに尋ねずにこれを実行できますか?"],
}


# The same failure in tool choice: from round 2 the teacher described the needed tool in the question ("which action places a market
# sell order..."), and the question alone picked the right tool in 86 % of cases (round 1: 11 %) - the request was not needed.
# Argent's own question names no tool either (TOOL_QUESTION); these are its app-neutral equivalents.
TOOL_QUESTIONS: Dict[str, List[str]] = {
    "ru": ["Какой инструмент должен выполнить запрос?", "Какой инструмент нужно вызвать в первую очередь?",
           "Какая функция выполнит этот запрос?", "Что из этого нужно вызвать для запроса?", "Какое действие подходит для этого запроса?"],
    "en": ["Which tool should handle this request?", "Which tool should be called first?", "Which action handles this request?",
           "Which of these should be called for the request?", "What function fits this request?"],
    "de": ["Welches Werkzeug soll diese Anfrage ausführen?", "Welche Funktion passt zu dieser Anfrage?",
           "Welche Aktion soll zuerst aufgerufen werden?"],
    "es": ["¿Qué herramienta debe atender esta solicitud?", "¿Qué función corresponde a esta petición?",
           "¿Qué acción hay que llamar primero?"],
    "fr": ["Quel outil doit traiter cette demande ?", "Quelle fonction correspond à cette demande ?",
           "Quelle action faut-il appeler en premier ?"],
    "pt": ["Qual ferramenta deve atender este pedido?", "Qual função corresponde a esta solicitação?",
           "Qual ação deve ser chamada primeiro?"],
    "it": ["Quale strumento deve gestire questa richiesta?", "Quale funzione corrisponde a questa richiesta?",
           "Quale azione va chiamata per prima?"],
    "pl": ["Które narzędzie powinno obsłużyć tę prośbę?", "Która funkcja pasuje do tej prośby?",
           "Którą akcję należy wywołać najpierw?"],
    "tr": ["Bu isteği hangi araç karşılamalı?", "Bu isteğe hangi işlev uygun?", "Önce hangi eylem çağrılmalı?"],
    "uk": ["Який інструмент має виконати цей запит?", "Яка функція підходить для цього запиту?",
           "Яку дію слід викликати першою?"],
    "kk": ["Бұл сұранысты қай құрал орындауы керек?", "Бұл сұранысқа қай функция сәйкес келеді?",
           "Алдымен қай әрекетті шақыру керек?"],
    "zh": ["应该由哪个工具处理这个请求?", "哪个功能适合这个请求?", "应该先调用哪个操作?"],
    "ja": ["このリクエストを処理すべきツールはどれですか?", "このリクエストに合う機能はどれですか?",
           "最初に呼び出すべき操作はどれですか?"],
}

NEUTRAL: Dict[str, Dict[str, List[str]]] = {"safety": SAFETY_QUESTIONS, "tools": TOOL_QUESTIONS}
assert set(NEUTRAL) == set(QUESTION_FREE_DOMAINS), "the dedup key and the neutral questions must cover the same domains"


def neutral_question(domain: str, record_id: str, lang: str) -> str:
    """The neutral question of a domain whose question must carry no information; KeyError for other domains."""
    table = NEUTRAL[domain]
    pool = table.get(lang) or table["en"]
    return pool[int(hashlib.sha256(record_id.encode("utf-8")).hexdigest(), 16) % len(pool)]


def neutral_safety_question(record_id: str, lang: str) -> str:
    return neutral_question("safety", record_id, lang)
