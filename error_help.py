"""Turning a Python traceback line into something the user can act on.

A failure reached the user as

    Error: Error: 'utf-8' codec can't encode characters in position
    53000-53001: surrogates not allowed

which is true, correctly spelled, and useless: it names neither what broke nor
what to do. The raw line is still printed — it is what you quote in a bug
report — but a sentence underneath says what it means here.

Only failures that have actually happened are listed. A guessed explanation for
an error nobody has hit is worse than none: it sends the reader down a path
that may have nothing to do with their problem.
"""

import re

# (pattern, what it means HERE — not what the exception means in general)
_EXPLANATIONS = [
    (r"surrogates not allowed",
     "В тексте оказался «половинчатый» символ Unicode — обычно от обрезанного "
     "эмодзи в ответе модели. Диалог чинится сам на следующем ходу; если нет — /clear."),
    (r"context (window|length)|too many tokens|maximum context",
     "Запрос не помещается в окно контекста модели. Сократите историю (/clear) "
     "или увеличьте окно в /model."),
    (r"connection refused|failed to establish|max retries exceeded|connection aborted",
     "Не удалось подключиться к провайдеру. Проверьте, запущен ли Ollama "
     "(`ollama ps`), и адрес в /provider."),
    (r"\b(401|403)\b|unauthorized|invalid api key|authentication",
     "Провайдер отклонил ключ. Проверьте его в /provider."),
    (r"\b429\b|rate limit|quota",
     "Провайдер ограничил частоту запросов. Подождите или смените модель в /model."),
    (r"\b404\b.*model|model .*not found|no such model",
     "Провайдер не знает такую модель — часто бывает после смены провайдера. "
     "Выберите модель заново: /model."),
    (r"no endpoints found that support tool use",
     "У этой модели нет канала вызова инструментов. Argent сам перейдёт на "
     "текстовый формат вызовов, но качество будет ниже — для работы с кодом "
     "лучше выбрать модель с поддержкой tools."),
    (r"timed out|timeout",
     "Ответа не дождались. Если это MCP — проверьте, отвечает ли приложение; "
     "таймаут настраивается ключом mcp_call_timeout."),
    (r"permission denied|access is denied|\[errno 13\]",
     "Нет прав на файл или папку. Закройте программу, которая держит файл, "
     "или запустите Argent от имени пользователя с доступом."),
    (r"no space left|not enough space|disk full",
     "На диске нет места."),
    (r"\[errno 2\]|no such file or directory|cannot find the (file|path)",
     "Файл или папка не найдены — проверьте путь и текущую директорию (/cd)."),
    (r"json.*decode|expecting value|invalid json",
     "Ответ пришёл в неверном JSON. Обычно это модель сломала формат — "
     "повторите запрос; если повторяется, помогает модель поумнее."),
]

_COMPILED = [(re.compile(p, re.IGNORECASE), text) for p, text in _EXPLANATIONS]


def explain_error(message: str) -> str | None:
    """A sentence the user can act on, or None when we genuinely do not know."""
    if not message:
        return None
    for pattern, explanation in _COMPILED:
        if pattern.search(message):
            return explanation
    return None
