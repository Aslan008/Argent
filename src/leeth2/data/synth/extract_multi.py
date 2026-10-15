"""Several answers per extract question: "all file paths", "every commit hash", "найди все email-адреса".

One pass, any number of entities of any length (the GLiNER 2.5 idea, on Leeth 2's start / end head: losses.span_loss spreads the
probability uniformly over the right answers, evaluate._decode_spans reads the peaks).  The label is exact because the document is
built piece by piece and every entity's character offset is recorded as it is written.

A document has 0..8 entities of the asked type among distractors of OTHER types (so "all emails" must skip a URL that contains an
@), filler lines, and a carrier style (prose, log, shell session, chat, list).  Questions come in EN and RU, plural and singular
forms.  Three entity types (uuid, MAC address, colour) are held out of training: they exist to measure whether the idea transfers
to a kind of entity the model has not seen.
"""

from __future__ import annotations

import random
from typing import Callable, Dict, List, Optional, Tuple

from ...schema import Sample
from . import argent_router as AR

HEX = "0123456789abcdef"
FIRST = ["anna", "ivan", "olga", "max", "kate", "dmitry", "sara", "john", "lena", "pavel", "nick", "maria", "alex", "tom", "irina"]
DOMAINS = ["example.com", "mail.ru", "corp.local", "gmail.com", "company.io", "yandex.ru", "proton.me", "university.edu", "shop.kz", "dev.team"]
ENV_NAMES = ["API_KEY", "DATABASE_URL", "HOME", "PATH", "SECRET_TOKEN", "NODE_ENV", "PORT", "DEBUG", "LOG_LEVEL", "REDIS_URL"]
UNITS = ["MB", "GB", "KB", "ms", "s", "%", "°C", "px", "fps", "rpm"]


def _hex(rng: random.Random, n: int) -> str:
    return "".join(rng.choice(HEX) for _ in range(n))


def e_path(rng: random.Random) -> str:
    r = rng.random()
    if r < 0.5:
        return AR.filepath(rng)
    if r < 0.75:
        return rng.choice(["/etc", "/var/log", "/usr/local/bin", "/home/dev", "/opt/app"]) + "/" + AR.filename(rng)
    if r < 0.9:
        return "C:\\" + rng.choice(["Users\\Anna", "Projects", "Windows\\Temp", "Work\\app"]) + "\\" + AR.filename(rng)
    return "~/" + AR.filepath(rng)


def e_url(rng: random.Random) -> str:
    u = AR.url(rng)
    return u if rng.random() < 0.8 else u + "/" + rng.choice(["index.html", "api/v1/items", "download?id=" + str(rng.randint(1, 9999))])


def e_email(rng: random.Random) -> str:
    return f"{rng.choice(FIRST)}{rng.choice(['', '.', '_'])}{rng.choice(AR.STEMS)}@{rng.choice(DOMAINS)}"


def e_hash(rng: random.Random) -> str:
    return _hex(rng, rng.choice([7, 7, 8, 12, 40]))


def e_ip(rng: random.Random) -> str:
    return ".".join(str(rng.choice([10, 192, 172, 203, 8, 127, 34])) if i == 0 else str(rng.randint(0, 255)) for i in range(4))


def e_version(rng: random.Random) -> str:
    v = f"{rng.randint(0, 9)}.{rng.randint(0, 40)}.{rng.randint(0, 20)}"
    return rng.choice(["v", "", "v"]) + v + rng.choice(["", "", "-rc1", "-beta.2"])


def e_date(rng: random.Random) -> str:
    y, m, d = rng.randint(2019, 2026), rng.randint(1, 12), rng.randint(1, 28)
    return rng.choice([f"{y}-{m:02d}-{d:02d}", f"{d:02d}.{m:02d}.{y}", f"{d:02d}/{m:02d}/{y}"])


def e_error(rng: random.Random) -> str:
    return rng.choice([f"E{rng.randint(100, 9999)}", f"ERR_{rng.choice(AR.STEMS).upper()}_{rng.randint(1, 99)}", f"0x{_hex(rng, 8).upper()}", f"ORA-{rng.randint(10000, 99999)}", f"HTTP {rng.choice([400, 403, 404, 500, 502, 503])}"])


def e_func(rng: random.Random) -> str:
    return rng.choice(AR.FUNCS) + "()"


def e_size(rng: random.Random) -> str:
    return f"{rng.choice([rng.randint(1, 999), round(rng.uniform(0.1, 99), 1)])} {rng.choice(UNITS)}"


def e_env(rng: random.Random) -> str:
    return "$" + rng.choice(ENV_NAMES)


# held out of training
def e_uuid(rng: random.Random) -> str:
    return f"{_hex(rng, 8)}-{_hex(rng, 4)}-{_hex(rng, 4)}-{_hex(rng, 4)}-{_hex(rng, 12)}"


def e_mac(rng: random.Random) -> str:
    return rng.choice([":", "-"]).join(_hex(rng, 2) for _ in range(6))


def e_color(rng: random.Random) -> str:
    return "#" + _hex(rng, 6)


# round 2: the first round had 11 trained entity types and the three held-out ones scored 0.47 (set F1) against 0.99 on the trained ones, so what
# was learnt was "these eleven", not "any kind of thing you are told to list".  22 more kinds, all of them things that show up in logs, chats and
# tickets.  uuid / MAC / colour stay held out, so the number still measures transfer.
def e_ipv6(rng: random.Random) -> str:
    form = rng.random()
    if form < 0.35:
        return ":".join(_hex(rng, 4) for _ in range(8))
    if form < 0.7:
        return "fe80::" + ":".join(_hex(rng, rng.choice([1, 2, 3, 4])) for _ in range(rng.randint(2, 4)))
    return rng.choice(["2001:db8", "2a02:6b8", "fd00"]) + ":" + _hex(rng, 4) + "::" + _hex(rng, rng.choice([1, 2, 4]))


def e_sha256(rng: random.Random) -> str:
    return _hex(rng, 64)


def e_md5(rng: random.Random) -> str:
    return _hex(rng, 32)


def e_isotime(rng: random.Random) -> str:
    return f"{rng.randint(2020, 2026)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}T{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}:{rng.randint(0, 59):02d}Z"


def e_clock(rng: random.Random) -> str:
    return f"{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}"


def e_money(rng: random.Random) -> str:
    amount = f"{rng.randint(1, 9999):,}" + (f".{rng.randint(0, 99):02d}" if rng.random() < 0.6 else "")
    return rng.choice([f"${amount}", f"€{amount}", f"{amount} USD", f"{amount} руб"])


def e_percent(rng: random.Random) -> str:
    return f"{rng.choice([rng.randint(1, 100), round(rng.uniform(0.1, 99.9), 1)])}%"


def e_hashtag(rng: random.Random) -> str:
    return "#" + rng.choice(AR.STEMS) + rng.choice(["", "2026", "_dev", "team"])


def e_mention(rng: random.Random) -> str:
    return "@" + rng.choice(FIRST) + rng.choice(["", "_", "."]) + rng.choice(AR.STEMS)


def e_regkey(rng: random.Random) -> str:
    return rng.choice(["HKLM", "HKCU"]) + "\\SOFTWARE\\" + rng.choice(["Vendor", "Microsoft", "Acme"]) + "\\" + rng.choice(AR.CLASSES)


def e_image(rng: random.Random) -> str:
    return rng.choice(["", "ghcr.io/org/", "registry.local/"]) + rng.choice(AR.STEMS) + ":" + rng.choice(["latest", "1.25-alpine", "v2", "3.12-slim", "stable"])


def e_k8s(rng: random.Random) -> str:
    return rng.choice(["pod", "deploy", "svc"]) + "/" + rng.choice(AR.STEMS) + rng.choice(["", "-api", "-worker"]) + "-" + _hex(rng, 5)


def e_s3(rng: random.Random) -> str:
    return "s3://" + rng.choice(AR.STEMS) + "-" + rng.choice(["prod", "backup", "data"]) + "/" + AR.filename(rng)


def e_jira(rng: random.Random) -> str:
    return rng.choice(["PROJ", "OPS", "DEV", "SEC", "WEB", "ARG"]) + "-" + str(rng.randint(1, 9999))


def e_cve(rng: random.Random) -> str:
    return f"CVE-{rng.randint(2015, 2026)}-{rng.randint(1000, 99999)}"


def e_hexaddr(rng: random.Random) -> str:
    return "0x" + _hex(rng, rng.choice([8, 12, 16]))


def e_branch(rng: random.Random) -> str:
    return rng.choice(["feature", "fix", "release", "hotfix"]) + "/" + rng.choice(AR.STEMS) + "-" + rng.choice(AR.STEMS)


def e_domain(rng: random.Random) -> str:
    return rng.choice(["api", "www", "cdn", "mail", "static"]) + "." + rng.choice(AR.STEMS) + "." + rng.choice(["com", "io", "net", "dev", "ru"])


def e_username(rng: random.Random) -> str:
    return rng.choice(FIRST) + rng.choice(["_", "-", ""]) + rng.choice(AR.STEMS) + str(rng.randint(1, 99))


def e_phone(rng: random.Random) -> str:
    return rng.choice([f"+1 555 01{rng.randint(0, 9)}{rng.randint(0, 9)}", f"+7 900 555-01-{rng.randint(10, 99)}", f"(555) 010-{rng.randint(1000, 9999)}"])


def e_isbn(rng: random.Random) -> str:
    return f"978-{rng.randint(0, 9)}-{rng.randint(10, 99)}-{rng.randint(100000, 999999)}-{rng.randint(0, 9)}"


def e_doi(rng: random.Random) -> str:
    return f"10.{rng.randint(1000, 9999)}/{rng.choice(AR.STEMS)}.{rng.randint(1000, 99999)}"


def e_port(rng: random.Random) -> str:
    return rng.choice(["localhost", "127.0.0.1", "0.0.0.0"]) + ":" + str(rng.choice([3000, 5173, 8000, 8080, 9000, 5432, 6379, 27017]))


def e_filesize_path(rng: random.Random) -> str:
    return AR.directory(rng).rstrip("/") + "/"


# type -> (generator, EN singular, EN plural, RU plural (accusative), RU singular)
TYPES: Dict[str, Tuple[Callable[[random.Random], str], str, str, str, str]] = {
    "path": (e_path, "file path", "file paths", "пути к файлам", "путь к файлу"),
    "url": (e_url, "URL", "URLs", "ссылки", "ссылку"),
    "email": (e_email, "email address", "email addresses", "email-адреса", "email-адрес"),
    "hash": (e_hash, "commit hash", "commit hashes", "хеши коммитов", "хеш коммита"),
    "ip": (e_ip, "IP address", "IP addresses", "IP-адреса", "IP-адрес"),
    "version": (e_version, "version number", "version numbers", "номера версий", "номер версии"),
    "date": (e_date, "date", "dates", "даты", "дату"),
    "error": (e_error, "error code", "error codes", "коды ошибок", "код ошибки"),
    "func": (e_func, "function name", "function names", "имена функций", "имя функции"),
    "size": (e_size, "measurement", "measurements", "измерения с единицами", "измерение с единицами"),
    "env": (e_env, "environment variable", "environment variables", "переменные окружения", "переменную окружения"),
    "uuid": (e_uuid, "UUID", "UUIDs", "UUID", "UUID"),
    "mac": (e_mac, "MAC address", "MAC addresses", "MAC-адреса", "MAC-адрес"),
    "color": (e_color, "hex colour", "hex colours", "цвета в формате hex", "цвет в формате hex"),
    "ipv6": (e_ipv6, "IPv6 address", "IPv6 addresses", "IPv6-адреса", "IPv6-адрес"),
    "sha256": (e_sha256, "SHA-256 digest", "SHA-256 digests", "SHA-256 хеши", "SHA-256 хеш"),
    "md5": (e_md5, "MD5 checksum", "MD5 checksums", "MD5-суммы", "MD5-сумму"),
    "isotime": (e_isotime, "ISO timestamp", "ISO timestamps", "метки времени ISO", "метку времени ISO"),
    "clock": (e_clock, "time of day", "times of day", "время суток", "время суток"),
    "money": (e_money, "amount of money", "amounts of money", "денежные суммы", "денежную сумму"),
    "percent": (e_percent, "percentage", "percentages", "проценты", "процент"),
    "hashtag": (e_hashtag, "hashtag", "hashtags", "хештеги", "хештег"),
    "mention": (e_mention, "@mention", "@mentions", "упоминания через @", "упоминание через @"),
    "regkey": (e_regkey, "registry key", "registry keys", "ключи реестра", "ключ реестра"),
    "image": (e_image, "container image", "container images", "образы контейнеров", "образ контейнера"),
    "k8s": (e_k8s, "Kubernetes resource", "Kubernetes resources", "ресурсы Kubernetes", "ресурс Kubernetes"),
    "s3": (e_s3, "S3 URI", "S3 URIs", "S3-адреса", "S3-адрес"),
    "jira": (e_jira, "ticket key", "ticket keys", "номера задач", "номер задачи"),
    "cve": (e_cve, "CVE identifier", "CVE identifiers", "идентификаторы CVE", "идентификатор CVE"),
    "hexaddr": (e_hexaddr, "memory address", "memory addresses", "адреса памяти", "адрес памяти"),
    "branch": (e_branch, "branch name", "branch names", "названия веток", "название ветки"),
    "domain": (e_domain, "domain name", "domain names", "доменные имена", "доменное имя"),
    "username": (e_username, "username", "usernames", "имена пользователей", "имя пользователя"),
    "phone": (e_phone, "phone number", "phone numbers", "номера телефонов", "номер телефона"),
    "isbn": (e_isbn, "ISBN", "ISBNs", "ISBN", "ISBN"),
    "doi": (e_doi, "DOI", "DOIs", "DOI", "DOI"),
    "port": (e_port, "host and port", "hosts with ports", "хосты с портами", "хост с портом"),
    "dirpath": (e_filesize_path, "directory", "directories", "каталоги", "каталог"),
}
HOLDOUT_TYPES = ("uuid", "mac", "color")
TRAIN_TYPES = tuple(t for t in TYPES if t not in HOLDOUT_TYPES)

Q_PLURAL_EN = ["Extract all {pl}.", "List every {sg} mentioned in the text.", "Which {pl} appear here? Give all of them.", "Find all {pl} in this text.", "Return every {sg}."]
Q_PLURAL_RU = ["Извлеките все {pl}.", "Найди все {pl} в тексте.", "Перечисли {pl}, которые упомянуты.", "Какие {pl} здесь встречаются? Назови все.", "Выпиши все {pl}."]

# carrier templates: {e} is the entity (or a distractor), {t} a timestamp, {n} a small number
CARRIERS_EN = [
    "Please check {e} before the release.", "The failure was reported in {e}.", "[{t}] INFO worker-{n}: processed {e}", "see also {e}", "- {e}", "* {e}",
    "Note: {e} was changed yesterday.", "> {e}", "$ echo {e}", "{e}", "Ticket {n} mentions {e} and nothing else.", "Hi team, FYI: {e}.", "[{t}] WARN retry {n}/5 for {e}",
    "Remember {e} when you deploy.", "It looks like {e} is the culprit.",
]
CARRIERS_RU = [
    "Посмотри {e}, пожалуйста.", "Ошибка возникла в {e}.", "[{t}] INFO воркер-{n}: обработан {e}", "см. также {e}", "- {e}", "Заметка: {e} вчера изменили.",
    "Привет! Вот {e}.", "Не забудь про {e} при деплое.", "{e}", "В задаче {n} упоминается {e}.", "Похоже, виноват {e}.",
]
FILLER_EN = ["The build finished without warnings.", "Thanks, that helps a lot.", "We will discuss it on Monday.", "Nothing else to report.", "All tests are green on my side.",
             "The deployment window is tonight.", "Let me know if anything is unclear.", "Meeting notes are attached."]
FILLER_RU = ["Сборка прошла без предупреждений.", "Спасибо, это очень помогло.", "Обсудим в понедельник.", "Больше сообщать нечего.", "Все тесты у меня зелёные.",
             "Окно деплоя сегодня ночью.", "Если что-то непонятно — напишите.", "Заметки со встречи во вложении."]


def _timestamp(rng: random.Random) -> str:
    return f"2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d} {rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}:{rng.randint(0, 59):02d}"


def _doc(rng: random.Random, want: str, m: int, ru: bool, types_pool: Tuple[str, ...]) -> Tuple[str, List[List[int]]]:
    """A document with exactly ``m`` entities of type ``want`` among distractors; returns (text, [[start, end], ...])."""
    carriers = CARRIERS_RU if ru else CARRIERS_EN
    filler = FILLER_RU if ru else FILLER_EN
    others = [t for t in types_pool if t != want]
    # a distractor must never read as the wanted type: dates are not versions, a path is not a URL, etc. -> keep only clearly different pairs
    clash = {"url": {"path", "email", "domain", "s3", "port", "image"}, "path": {"url", "email", "s3", "dirpath", "regkey", "image"}, "email": {"url", "mention", "domain", "username"},
             "version": {"ip", "date", "ipv6", "isbn", "doi", "phone", "image", "money", "percent", "clock"}, "ip": {"version", "ipv6", "port"}, "date": {"version", "isotime", "isbn", "phone"},
             "hash": {"uuid", "sha256", "md5", "hexaddr", "k8s", "color"}, "uuid": {"hash", "sha256", "md5"}, "sha256": {"hash", "md5", "uuid"}, "md5": {"hash", "sha256", "uuid"},
             "error": {"hash", "size", "hexaddr", "jira", "cve"}, "size": {"error", "money", "percent", "version"}, "func": {"branch", "k8s"}, "env": {"path", "mention"}, "mac": {"hash", "uuid", "ipv6"},
             "color": {"hash", "hashtag", "hexaddr"}, "ipv6": {"ip", "mac", "version", "clock", "port"}, "isotime": {"date", "clock"}, "clock": {"isotime", "ipv6", "version", "date"},
             "money": {"percent", "size", "version"}, "percent": {"money", "size", "version"}, "hashtag": {"color", "mention"}, "mention": {"email", "hashtag", "username", "env"},
             "regkey": {"path", "dirpath"}, "image": {"url", "version", "path", "k8s", "domain"}, "k8s": {"path", "image", "hash", "branch"}, "s3": {"url", "path"},
             "jira": {"cve", "error", "version"}, "cve": {"jira", "error", "date"}, "hexaddr": {"hash", "error", "color", "sha256", "md5"}, "branch": {"path", "func", "k8s", "dirpath"},
             "domain": {"url", "email", "image", "port"}, "username": {"email", "mention", "func"}, "phone": {"date", "version", "isbn", "ip"}, "isbn": {"phone", "date", "version", "doi"},
             "doi": {"isbn", "version", "date"}, "port": {"url", "ip", "ipv6", "domain"}, "dirpath": {"path", "branch", "regkey"}}
    others = [t for t in others if t not in clash.get(want, set()) and want not in clash.get(t, set())]
    items: List[Tuple[Optional[str], str]] = [(want, TYPES[want][0](rng)) for _ in range(m)]
    items += [(None, TYPES[rng.choice(others)][0](rng)) for _ in range(rng.randint(0, 4))]
    rng.shuffle(items)
    lines: List[str] = []
    spans: List[List[int]] = []
    pos = 0
    head = rng.choice(["", "", "Report:\n", "Лог:\n", "Notes\n", "Сообщение:\n"])
    out = head
    pos = len(head)
    n_filler = rng.randint(0, 3)
    for kind, ent in items:
        for _ in range(rng.randint(0, 1) if n_filler else 0):
            f = rng.choice(filler) + "\n"
            out += f
            pos += len(f)
            n_filler -= 1
        tpl = rng.choice(carriers)
        pre, post = tpl.split("{e}")
        pre = pre.replace("{t}", _timestamp(rng)).replace("{n}", str(rng.randint(1, 99)))
        post = post.replace("{t}", _timestamp(rng)).replace("{n}", str(rng.randint(1, 99)))
        out += pre
        start = len(out)
        out += ent
        if kind == want:
            spans.append([start, start + len(ent)])
        out += post + "\n"
    for _ in range(max(0, n_filler)):
        out += rng.choice(filler) + "\n"
    return out.rstrip("\n"), spans


def gen_multi(rng: random.Random, holdout: bool = False) -> Optional[List[Sample]]:
    """One document, one question. ``holdout``: ask only for the held-out entity types (distractors are anything)."""
    want = rng.choice(HOLDOUT_TYPES if holdout else TRAIN_TYPES)
    pool = tuple(TYPES) if holdout else TRAIN_TYPES
    m = rng.choices([0, 1, 2, 3, 4, 5, 6, 8], [0.12, 0.17, 0.18, 0.16, 0.13, 0.1, 0.08, 0.06])[0]
    ru = rng.random() < 0.4
    state, spans = _doc(rng, want, m, ru, pool)
    en_sg, en_pl, ru_pl, ru_sg = TYPES[want][1], TYPES[want][2], TYPES[want][3], TYPES[want][4]
    if ru:
        q = rng.choice(Q_PLURAL_RU).format(pl=ru_pl)
    else:
        q = rng.choice(Q_PLURAL_EN).format(pl=en_pl, sg=en_sg)
    s = Sample("extract_multi", "extract_multi", "ru" if ru else "en", "extract", state, q, None, spans or None, meta={"multi": True, "etype": want})
    return [s]


def gen_multi_holdout(rng: random.Random) -> Optional[List[Sample]]:
    return gen_multi(rng, holdout=True)


GENERATORS = [("multi", gen_multi, 40000)]
