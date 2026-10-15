# -*- coding: utf-8 -*-
"""scripts/_multidomain_battery.py — bake-off: Leeth (zero-shot) vs Laya на 3 чужих сферах.

Сферы: support (отдел/срочность/отток), email (тип/приоритет/контакт), moderation (safe/категория).
Каждый кейс: (sphere, state, [(qid, gold)]). Вопросы — единый источник правды (условие на строку).
Раннеры: leeth (текущий прод-чекпоинт, без дообучения) и laya (auto/english/multilingual/typed-decisions).
Метрики: acc / ECE (10 бинов, conf = уверенность в предсказанном классе) / ms на вопрос.
Выход: консольная таблица + .argent/artifacts/_multidomain_raw.json
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT_JSON = ROOT / ".argent/artifacts/_multidomain_raw.json"


def prod_envelope(text):
    """Канонический прод-конверт (dispatcher._build_state) — так state выглядит в проде."""
    normalized = text[:1].lower() + text[1:] if text[:1].isalpha() else text
    return f'Incoming user request in Argent session:\n"{normalized}"\nWorking directory: C:/Users/user/Desktop/Argent'

# ---------------------------------------------------------------------------
# Вопросы — единый источник правды (модель условия на точной строке)
# ---------------------------------------------------------------------------
Q_DEPARTMENT = "В какой отдел нужно направить это обращение?"
DEPT_CHOICES = ["billing", "technical", "sales", "account", "other"]
DEPT_CRITERIA = {
    "billing": "счета, платежи, возвраты, двойные списания",
    "technical": "баги, сбои, ошибки, не работает функция",
    "sales": "цены, тарифы, договоры, покупка лицензий",
    "account": "аккаунт, доступ, пароль, регистрация, профиль",
    "other": "всё остальное",
}
Q_URGENCY = "Насколько срочно нужно решить проблему клиента?"
URGENCY_RUBRIC = [
    "не срочно — можно подождать",
    "скоро — в течение дней",
    "блокер — работа клиента остановлена",
]
Q_CHURN = "Клиент угрожает отменить подписку или уйти к конкуренту?"

Q_PRIORITY = "Какой приоритет у этого письма?"
PRIORITY_RUBRIC = [
    "низкий — можно ответить позже",
    "обычный — ответить в рабочем порядке",
    "высокий — требует немедленного ответа",
]
Q_EMAIL_TYPE = "К какому типу относится это письмо?"
EMAIL_TYPE_CHOICES = ["question", "complaint", "meeting_request", "newsletter", "personal", "spam"]
EMAIL_TYPE_CRITERIA = {
    "question": "вопрос о продукте или услуге",
    "complaint": "жалоба, недовольство, требование компенсации",
    "meeting_request": "предложение созвона или встречи",
    "newsletter": "рассылка, дайджест, подборка новостей",
    "personal": "личная переписка, не по работе",
    "spam": "реклама, выигрыши, массовая рассылка",
}
Q_CONTACT = "Извлеките email-адрес, упомянутый в тексте письма"

Q_SAFE = "Безопасно ли публиковать этот текст? (True = безопасно)"
Q_VIOLATION = "Какая категория нарушения в этом тексте?"
VIOLATION_CHOICES = ["none", "spam", "harassment", "violence", "adult", "fraud"]
VIOLATION_CRITERIA = {
    "none": "нарушений нет, обычный текст",
    "spam": "реклама, массовые рассылки, навязчивые ссылки",
    "harassment": "оскорбления, травля, унижение человека",
    "violence": "угрозы насилием, призывы к насилию",
    "adult": "откровенный контент 18+",
    "fraud": "мошенничество, обман, выманивание денег или данных",
}

QTYPE = {
    "department": "choice", "urgency": "score", "churn": "noul",
    "etype": "choice", "priority": "score", "contact": "extract",
    "safe": "noul", "violation": "choice",
}

# ---------------------------------------------------------------------------
# Кейсы: (sphere, state, [(qid, gold)]) — gold: choice=str, score=уровень 1..3,
# noul=bool, extract=подстрока
# ---------------------------------------------------------------------------
CASES = [
    # ---------- SUPPORT: department ----------
    ("support", "Здравствуйте! Списали оплату дважды за март — 4 990 и ещё раз 4 990. Верните дубль, пожалуйста.", [("department", "billing")]),
    ("support", "Приложение вылетает каждый раз, когда открываю раздел Настройки. Android 14, версия 5.2.1.", [("department", "technical")]),
    ("support", "Планируем расшириться: сколько стоит годовой тариф на 50 мест и есть ли скидка?", [("department", "sales")]),
    ("support", "После сброса пароля новый пароль не подходит, войти в аккаунт не могу.", [("department", "account")]),
    ("support", "Хочу перенести все данные со старого аккаунта на новый, а старый удалить.", [("department", "account")]),
    ("support", "Отмена подписки не прошла, деньги списались снова. Оформите возврат.", [("department", "billing")]),
    ("support", "Интеграция со Slack отвалилась после вашего обновления в среду, вебхуки не доставляются.", [("department", "technical")]),
    ("support", "Пришлите, пожалуйста, договор и счёт на 20 лицензий для юрлица.", [("department", "sales")]),
    ("support", "Платёж по карте проходит с ошибкой 402, хотя карта рабочая. Проверьте начисления.", [("department", "billing")]),
    ("support", "Не приходит письмо для подтверждения регистрации, уже три раза запрашивал.", [("department", "account")]),
    ("support", "Расскажите про корпоративные условия: есть ли SLA и выделенный менеджер?", [("department", "sales")]),
    ("support", "Кнопка Экспорт крутится бесконечно и файл так и не скачивается.", [("department", "technical")]),
    ("support", "Списали по тарифу Pro, хотя я на Start. Разберите, пожалуйста, начисления.", [("department", "billing")]),
    ("support", "Хочу сменить email в профиле, но доступ к старому ящику уже потерял.", [("department", "account")]),
    # ---------- SUPPORT: urgency (1..3) ----------
    ("support", "Вся команда не может работать вторые сутки, дедлайн завтра. Это полный блокер.", [("urgency", 3)]),
    ("support", "Опечатка в письме-подтверждении заказа, поправьте когда будет удобно.", [("urgency", 1)]),
    ("support", "Платежи не проходят с утра, клиенты не могут оформить заказ.", [("urgency", 3)]),
    ("support", "Хотелось бы тёмную тему в одном из будущих релизов.", [("urgency", 1)]),
    ("support", "TLS-сертификат истекает через три дня, продлите до конца недели.", [("urgency", 2)]),
    ("support", "Экспорт отчётов иногда зависает, перезапускаем вручную. Не критично, но почините.", [("urgency", 2)]),
    ("support", "Сайт лежит полностью, ни один клиент не может зайти.", [("urgency", 3)]),
    ("support", "Пара кнопок съехала на странице тарифов в Safari.", [("urgency", 1)]),
    ("support", "API отвечает 500 на каждый второй запрос с обеда.", [("urgency", 3)]),
    ("support", "Уведомления приходят с задержкой в пару часов, терпимо, но поправьте.", [("urgency", 2)]),
    ("support", "Не работает поиск по базе знаний уже неделю, сотрудники жалуются.", [("urgency", 2)]),
    ("support", "Всё сломалось после обновления, мы не можем провести ни одну сделку.", [("urgency", 3)]),
    ("support", "Добавьте, пожалуйста, поддержку CSV-импорта в следующей версии.", [("urgency", 1)]),
    # ---------- SUPPORT: churn (bool) ----------
    ("support", "Если в течение суток не решите проблему, мы уходим к вашему конкуренту.", [("churn", True)]),
    ("support", "Отмените, пожалуйста, подписку — с конца месяца она нам не нужна.", [("churn", True)]),
    ("support", "Спасибо за быстрый ответ, всё заработало, очень довольны!", [("churn", False)]),
    ("support", "Как добавить коллегу в наше рабочее пространство?", [("churn", False)]),
    ("support", "Мы рассматриваем переход на другую платформу, если тариф не пересмотрят.", [("churn", True)]),
    ("support", "Продлите подписку ещё на год, всё устраивает.", [("churn", False)]),
    ("support", "Третий месяц платим за функцию, которой нет. Возвращайте деньги или мы уходим.", [("churn", True)]),
    ("support", "Подскажите, как настроить двухфакторную аутентификацию?", [("churn", False)]),
    ("support", "Ваш менеджер игнорирует нас вторую неделю. Готовим расторжение договора.", [("churn", True)]),
    ("support", "Когда будет поддержка нового формата файлов?", [("churn", False)]),
    ("support", "Расторгаем договор — бюджет на следующий год урезали.", [("churn", True)]),
    ("support", "Отличный продукт, рекомендую всем коллегам.", [("churn", False)]),
    ("support", "Считаем переход на другой сервис, ваш тариф для нас дороговат.", [("churn", True)]),
    # ---------- EMAIL: type ----------
    ("email", "Подскажите, поддерживаете ли вы экспорт отчётов в CSV?", [("etype", "question")]),
    ("email", "Ваше обновление сломало наш производственный процесс. Требуем компенсацию простоя.", [("etype", "complaint")]),
    ("email", "Предлагаю созвониться во вторник в 15:00 и обсудить интеграцию.", [("etype", "meeting_request")]),
    ("email", "ТОЛЬКО СЕГОДНЯ скидка 50% на все курсы! Успей перейти по ссылке!", [("etype", "spam")]),
    ("email", "Мам, я буду дома в восемь, купи хлеба.", [("etype", "personal")]),
    ("email", "Февральская подборка: 10 новых функций продукта, о которых вы могли не знать.", [("etype", "newsletter")]),
    ("email", "Здравствуйте, подскажите тариф для некоммерческого использования?", [("etype", "question")]),
    ("email", "Это уже третий тикет за неделю! Ваша поддержка игнорирует клиентов, верните деньги.", [("etype", "complaint")]),
    ("email", "Прикрепляю повестку квартального обзора, жду правки до пятницы.", [("etype", "meeting_request")]),
    ("email", "Вы выиграли iPhone! Переведите 300 за доставку и получите приз.", [("etype", "spam")]),
    ("email", "Привет! Как отдохнул? Созвонимся на выходных?", [("etype", "personal")]),
    ("email", "Дайджест за март: главные новости индустрии и пять подборок инструментов.", [("etype", "newsletter")]),
    ("email", "Возможен ли перенос данных из старой системы? Интересует процедура и сроки.", [("etype", "question")]),
    ("email", "Ваш сервис дважды за месяц подвёл наш релиз. Мы очень недовольны.", [("etype", "complaint")]),
    # ---------- EMAIL: priority (1..3) ----------
    ("email", "Продакшен лежит, клиенты не могут оформить заказ с утра. Ответьте немедленно.", [("priority", 3)]),
    ("email", "Когда-нибудь сделаете тёмную тему? Было бы здорово. Спасибо!", [("priority", 1)]),
    ("email", "Уточните, пожалуйста, реквизиты счёта за июнь — когда будет удобно.", [("priority", 2)]),
    ("email", "У нас аудит в пятницу, нужен экспорт логов до конца дня.", [("priority", 3)]),
    ("email", "Расскажите про вашу реферальную программу, интересно на досуге.", [("priority", 1)]),
    ("email", "Подтвердите, пожалуйста, участие в демо в четверг — жду до завтра.", [("priority", 2)]),
    ("email", "Юридический отдел просит подписанный договор до 15:00 сегодня.", [("priority", 3)]),
    ("email", "Присылайте новости продукта, как будет удобно.", [("priority", 1)]),
    ("email", "Счёт за август пришёл с неверной суммой, проверьте до конца недели.", [("priority", 2)]),
    ("email", "Срочно: двойное списание у ключевого клиента, он готовит претензию.", [("priority", 3)]),
    ("email", "Поздравляем с запуском! Небольшое предложение на будущее.", [("priority", 1)]),
    ("email", "Напомните, пожалуйста, до пятницы, продлеваем ли мы лицензию.", [("priority", 2)]),
    ("email", "Инцидент безопасности: подозрительный вход в аккаунт администратора. Реагируйте сейчас.", [("priority", 3)]),
    # ---------- EMAIL: contact (extract) ----------
    ("email", "По всем вопросам пишите на ivan.petrov@acme.com, отвечаю в течение дня.", [("contact", "ivan.petrov@acme.com")]),
    ("email", "Мой адрес для связи: o.smirnova@company.org (не корпоративный).", [("contact", "o.smirnova@company.org")]),
    ("email", "Продублируйте ответ мне лично: alex.k@gmail.com", [("contact", "alex.k@gmail.com")]),
    ("email", "Контакты отдела: support@vendor.ru и sales@vendor.ru. Пишите в поддержку.", [("contact", "support@vendor.ru")]),
    ("email", "Отвечайте на n.orlova@corp.example, это рабочий адрес.", [("contact", "n.orlova@corp.example")]),
    ("email", "Счёт отправьте в бухгалтерию: buh@firma24.ru", [("contact", "buh@firma24.ru")]),
    ("email", "Свяжитесь с нашим менеджером Павлом: p.sidorov@partner.com", [("contact", "p.sidorov@partner.com")]),
    ("email", "Все документы — на legal@holding.net, дублировать не нужно.", [("contact", "legal@holding.net")]),
    ("email", "Моя почта для уведомлений: dmitry_1978@mail.ru", [("contact", "dmitry_1978@mail.ru")]),
    ("email", "Напишите на info@npo-mir.ru, если решение по заявке готово.", [("contact", "info@npo-mir.ru")]),
    ("email", "Резюме присылать на hr@techcompany.io до конца месяца.", [("contact", "hr@techcompany.io")]),
    ("email", "Ответьте, пожалуйста, на tatyana.v@clinic-med.ru", [("contact", "tatyana.v@clinic-med.ru")]),
    ("email", "Для прессы: press@newslab.media", [("contact", "press@newslab.media")]),
    # ---------- MODERATION: safe (bool) ----------
    ("moderation", "Лучший магазин электроники! Заказ пришёл за два дня, рекомендую.", [("safe", True)]),
    ("moderation", "Все такие, как он, одинаковые — их место на необитаемом острове.", [("safe", False)]),
    ("moderation", "Отличная статья про квантовые вычисления, спасибо автору!", [("safe", True)]),
    ("moderation", "Куплю базу паспортов с фото, пишите в личку, оплата криптой.", [("safe", False)]),
    ("moderation", "Собрание отдела в четверг в 11:00, повестка на портале.", [("safe", True)]),
    ("moderation", "Найду тебя по IP, сломаю руки, будешь помнить этот разговор.", [("safe", False)]),
    ("moderation", "Рецепт борща: свёкла, капуста, картофель, томатная паста и чеснок.", [("safe", True)]),
    ("moderation", "Погода на выходные обещает плюс двадцать, отличное время для прогулки.", [("safe", True)]),
    ("moderation", "Ты абсолютно бездарный разработчик, твой код — мусор, иди мой полы.", [("safe", False)]),
    ("moderation", "Итоги квартала: выручка выросла на 12 процентов, спасибо команде.", [("safe", True)]),
    ("moderation", "Переведите 5000 на этот кошелёк, и ваш аккаунт разблокируют в течение часа.", [("safe", False)]),
    ("moderation", "Какой сегодня прекрасный день для пробежки в парке!", [("safe", True)]),
    ("moderation", "Заходи на мой сайт, там всё бесплатно без регистрации: bit.ly/free-stuff", [("safe", False)]),
    ("moderation", "Поздравляем с днём рождения, команда вас ценит!", [("safe", True)]),
    ("moderation", "УВЕЛИЧЬ ДОХОД В 10 РАЗ ЗА НЕДЕЛЮ! ЖМИ ССЫЛКУ, ПРЕДЛОЖЕНИЕ ОГРАНИЧЕНО!", [("safe", False)]),
    # ---------- MODERATION: violation ----------
    ("moderation", "КУПИМ ВАШИ ЗОЛОТЫЕ УКРАШЕНИЯ ДОРОГО! СРОЧНО! ПЕРЕХОДИТЕ ПО ССЫЛКЕ!", [("violation", "spam")]),
    ("moderation", "Ты бездарность, твои проекты — позорище, иди подметай дворы.", [("violation", "harassment")]),
    ("moderation", "Я запомню твоё лицо. Встретимся однажды — разговор будет коротким.", [("violation", "violence")]),
    ("moderation", "Вы выиграли главный приз! Оплатите доставку 990 и получите миллион.", [("violation", "fraud")]),
    ("moderation", "Итоги квартала: план выполнен на 103 процента, премия всем отделам.", [("violation", "none")]),
    ("moderation", "Смотри откровенные фото знаменитостей без цензуры, 18+, по ссылке.", [("violation", "adult")]),
    ("moderation", "Куплю справку о прививке с печатями, недорого, самовывоз.", [("violation", "fraud")]),
    ("moderation", "Спасибо за подробный разбор, теперь понятно, как настроить бэкапы.", [("violation", "none")]),
    ("moderation", "Все программисты — тупые задроты, нормальная работа только у грузчиков.", [("violation", "harassment")]),
    ("moderation", "ПРОДАМ АЙФОН ЗА ПОЛЦЕНЫ, ОТПРАВКА СЕГОДНЯ, ПИШИТЕ В ЛИЧКУ КАЖДОМУ!", [("violation", "spam")]),
    ("moderation", "Если ещё раз увидлю тебя у нашей школы — порву, понял?", [("violation", "violence")]),
    ("moderation", "Расписание тренировок на неделю: пн — ноги, ср — спина, пт — грудь.", [("violation", "none")]),
    ("moderation", "Скину откровенные фото за 500, всё конфиденциально, только для взрослых.", [("violation", "adult")]),
    ("moderation", "Гарантирую возврат денег через волшебный кошелёк, просто переведите сумму вперёд.", [("violation", "fraud")]),
    ("moderation", "Обсуждаем архитектуру нового сервиса: очередь на Kafka, хранение в S3.", [("violation", "none")]),
]


# ---------------------------------------------------------------------------
# Построение вопросов
# ---------------------------------------------------------------------------
def build_questions(qids):
    from src.leeth.primitives import ChoiceQuestion, ExtractQuestion, NoulQuestion, ScoreQuestion

    builders = {
        "department": lambda: ChoiceQuestion(id="department", question=Q_DEPARTMENT, choices=DEPT_CHOICES),
        "urgency": lambda: ScoreQuestion(id="urgency", question=Q_URGENCY, levels=3, rubric=URGENCY_RUBRIC),
        "churn": lambda: NoulQuestion(id="churn", question=Q_CHURN),
        "etype": lambda: ChoiceQuestion(id="etype", question=Q_EMAIL_TYPE, choices=EMAIL_TYPE_CHOICES),
        "priority": lambda: ScoreQuestion(id="priority", question=Q_PRIORITY, levels=3, rubric=PRIORITY_RUBRIC),
        "contact": lambda: ExtractQuestion(id="contact", question=Q_CONTACT, target_type="email"),
        "safe": lambda: NoulQuestion(id="safe", question=Q_SAFE),
        "violation": lambda: ChoiceQuestion(id="violation", question=Q_VIOLATION, choices=VIOLATION_CHOICES),
    }
    return [builders[qid]() for qid in qids]


def laya_questions(qids):
    qs = {}
    for qid in qids:
        if qid == "department":
            qs[qid] = {"type": "choice", "instructions": Q_DEPARTMENT, "criteria": DEPT_CRITERIA}
        elif qid == "urgency":
            qs[qid] = {"type": "score", "instructions": Q_URGENCY, "criteria": URGENCY_RUBRIC}
        elif qid == "churn":
            qs[qid] = {"type": "noul", "instructions": Q_CHURN}
        elif qid == "etype":
            qs[qid] = {"type": "choice", "instructions": Q_EMAIL_TYPE, "criteria": EMAIL_TYPE_CRITERIA}
        elif qid == "priority":
            qs[qid] = {"type": "score", "instructions": Q_PRIORITY, "criteria": PRIORITY_RUBRIC}
        elif qid == "safe":
            qs[qid] = {"type": "noul", "instructions": Q_SAFE}
        elif qid == "violation":
            qs[qid] = {"type": "choice", "instructions": Q_VIOLATION, "criteria": VIOLATION_CRITERIA}
        # contact (extract): у Laya нет extract-головы — пропускается
    return qs


# ---------------------------------------------------------------------------
# Метрики
# ---------------------------------------------------------------------------
def ece(confs, corrects, n_bins=10):
    bins = [[] for _ in range(n_bins)]
    for c, o in zip(confs, corrects):
        bins[min(n_bins - 1, int(c * n_bins))].append((c, o))
    total = len(confs)
    if total == 0:
        return 0.0
    e = 0.0
    for b in bins:
        if not b:
            continue
        avg_c = sum(c for c, _ in b) / len(b)
        acc = sum(o for _, o in b) / len(b)
        e += len(b) / total * abs(acc - avg_c)
    return e


def score_answer(qid, gold, pred, conf):
    """pred/conf уже нормализованы раннером; возвращает (correct, conf_in_pred)."""
    qtype = QTYPE[qid]
    if qtype == "choice":
        return pred == gold, conf
    if qtype == "noul":
        return pred == bool(gold), conf
    if qtype == "score":
        return pred == int(gold), conf
    if qtype == "extract":
        return (gold.lower() in (pred or "").lower()), conf
    raise ValueError(qid)


# ---------------------------------------------------------------------------
# Раннер Leeth (текущий прод-чекпоинт, zero-shot на новых сферах)
# ---------------------------------------------------------------------------
def run_leeth(cases):
    from src.leeth.runtime import leeth

    if not leeth.initialize():
        print(f"leeth init error: {leeth._init_error}")
        sys.exit(1)
    records = []
    t_all = time.perf_counter()
    for sphere, state, qspecs in cases:
        qids = [qid for qid, _ in qspecs]
        questions = build_questions(qids)
        t0 = time.perf_counter()
        batch = leeth.decide(prod_envelope(state), questions)
        ms = (time.perf_counter() - t0) * 1000.0
        for (qid, gold), q in zip(qspecs, questions):
            res = batch[qid]
            qtype = QTYPE[qid]
            if qtype == "choice":
                pred, conf = res.choice, res.confidence
            elif qtype == "noul":
                pred, conf = res.value, (res.probability if res.probability >= 0.5 else 1.0 - res.probability)
            elif qtype == "score":
                pred, conf = res.level, max(res.probabilities.values()) if res.probabilities else 0.0
            else:  # extract
                pred, conf = res.text, res.confidence
            ok, conf_in_pred = score_answer(qid, gold, pred, conf)
            records.append({
                "runner": "leeth", "model": "prod", "sphere": sphere, "qid": qid,
                "qtype": qtype, "gold": gold, "pred": str(pred), "conf": round(conf_in_pred, 4),
                "correct": bool(ok), "ms": round(ms / len(qspecs), 2),
            })
    total_ms = (time.perf_counter() - t_all) * 1000
    print(f"[leeth] готово за {total_ms:.0f} мс", flush=True)
    return records


# ---------------------------------------------------------------------------
# Раннер Laya (Router, 4 варианта чекпоинта)
# ---------------------------------------------------------------------------
def run_laya(cases, model_name):
    import gc

    from laya import Router

    # Тяжёлые чекпоинты (multilingual/typed-decisions) могут не влезть в CUDA-путь
    # при дефиците commit-памяти (os error 1455) — грузим их сразу на CPU.
    device_order = ("cpu", "cuda") if model_name in ("multilingual", "typed-decisions") else ("cuda", "cpu")
    router = None
    device_note = device_order[0]
    for dev in device_order:
        try:
            gc.collect()
            router = Router(device=dev)
            device_note = dev
            break
        except Exception as e:
            print(f"[laya:{model_name}] Router({dev}) недоступен ({e})", flush=True)
    if router is None:
        raise RuntimeError(f"[laya:{model_name}] ни один device не доступен")

    records = []
    t_all = time.perf_counter()
    n_predicts = 0
    for sphere, state, qspecs in cases:
        qids = [qid for qid, _ in qspecs]
        qs = laya_questions(qids)
        if not qs:
            continue
        try:
            t0 = time.perf_counter()
            r = router.predict(state, qs, model=None if model_name == "auto" else model_name)
            ms = (time.perf_counter() - t0) * 1000
        except OSError as e:
            # Чекпоинт не загрузился на этом устройстве — пересоздаём роутер на CPU.
            print(f"[laya:{model_name}] OSError на {device_note} ({e}); перезапуск на CPU", flush=True)
            gc.collect()
            router = Router(device="cpu")
            device_note = "cpu"
            t0 = time.perf_counter()
            r = router.predict(state, qs, model=None if model_name == "auto" else model_name)
            ms = (time.perf_counter() - t0) * 1000
        n_predicts += 1
        for qid, gold in qspecs:
            if qid not in qs:
                continue  # extract: у Laya нет головы
            ans = r["answers"][qid]
            qtype = QTYPE[qid]
            if qtype == "choice":
                pred = ans.get("choice")
                probs = ans.get("probabilities") or {}
                conf = max(probs.values()) if probs else float(ans.get("confidence") or 0.0)
            elif qtype == "score":
                probs = {int(k): float(v) for k, v in (ans.get("probabilities") or {}).items()}
                if not probs:
                    continue
                pred = max(probs, key=probs.get) + 1  # Laya 0-индексация -> уровень 1..3
                conf = max(probs.values())
            else:  # noul
                p = float(ans.get("noul", 0.5))
                pred = p >= 0.5
                conf = p if p >= 0.5 else 1.0 - p
            ok, conf_in_pred = score_answer(qid, gold, pred, conf)
            records.append({
                "runner": "laya", "model": model_name, "sphere": sphere, "qid": qid,
                "qtype": qtype, "gold": gold, "pred": str(pred), "conf": round(conf_in_pred, 4),
                "correct": bool(ok), "ms": round(ms / len(qspecs), 2), "device": device_note,
            })
    total_ms = (time.perf_counter() - t_all) * 1000
    print(f"[laya:{model_name}] готово за {total_ms:.0f} мс ({n_predicts} predict-вызовов)", flush=True)
    return records


# ---------------------------------------------------------------------------
# Отчёт
# ---------------------------------------------------------------------------
def summarize(records):
    rows = {}
    for r in records:
        key = (r["runner"], r.get("model", "prod"), r["sphere"], r["qtype"])
        rows.setdefault(key, []).append(r)
    lines = []
    for key in sorted(rows):
        grp = rows[key]
        n = len(grp)
        acc = sum(g["correct"] for g in grp) / n
        e = ece([g["conf"] for g in grp], [g["correct"] for g in grp])
        ms = sum(g["ms"] for g in grp) / n
        runner, model, sphere, qtype = key
        label = f"{runner}:{model}" if runner == "laya" else "leeth:prod"
        lines.append(f"{label:<28} {sphere:<11} {qtype:<8} n={n:<4} acc={acc*100:5.1f}%  ECE={e:.3f}  {ms:6.1f} мс/вопрос")
    return lines


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"
    laya_single = mode.startswith("laya:") and ":" in mode
    if laya_single:
        laya_models = [mode.split(":", 1)[1]]
        mode = "laya"
    all_records = []
    if OUT_JSON.exists():
        all_records = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        print(f"догружено из прошлого прогона: {len(all_records)} записей", flush=True)

    def save():
        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        OUT_JSON.write_text(json.dumps(all_records, ensure_ascii=False, indent=1), encoding="utf-8")

    if mode in ("both", "leeth") and not any(r["runner"] == "leeth" for r in all_records):
        all_records += run_leeth(CASES)
        save()
    if mode in ("both", "laya"):
        for model_name in (laya_models if laya_single else ("auto", "english", "multilingual", "typed-decisions")):
            if any(r["runner"] == "laya" and r.get("model") == model_name for r in all_records):
                continue
            all_records += run_laya(CASES, model_name)
            save()

    print("\n=== МУЛЬТИДОМЕННАЯ БАТАРЕЯ: Leeth zero-shot vs Laya ===")
    for line in summarize(all_records):
        print(line)
    print(f"\nсырые записи: {OUT_JSON} ({len(all_records)} шт.)")


if __name__ == "__main__":
    main()