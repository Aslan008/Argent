"""Does Leeth sort posts / bookmarks / messages by CONTENT, and how much do a few examples help?

Measures the yes/no head on "is this text about X?" questions the way ``leeth_map`` / ``leeth_scan_page`` use it:

  * ranking quality (AUC) per question;
  * what the fixed bounds (accept 0.7 / reject 0.3) give: recall, false "yes", share of "unsure";
  * what a cut learnt from a few labelled examples gives (``bulk.cut_from_examples``): same figures, averaged over random picks.

Data: the built-in bench (6 topics x 8 posts, RU + EN) or your own JSON file::

    [{"criterion": "Это жалоба на доставку?", "positive": ["...", "..."], "negative": ["...", "..."]}, ...]

Run (CPU, a minute)::

    .venv/Scripts/python.exe scripts/leeth_content_bench.py [--data my_posts.json] [--model .argent/models/leeth2/final] [--picks 20]
"""

import argparse
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.leeth import bulk  # noqa: E402

BUILTIN_QUESTIONS = {
    "delivery_complaint": "Это жалоба на доставку?",
    "recipe": "Это рецепт?",
    "football": "Это пост про футбол?",
    "gadget": "Это обзор гаджета или техники?",
    "politics": "Это политическая новость?",
    "job": "Это вакансия?",
}

BUILTIN_POSTS = {
    "delivery_complaint": [
        "Курьер опоздал на три часа, еда остыла, на звонки не отвечал. Больше не буду заказывать в этом сервисе.",
        "Заказ не доставили, трекинг не обновляется вторую неделю, поддержка присылает отписки. Верните деньги!",
        "Посылку отдали соседям без моего ведома, а в коробке оказалась разбитая чашка. Доставка просто ужасная.",
        "Привезли не тот товар и ещё пытались заставить меня самого оформлять возврат. Сервис доставки — позор.",
        "Ждал пиццу полтора часа, привезли холодную и без напитков. Курьер даже не извинился.",
        "Delivery was a disaster: the parcel arrived two weeks late and the box was crushed. Support did not help.",
        "Вчера заказал мебель, доставку перенесли уже трижды, и никто не предупредил. Просто издевательство.",
        "Почта потеряла моё отправление, а на жалобу ответили, что «срок рассмотрения 30 дней». Возмутительно.",
    ],
    "recipe": [
        "Тесто для пиццы: 300 г муки, 200 мл воды, щепотка соли и дрожжи. Дать подойти час и раскатать.",
        "Борщ по бабушкиному рецепту: говядина на кости, свёкла, капуста, морковь, лук и зажарка на сале.",
        "Как приготовить идеальную карбонару: желтки, пекорино, гуанчале и чёрный перец, без сливок.",
        "Шарлотка за 40 минут: яйца, сахар, мука и яблоки. Выпекать при 180 градусах до румяной корочки.",
        "Simple pancake recipe: flour, milk, two eggs and a pinch of salt; fry on a hot pan for a minute per side.",
        "Салат цезарь дома: куриная грудка, пармезан, сухарики, листья салата и соус на анчоусах.",
        "Запечённая сёмга с лимоном и укропом: 20 минут в духовке при 200 градусах.",
        "Домашний хлеб без замеса: мука, вода, соль, дрожжи и 12 часов времени.",
    ],
    "football": [
        "Вчера наша команда выиграла со счётом 3:0, молодой нападающий оформил дубль на дебюте.",
        "Тренер объявил состав на матч Лиги чемпионов: в защите вернулся капитан после травмы.",
        "Трансферное окно: клуб подписал вингера из Бундеслиги за 25 миллионов евро.",
        "The striker scored a hat-trick and the stadium went wild in the final minutes of the derby.",
        "Судья назначил спорный пенальти на 90-й минуте, и болельщики до сих пор спорят о решении.",
        "Сборная прошла в плей-офф чемпионата после победы над соперником в серии пенальти.",
        "Вратарь отразил два пенальти подряд, и клуб вышел в финал кубка страны.",
        "Футбольный сезон стартует в эти выходные: календарь первых туров уже опубликован.",
    ],
    "gadget": [
        "Обзор нового смартфона: экран 120 Гц, камера 200 Мп, батарея держит два дня.",
        "Ноутбук с процессором нового поколения оказался быстрее прежней модели на 30 процентов.",
        "Наушники с шумоподавлением: звук отличный, но кейс слишком скользкий и маленький.",
        "Review: this tablet has a gorgeous display but the software updates are painfully slow.",
        "Умные часы получили измерение давления, но точность оставляет желать лучшего.",
        "Видеокарта нового поколения: прирост в играх 40 процентов, потребление выросло.",
        "Тест роутера Wi-Fi 7: скорость выше, но настройка требует терпения.",
        "Электросамокат с запасом хода 80 км: впечатления после месяца использования.",
    ],
    "politics": [
        "Парламент принял закон о бюджете на следующий год после долгих споров депутатов.",
        "Президент провёл встречу с главами регионов и обсудил экономические планы.",
        "Выборы назначены на осень: партии начали формировать списки кандидатов.",
        "The government announced a new tax reform that will take effect next year.",
        "Министр заявил о планах повысить пенсии и индексировать зарплаты бюджетников.",
        "Переговоры по торговому соглашению между странами продлились двое суток.",
        "Оппозиция потребовала отставки министра после скандала с закупками.",
        "Суд рассмотрел дело о нарушении избирательного законодательства.",
    ],
    "job": [
        "Ищем Python-разработчика в команду: удалёнка, зарплата от 250 тысяч, опыт от трёх лет.",
        "Вакансия: менеджер по продажам, офис в центре, оклад плюс проценты, график 5/2.",
        "We are hiring a senior data engineer, remote friendly, competitive salary.",
        "Требуется бухгалтер на полный день, опыт работы с 1С обязателен.",
        "Открыта вакансия дизайнера интерфейсов: портфолио обязательно, гибкий график.",
        "Компания набирает курьеров на подработку: свободный график, выплаты каждую неделю.",
        "Нужен администратор в салон красоты, обучение за счёт компании.",
        "Стажировка для студентов: аналитик данных, оплачиваемая, старт в сентябре.",
    ],
}


def builtin_tasks():
    tasks = []
    for topic, q in BUILTIN_QUESTIONS.items():
        pos = BUILTIN_POSTS[topic]
        neg = [x for t, xs in BUILTIN_POSTS.items() if t != topic for x in xs]
        tasks.append({"criterion": q, "positive": pos, "negative": neg})
    return tasks


def auc(pos, neg):
    return sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg))


def logit(p):
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="JSON file with tasks (see the module docstring); the built-in bench when omitted")
    ap.add_argument("--model", default=str(ROOT / ".argent" / "models" / "leeth2" / "final"))
    ap.add_argument("--picks", type=int, default=20, help="random example picks per task and setting")
    ap.add_argument("--accept", type=float, default=0.7)
    ap.add_argument("--reject", type=float, default=0.3)
    args = ap.parse_args()

    from src.leeth2.runtime import Leeth2Runtime

    tasks = json.loads(Path(args.data).read_text(encoding="utf-8")) if args.data else builtin_tasks()
    rt = Leeth2Runtime.load(Path(args.model))

    print(f"{'question':44s} {'AUC':>5s} | fixed {args.accept}/{args.reject}: recall  false-yes  unsure | learnt from 2+2 examples: recall  false-yes  unsure")
    totals = {"fixed": [0, 0, 0, 0, 0], "ex": [0, 0, 0, 0, 0]}  # tp, P, fp, N, unsure
    for k, t in enumerate(tasks):
        pos, neg = list(t["positive"]), list(t["negative"])
        probs = [float(r.probability) for r in rt.judge_many(pos + neg, t["criterion"])]
        pp, pn = probs[: len(pos)], probs[len(pos):]
        a = auc(pp, pn)
        yes_p = sum(p >= args.accept for p in pp)
        yes_n = sum(p >= args.accept for p in pn)
        uns = sum(args.reject < p < args.accept for p in pp + pn)
        # learnt cut, averaged over random example picks
        tp = fp = un = P = N = 0
        for seed in range(args.picks):
            rnd = random.Random(seed)
            ip = rnd.sample(range(len(pp)), min(2, len(pp) - 1))
            ineg = rnd.sample(range(len(pn)), min(2, len(pn) - 1))
            cut = bulk.cut_from_examples([pp[i] for i in ip], [pn[i] for i in ineg])
            for i, p in enumerate(pp):
                if i in ip:
                    continue
                P += 1
                z = logit(p)
                tp += z >= cut["yes"]
                un += cut["no"] < z < cut["yes"]
            for i, p in enumerate(pn):
                if i in ineg:
                    continue
                N += 1
                z = logit(p)
                fp += z >= cut["yes"]
                un += cut["no"] < z < cut["yes"]
        print(f"{t['criterion'][:44]:44s} {a:5.2f} |                       {yes_p / len(pp):6.2f}  {yes_n / len(pn):9.3f}  {uns / len(probs):6.2f} |"
              f"                            {tp / P:6.2f}  {fp / N:9.3f}  {un / (P + N):6.2f}")
        for key, vals in (("fixed", (yes_p, len(pp), yes_n, len(pn), uns, len(probs))), ("ex", (tp, P, fp, N, un, P + N))):
            tot = totals[key]
            tot[0] += vals[0] / vals[1]
            tot[2] += vals[2] / vals[3]
            tot[4] += vals[4] / vals[5]
            tot[1] += 1
    n = max(1, len(tasks))
    print(f"{'mean':44s}       |                       {totals['fixed'][0] / n:6.2f}  {totals['fixed'][2] / n:9.3f}  {totals['fixed'][4] / n:6.2f} |"
          f"                            {totals['ex'][0] / n:6.2f}  {totals['ex'][2] / n:9.3f}  {totals['ex'][4] / n:6.2f}")


if __name__ == "__main__":
    main()
