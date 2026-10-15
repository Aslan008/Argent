"""NPC decisions: follow a written policy against a world state (in-context rule execution), and pick an in-character reply.

Nothing about the NPC is baked into the weights: the rules, their priority and the persona are all in the text,
so the model learns to *execute what it is told* — which is what a game designer needs from it.
"""

from __future__ import annotations

import random
from typing import Any, Callable, Dict, List, Optional, Tuple

from ...schema import Sample

NAMES = {
    "en": ["Elara", "Borin", "Mira", "Tobias", "Kael", "Sunny", "Garrick", "Lyra", "Odo", "Vex", "Nessa", "Hakon", "Tilda", "Rurik", "Isolde"],
    "ru": ["Элара", "Борин", "Мира", "Тобиас", "Каэль", "Санни", "Гаррик", "Лира", "Одо", "Векс", "Несса", "Хакон", "Тильда", "Рюрик", "Изольда"],
}
ROLES = {
    "en": ["blacksmith", "guard", "merchant", "healer", "farmer", "innkeeper", "scout", "miner", "fisherman", "bard"],
    "ru": ["кузнец", "стражник", "торговец", "лекарь", "фермер", "трактирщик", "разведчик", "шахтёр", "рыбак", "бард"],
}
LOCATIONS = ["home", "market", "tavern", "forest", "road", "mine", "gate"]
LOC_RU = {"home": "дом", "market": "рынок", "tavern": "таверна", "forest": "лес", "road": "дорога", "mine": "шахта", "gate": "ворота"}
ITEMS = ["potion", "food", "sword", "key", "map", "torch"]
ITEM_RU = {"potion": "зелье", "food": "еда", "sword": "меч", "key": "ключ", "map": "карта", "torch": "факел"}
TIMES = ["day", "evening", "night"]
TIME_RU = {"day": "день", "evening": "вечер", "night": "ночь"}


class World:
    def __init__(self, rng: random.Random):
        self.health = rng.randint(5, 100)
        self.stamina = rng.randint(5, 100)
        self.hunger = rng.randint(0, 100)
        self.gold = rng.choice([0, 5, 20, 50, 120, 300, 800])
        self.time = rng.choice(TIMES)
        self.loc = rng.choice(LOCATIONS)
        self.enemy = rng.choice([None, None, rng.randint(1, 10)])  # distance in tiles
        self.items = sorted(rng.sample(ITEMS, rng.randint(0, 4)))
        self.relation = rng.randint(-3, 3)

    def describe(self, lang: str, rng: random.Random) -> str:
        if lang == "ru":
            parts = [f"Здоровье: {self.health}/100", f"Выносливость: {self.stamina}", f"Голод: {self.hunger}", f"Золото: {self.gold}", f"Время: {TIME_RU[self.time]}", f"Место: {LOC_RU[self.loc]}"]
            parts.append("Враг рядом: в " + str(self.enemy) + " клетках" if self.enemy else "Врагов поблизости нет")
            parts.append("Инвентарь: " + (", ".join(ITEM_RU[i] for i in self.items) if self.items else "пусто"))
            parts.append(f"Отношение игрока: {self.relation:+d}")
        else:
            parts = [f"Health: {self.health}/100", f"Stamina: {self.stamina}", f"Hunger: {self.hunger}", f"Gold: {self.gold}", f"Time: {self.time}", f"Location: {self.loc}"]
            parts.append(f"Enemy nearby: {self.enemy} tiles away" if self.enemy else "No enemies nearby")
            parts.append("Inventory: " + (", ".join(self.items) if self.items else "empty"))
            parts.append(f"Player relation: {self.relation:+d}")
        rng.shuffle(parts)
        return "; ".join(parts)


# condition = (text_en(t), text_ru(t), predicate(world, t), threshold sampler)
def _c(en, ru, pred, thr=None):
    return {"en": en, "ru": ru, "pred": pred, "thr": thr}


CONDS: List[Dict[str, Any]] = [
    _c("health is below {t}", "здоровье ниже {t}", lambda w, t: w.health < t, lambda r: r.choice([20, 30, 40, 50])),
    _c("stamina is below {t}", "выносливость ниже {t}", lambda w, t: w.stamina < t, lambda r: r.choice([20, 30, 40])),
    _c("hunger is above {t}", "голод выше {t}", lambda w, t: w.hunger > t, lambda r: r.choice([50, 60, 70, 80])),
    _c("it is night", "сейчас ночь", lambda w, t: w.time == "night"),
    _c("it is daytime", "сейчас день", lambda w, t: w.time == "day"),
    _c("an enemy is within {t} tiles", "враг ближе чем в {t} клетках", lambda w, t: w.enemy is not None and w.enemy <= t, lambda r: r.choice([2, 3, 5, 7])),
    _c("there are no enemies nearby", "врагов поблизости нет", lambda w, t: w.enemy is None),
    _c("you have a potion", "у вас есть зелье", lambda w, t: "potion" in w.items),
    _c("you have food", "у вас есть еда", lambda w, t: "food" in w.items),
    _c("you carry a sword", "у вас есть меч", lambda w, t: "sword" in w.items),
    _c("you have at least {t} gold", "у вас не меньше {t} золота", lambda w, t: w.gold >= t, lambda r: r.choice([20, 50, 100, 300])),
    _c("you are at the market", "вы на рынке", lambda w, t: w.loc == "market"),
    _c("you are at home", "вы дома", lambda w, t: w.loc == "home"),
    _c("you are in the forest", "вы в лесу", lambda w, t: w.loc == "forest"),
    _c("the player is friendly", "игрок настроен дружелюбно", lambda w, t: w.relation >= 1),
    _c("the player is hostile", "игрок настроен враждебно", lambda w, t: w.relation <= -1),
]
# actions: key -> (en, ru, needs) ; some actions are only sensible with an item, the policy decides, not common sense
ACTIONS: Dict[str, Tuple[str, str]] = {
    "drink_potion": ("drink a potion", "выпить зелье"), "eat": ("eat food", "поесть"), "flee": ("flee", "бежать"), "fight": ("fight the enemy", "сражаться с врагом"),
    "sleep": ("go to sleep", "лечь спать"), "trade": ("offer to trade", "предложить торговлю"), "greet": ("greet the player", "поприветствовать игрока"),
    "ignore": ("ignore the player", "игнорировать игрока"), "wander": ("wander around", "бродить по округе"), "patrol": ("patrol the area", "патрулировать территорию"),
    "hide": ("hide", "спрятаться"), "call_guards": ("call the guards", "позвать стражу"), "rest": ("rest", "отдохнуть"), "go_home": ("go home", "идти домой"),
    "buy_sword": ("buy a sword", "купить меч"), "work": ("keep working", "продолжать работу"), "warn": ("warn the player", "предупредить игрока"),
}


def _expert(world: World, rules: List[Tuple[Dict[str, Any], Optional[int], str]], default: str) -> Tuple[str, Optional[int]]:
    for i, (cond, t, act) in enumerate(rules):
        if cond["pred"](world, t):
            return act, i
    return default, None


def gen_policy(rng: random.Random) -> Optional[List[Sample]]:
    lang = rng.choice(["en", "ru"])
    world = World(rng)
    n_rules = rng.randint(3, 6)
    conds = rng.sample(CONDS, n_rules)
    acts = list(ACTIONS)
    rules: List[Tuple[Dict[str, Any], Optional[int], str]] = []
    used = set()
    for c in conds:
        t = c["thr"](rng) if c["thr"] else None
        a = rng.choice([x for x in acts if x not in used])
        used.add(a)
        rules.append((c, t, a))
    default = rng.choice([x for x in acts if x not in used])
    act, which = _expert(world, rules, default)
    # priorities: sometimes written out of order with explicit numbers
    explicit = rng.random() < 0.35
    items = []
    for i, (c, t, a) in enumerate(rules):
        txt = c[lang].format(t=t)
        if lang == "ru":
            sent = f"Если {txt} — {ACTIONS[a][1]}."
        else:
            sent = f"If {txt}: {ACTIONS[a][0]}."
        items.append((i, sent))
    if explicit:
        order = list(range(len(items)))
        rng.shuffle(order)
        rule_lines = []
        for pos in order:
            prio = pos + 1
            rule_lines.append((pos, (f"[Приоритет {prio}] " if lang == "ru" else f"[Priority {prio}] ") + items[pos][1]))
        header = "Правила (меньший номер приоритета важнее; применяется первое подходящее по приоритету):" if lang == "ru" else "Rules (a lower priority number wins; apply the first matching rule by priority):"
    else:
        rule_lines = [(i, f"{i + 1}) {s}") for i, s in items]
        header = "Правила (применяйте первое подходящее по порядку):" if lang == "ru" else "Rules (apply the first one that matches, in order):"
    other = ("Иначе: " + ACTIONS[default][1] + ".") if lang == "ru" else ("Otherwise: " + ACTIONS[default][0] + ".")
    name = rng.choice(NAMES[lang])
    role = rng.choice(ROLES[lang])
    persona = f"{name} — {role}." if lang == "ru" else f"{name} the {role}."
    body = "\n".join(l for _, l in rule_lines)
    state = f"{persona}\n{world.describe(lang, rng)}\n{header}\n{body}\n{other}"
    pool = [ACTIONS[a][1 if lang == "ru" else 0] for _, _, a in rules] + [ACTIONS[default][1 if lang == "ru" else 0]]
    extras = [ACTIONS[a][1 if lang == "ru" else 0] for a in acts if a not in used and a != default]
    rng.shuffle(extras)
    options = list(dict.fromkeys(pool + extras[: rng.randint(0, 3)]))
    rng.shuffle(options)
    target = options.index(ACTIONS[act][1 if lang == "ru" else 0])
    q = rng.choice(["Какое действие должен выполнить персонаж?", "Что делает NPC сейчас?", "Выберите действие по правилам."] if lang == "ru" else ["What should the NPC do now?", "Which action follows from the rules?", "Choose the NPC's next action."])
    out = [Sample("npc_policy", "npc", lang, "choice", state, q, options, target)]
    # does a particular rule apply?
    j = rng.randrange(n_rules)
    c, t, a = rules[j]
    holds = bool(c["pred"](world, t))
    txt = c[lang].format(t=t)
    qn = (f"Выполняется ли условие «{txt}» прямо сейчас?" if lang == "ru" else f"Does the condition “{txt}” hold right now?")
    out.append(Sample("npc_condition", "npc", lang, "noul", state, qn, None, float(holds)))
    # which rule fires -> extract the rule sentence from the state
    if which is not None and not explicit:
        line = [l for i, l in rule_lines if i == which][0]
        qe = "Какое правило сработало? Скопируйте его." if lang == "ru" else "Which rule fires? Copy it from the text."
        i0 = state.find(line)
        if i0 >= 0:
            out.append(Sample("npc_rule", "npc", lang, "extract", state, qe, None, [i0, i0 + len(line)]))
    return out


# ---------------------------------------------------------------------------------------------- persona replies
PERSONAS = {
    "friendly": ("friendly", "добродушный"), "grumpy": ("grumpy", "ворчливый"), "fearful": ("fearful", "трусливый"), "greedy": ("greedy", "жадный"), "noble": ("noble", "благородный"),
}
UTTER = {
    "greet": ("Hello there!", "Привет!"), "ask_price": ("How much for the sword?", "Сколько стоит этот меч?"), "threat": ("Hand over your gold or else.", "Отдай золото, или пожалеешь."),
    "thanks": ("Thank you for your help.", "Спасибо за помощь."), "quest": ("Do you have any work for me?", "Есть для меня работа?"), "farewell": ("Goodbye, see you later.", "До свидания, ещё увидимся."),
}
# reply bank: (utterance, persona) -> list of acceptable replies; wrong ones are drawn from other cells
REPLIES: Dict[Tuple[str, str], Tuple[str, str]] = {
    ("greet", "friendly"): ("Well met, friend! Come in, come in!", "Добро пожаловать, друг! Проходи!"),
    ("greet", "grumpy"): ("Hmph. What do you want?", "Хм. Чего тебе?"),
    ("greet", "fearful"): ("Oh! Y-you startled me... hello.", "О! В-вы меня напугали... здравствуйте."),
    ("greet", "greedy"): ("Greetings! Got any coin to spend?", "Приветствую! Монеты на покупки есть?"),
    ("greet", "noble"): ("Good day to you, traveler.", "Доброго дня вам, путник."),
    ("ask_price", "friendly"): ("For you? A fair price, thirty gold.", "Для тебя? Честная цена — тридцать золотых."),
    ("ask_price", "grumpy"): ("Sixty gold, and no haggling.", "Шестьдесят золотых, и не торгуйся."),
    ("ask_price", "fearful"): ("T-thirty... or whatever you think is fair, please.", "Т-тридцать... или сколько сочтёте нужным, пожалуйста."),
    ("ask_price", "greedy"): ("A hundred gold — cheap at the price!", "Сто золотых — просто даром!"),
    ("ask_price", "noble"): ("Fifty gold; quality has its worth.", "Пятьдесят золотых; качество чего-то стоит."),
    ("threat", "friendly"): ("Please, friend, there is no need for this.", "Прошу, друг, в этом нет нужды."),
    ("threat", "grumpy"): ("Try it and you will lose more than gold.", "Попробуй — потеряешь больше, чем золото."),
    ("threat", "fearful"): ("Please don't hurt me! Take it, take everything!", "Не бейте! Берите, берите всё!"),
    ("threat", "greedy"): ("My gold? Over my dead body!", "Моё золото? Только через мой труп!"),
    ("threat", "noble"): ("You dishonour yourself. Draw your weapon, then.", "Вы бесчестите себя. Тогда обнажайте оружие."),
    ("thanks", "friendly"): ("Anytime, my friend!", "Всегда рад, друг!"),
    ("thanks", "grumpy"): ("Yeah, yeah. Don't mention it.", "Ага, ага. Не стоит."),
    ("thanks", "fearful"): ("Oh, I-it was nothing, really...", "О, э-это пустяки, правда..."),
    ("thanks", "greedy"): ("Gratitude is nice. Coins are nicer.", "Благодарность — это хорошо. Монеты — лучше."),
    ("thanks", "noble"): ("It was my duty and my honour.", "Это был мой долг и честь."),
    ("quest", "friendly"): ("Actually, yes! I could use a hand with the wolves.", "Вообще-то да! Мне пригодится помощь с волками."),
    ("quest", "grumpy"): ("Work? Fetch my hammer from the mine. Don't dawdle.", "Работа? Принеси мой молот из шахты. Не мешкай."),
    ("quest", "fearful"): ("Maybe... someone must check the cellar. I can't.", "Может быть... кто-то должен проверить подвал. Я не могу."),
    ("quest", "greedy"): ("Work pays well — if you split the loot with me.", "Работа хорошо оплачивается — если поделишься добычей со мной."),
    ("quest", "noble"): ("There is a matter of honour that needs a steady hand.", "Есть дело чести, где нужна твёрдая рука."),
    ("farewell", "friendly"): ("Safe travels! Come back soon!", "Счастливого пути! Возвращайся скорее!"),
    ("farewell", "grumpy"): ("Finally. Shut the door behind you.", "Наконец-то. Закрой за собой дверь."),
    ("farewell", "fearful"): ("G-goodbye... please be careful out there.", "П-прощайте... будьте осторожны там."),
    ("farewell", "greedy"): ("Bye! Remember: I buy and sell at fair prices.", "Пока! Помни: я покупаю и продаю по честным ценам."),
    ("farewell", "noble"): ("Farewell. May your path be honourable.", "Прощайте. Пусть ваш путь будет достойным."),
}


def gen_persona(rng: random.Random) -> Optional[List[Sample]]:
    lang = rng.choice(["en", "ru"])
    li = 1 if lang == "ru" else 0
    persona = rng.choice(list(PERSONAS))
    utter = rng.choice(list(UTTER))
    name = rng.choice(NAMES[lang])
    role = rng.choice(ROLES[lang])
    k = rng.randint(3, 5)
    wrong_cells = [(u, p) for (u, p) in REPLIES if (u, p) != (utter, persona) and (u == utter or p == persona)]
    rng.shuffle(wrong_cells)
    options = [REPLIES[(utter, persona)][li]] + [REPLIES[c][li] for c in wrong_cells[: k - 1]]
    options = list(dict.fromkeys(options))
    if len(options) < 3:
        return None
    rng.shuffle(options)
    target = options.index(REPLIES[(utter, persona)][li])
    if lang == "ru":
        state = f"{name} — {role}, характер: {PERSONAS[persona][1]}.\nИгрок говорит: «{UTTER[utter][1]}»"
        q = rng.choice(["Какой ответ соответствует характеру персонажа?", "Что ответит NPC?"])
    else:
        state = f"{name} the {role}, personality: {PERSONAS[persona][0]}.\nThe player says: “{UTTER[utter][0]}”"
        q = rng.choice(["Which reply fits the character's personality?", "What does the NPC answer?"])
    out = [Sample("npc_persona", "npc", lang, "choice", state, q, options, target)]
    # noul: does this reply fit the persona?
    cand = rng.choice(options)
    fits = cand == options[target]
    qn = (f"Является ли ответ «{cand}» уместным ответом на слова игрока и соответствует ли он характеру персонажа?" if lang == "ru"
          else f"Is the reply “{cand}” an appropriate answer to what the player said, and in character?")
    out.append(Sample("npc_persona_fit", "npc", lang, "noul", state, qn, None, float(fits)))
    return out


GENERATORS: Dict[str, Tuple[Callable[[random.Random], Optional[List[Sample]]], float]] = {"policy": (gen_policy, 4.0), "persona": (gen_persona, 1.5)}
