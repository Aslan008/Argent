"""NPC rule-following with an OPEN vocabulary.

``npc.py`` teaches the skill with 16 conditions and 17 actions; the model reaches 99.8 % there, which mostly proves that it knows
those 16 and 17.  A game designer writes *their own* stats ("mana", "morale"), their own conditions ("the bridge is raised") and their
own verbs ("cast a ward").  Here the world is a bag of invented numbers, flags, items and creatures, a rule is a composed condition
(comparisons, ranges, negations, AND / OR) and an action is a verb phrase, and the vocabulary comes in two disjoint pools:

    pool A  trained        pool B  never trained -- the measure of whether "execute the rules in the text" transfers

Every label is the first rule whose condition holds (or the default), computed by evaluating the structure that also generated the
sentence, so text and label cannot disagree.  Russian and English.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from ...schema import Sample

# (en, ru) wordings.  Stats are numbers 0..100; flags are true/false facts; creatures stand at a distance; items are carried.
POOLS: Dict[str, Dict[str, List[Tuple[str, str]]]] = {
    "A": {
        "stat": [("mana", "мана"), ("morale", "боевой дух"), ("ammo", "патроны"), ("fuel", "топливо"), ("reputation", "репутация"), ("temperature", "температура"), ("noise", "шум"),
                 ("trust", "доверие"), ("hunger", "голод"), ("alertness", "бдительность"), ("energy", "энергия"), ("armor", "броня"), ("patience", "терпение"), ("light", "освещённость")],
        "flag": [("the bridge is raised", "мост поднят"), ("it is raining", "идёт дождь"), ("an ally is nearby", "рядом союзник"), ("the door is locked", "дверь заперта"), ("the alarm is ringing", "звучит тревога"),
                 ("the shop is open", "магазин открыт"), ("the player is armed", "игрок вооружён"), ("the road is blocked", "дорога перекрыта"), ("the fire is lit", "костёр горит"),
                 ("the moon is up", "взошла луна"), ("the gate is guarded", "ворота охраняются"), ("the player is hurt", "игрок ранен"), ("a storm is coming", "надвигается буря"), ("the well is dry", "колодец пересох")],
        "item": [("lantern", "фонарь"), ("rope", "верёвка"), ("scroll", "свиток"), ("bandage", "бинт"), ("coin pouch", "кошелёк"), ("amulet", "амулет"), ("crowbar", "лом"), ("herbs", "травы"),
                 ("flute", "флейта"), ("shield", "щит"), ("bread", "хлеб"), ("telescope", "подзорная труба")],
        "creature": [("wolf", "волк"), ("bandit", "бандит"), ("merchant", "торговец"), ("child", "ребёнок"), ("ghost", "призрак"), ("knight", "рыцарь"), ("crow", "ворона"), ("beggar", "нищий")],
        "action": [("light the lantern", "зажечь фонарь"), ("raise the alarm", "поднять тревогу"), ("hide in the shadows", "укрыться в тени"), ("greet the visitor", "поприветствовать гостя"),
                   ("close the shutters", "закрыть ставни"), ("offer a discount", "предложить скидку"), ("lower the drawbridge", "опустить мост"), ("sharpen the blade", "заточить клинок"),
                   ("sing a song", "спеть песню"), ("follow the player", "идти за игроком"), ("count the coins", "пересчитать монеты"), ("brew a potion", "сварить зелье"), ("lock the door", "запереть дверь"),
                   ("share a rumour", "рассказать слух"), ("check the traps", "проверить ловушки"), ("pray quietly", "тихо помолиться"), ("scold the child", "отчитать ребёнка"), ("feed the fire", "подбросить в костёр"),
                   ("send a messenger", "отправить гонца"), ("take a nap", "вздремнуть"), ("sound the horn", "протрубить в рог"), ("bar the gate", "забаррикадировать ворота"), ("tend the garden", "ухаживать за садом"),
                   ("wave at the player", "помахать игроку"), ("sweep the floor", "подмести пол"), ("give a warning", "дать предупреждение"), ("patch the roof", "залатать крышу"), ("ring the bell", "позвонить в колокол"),
                   ("draw a map", "нарисовать карту"), ("sit by the fire", "сесть у огня")],
    },
    "B": {
        "stat": [("shield charge", "заряд щита"), ("suspicion", "подозрение"), ("battery", "заряд батареи"), ("credit", "кредит"), ("gravity", "гравитация"), ("stealth", "скрытность"), ("hull", "прочность корпуса"),
                 ("curiosity", "любопытство"), ("signal", "уровень сигнала"), ("oxygen", "кислород"), ("heat", "перегрев"), ("loyalty", "лояльность")],
        "flag": [("the airlock is sealed", "шлюз закрыт"), ("the reactor is unstable", "реактор нестабилен"), ("a drone is overhead", "над головой дрон"), ("the lights are out", "свет выключен"),
                 ("the elevator is working", "лифт работает"), ("the terminal is unlocked", "терминал разблокирован"), ("the player is carrying cargo", "игрок несёт груз"), ("the station is on lockdown", "станция в карантине"),
                 ("the comms are jammed", "связь заглушена"), ("the cat is asleep", "кот спит"), ("the vault is open", "хранилище открыто"), ("a patrol is passing", "проходит патруль")],
        "item": [("keycard", "ключ-карта"), ("battery pack", "аккумулятор"), ("wrench", "гаечный ключ"), ("datapad", "планшет"), ("medkit", "аптечка"), ("hologram", "голограмма"), ("welding torch", "сварочная горелка"),
                 ("thermos", "термос"), ("signal flare", "сигнальная ракета"), ("toolbox", "ящик с инструментами")],
        "creature": [("drone", "дрон"), ("technician", "техник"), ("android", "андроид"), ("stowaway", "безбилетник"), ("inspector", "инспектор"), ("rat", "крыса"), ("pilot", "пилот")],
        "action": [("recalibrate the sensors", "перекалибровать датчики"), ("seal the bulkhead", "задраить переборку"), ("reroute the power", "перенаправить питание"), ("hail the inspector", "вызвать инспектора"),
                   ("hack the terminal", "взломать терминал"), ("vent the corridor", "продуть коридор"), ("dim the lights", "приглушить свет"), ("scan the cargo", "просканировать груз"), ("whistle a tune", "насвистывать мелодию"),
                   ("pour some tea", "налить чаю"), ("file a report", "составить рапорт"), ("reboot the console", "перезагрузить консоль"), ("hide the datapad", "спрятать планшет"), ("call the elevator", "вызвать лифт"),
                   ("calm the android", "успокоить андроида"), ("tighten the bolts", "подтянуть болты"), ("send a ping", "отправить пинг"), ("water the plants", "полить растения"), ("switch to manual", "перейти на ручное управление"),
                   ("pet the cat", "погладить кота"), ("log the visitor", "записать посетителя"), ("burn off the debris", "выжечь мусор"), ("warm up the engines", "прогреть двигатели"), ("tag the stowaway", "пометить безбилетника")],
    },
}
COMP = {"lt": ("is below {n}", "ниже {n}"), "gt": ("is above {n}", "выше {n}"), "ge": ("is at least {n}", "не меньше {n}"), "le": ("is at most {n}", "не больше {n}"),
        "eq": ("is exactly {n}", "ровно {n}"), "between": ("is between {n} and {m}", "между {n} и {m}")}


@dataclass
class Cond:
    kind: str                      # stat | flag | item | creature
    key: int
    op: str = ""
    n: int = 0
    m: int = 0
    neg: bool = False


class World2:
    def __init__(self, rng: random.Random, pool: str):
        P = POOLS[pool]
        self.stats = {i: rng.randint(0, 100) for i in rng.sample(range(len(P["stat"])), rng.randint(4, 7))}
        self.flags = {i: rng.random() < 0.5 for i in rng.sample(range(len(P["flag"])), rng.randint(3, 6))}
        self.items = set(rng.sample(range(len(P["item"])), rng.randint(0, 4)))
        self.creatures = {i: rng.randint(1, 9) for i in rng.sample(range(len(P["creature"])), rng.randint(0, 3))}

    def holds(self, c: Cond) -> bool:
        if c.kind == "stat":
            v = self.stats[c.key]
            ok = {"lt": v < c.n, "gt": v > c.n, "ge": v >= c.n, "le": v <= c.n, "eq": v == c.n, "between": c.n <= v <= c.m}[c.op]
        elif c.kind == "flag":
            ok = self.flags[c.key]
        elif c.kind == "item":
            ok = c.key in self.items
        else:
            ok = c.key in self.creatures and self.creatures[c.key] <= c.n
        return ok != c.neg


def _word(pair: Tuple[str, str], lang: str) -> str:
    return pair[1 if lang == "ru" else 0]


def describe_world(w: World2, pool: str, lang: str, rng: random.Random) -> str:
    P = POOLS[pool]
    parts = [f"{_word(P['stat'][i], lang)}: {v}" for i, v in w.stats.items()]
    parts += [(_word(P["flag"][i], lang) + (": да" if lang == "ru" else ": yes") if v else _word(P["flag"][i], lang) + (": нет" if lang == "ru" else ": no")) for i, v in w.flags.items()]
    parts.append(("В сумке: " if lang == "ru" else "Carrying: ") + (", ".join(_word(P["item"][i], lang) for i in sorted(w.items)) or ("ничего" if lang == "ru" else "nothing")))
    parts.append((", ".join(f"{_word(P['creature'][i], lang)} — {d} шаг." for i, d in w.creatures.items()) if lang == "ru" else ", ".join(f"{_word(P['creature'][i], lang)} {d} steps away" for i, d in w.creatures.items()))
                 or ("поблизости никого" if lang == "ru" else "nobody nearby"))
    rng.shuffle(parts)
    return "; ".join(parts)


def cond_text(c: Cond, pool: str, lang: str) -> str:
    P = POOLS[pool]
    if c.kind == "stat":
        en, ru = COMP[c.op]
        t = (ru if lang == "ru" else en).format(n=c.n, m=c.m)
        return f"{_word(P['stat'][c.key], lang)} {t}"
    if c.kind == "flag":
        name = _word(P["flag"][c.key], lang)
        return (f"неверно, что {name}" if lang == "ru" else f"it is not true that {name}") if c.neg else name
    if c.kind == "item":
        name = _word(P["item"][c.key], lang)
        return (f"у вас нет: {name}" if lang == "ru" else f"you do not carry the {name}") if c.neg else (f"у вас есть: {name}" if lang == "ru" else f"you carry the {name}")
    name = _word(P["creature"][c.key], lang)
    t = f"{name} ближе чем в {c.n + 1} шагах" if lang == "ru" else f"a {name} is within {c.n} steps"
    return (f"неверно, что {t}" if lang == "ru" else f"it is not true that {t}") if c.neg else t


def _random_cond(rng: random.Random, w: World2, pool: str) -> Cond:
    P = POOLS[pool]
    kinds = [("stat", 5), ("flag", 3), ("item", 2), ("creature", 2)]
    kind = rng.choices([k for k, _ in kinds], [x for _, x in kinds])[0]
    if kind == "stat":
        key = rng.choice(list(w.stats))
        op = rng.choice(list(COMP))
        n = rng.choice([10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90])
        m = n + rng.choice([10, 20, 30]) if op == "between" else 0
        return Cond("stat", key, op, n, min(m, 100) if op == "between" else 0, neg=False)
    if kind == "flag":
        return Cond("flag", rng.choice(list(w.flags)), neg=rng.random() < 0.3)
    if kind == "item":
        return Cond("item", rng.randrange(len(P["item"])), neg=rng.random() < 0.4)
    return Cond("creature", rng.randrange(len(P["creature"])), n=rng.choice([2, 3, 5, 7]), neg=rng.random() < 0.3)


def gen_policy2(rng: random.Random, pool: str = "A") -> Optional[List[Sample]]:
    lang = rng.choice(["en", "ru"])
    P = POOLS[pool]
    w = World2(rng, pool)
    n_rules = rng.randint(3, 7)
    actions = rng.sample(range(len(P["action"])), n_rules + 1)
    rules: List[Tuple[List[Cond], str, int]] = []                      # (conditions, "and"/"or", action)
    for a in actions[:-1]:
        k = rng.choice([1, 1, 1, 2])
        conds = [_random_cond(rng, w, pool) for _ in range(k)]
        rules.append((conds, rng.choice(["and", "or"]), a))
    default = actions[-1]

    def fires(r) -> bool:
        vals = [w.holds(c) for c in r[0]]
        return all(vals) if r[1] == "and" else any(vals)

    which = next((i for i, r in enumerate(rules) if fires(r)), None)
    act = rules[which][2] if which is not None else default
    join = {"and": (" и " if lang == "ru" else " and "), "or": (" или " if lang == "ru" else " or ")}
    lines = []
    for i, (conds, op, a) in enumerate(rules):
        text = join[op].join(cond_text(c, pool, lang) for c in conds)
        lines.append(f"{i + 1}) " + (f"Если {text} — {_word(P['action'][a], lang)}." if lang == "ru" else f"If {text}: {_word(P['action'][a], lang)}."))
    header = "Правила (применяйте первое подходящее по порядку):" if lang == "ru" else "Rules (apply the first one that matches, in order):"
    other = ("Иначе: " if lang == "ru" else "Otherwise: ") + _word(P["action"][default], lang) + "."
    state = f"{describe_world(w, pool, lang, rng)}\n{header}\n" + "\n".join(lines) + f"\n{other}"
    options = [_word(P["action"][a], lang) for _, _, a in rules] + [_word(P["action"][default], lang)]
    extras = [_word(P["action"][a], lang) for a in range(len(P["action"])) if a not in actions]
    rng.shuffle(extras)
    options = list(dict.fromkeys(options + extras[: rng.randint(0, 3)]))
    rng.shuffle(options)
    target = options.index(_word(P["action"][act], lang))
    q = rng.choice(["Какое действие должен выполнить персонаж?", "Что делает NPC сейчас?", "Выберите действие по правилам."] if lang == "ru"
                   else ["What should the NPC do now?", "Which action follows from the rules?", "Choose the NPC's next action."])
    task = "npc_policy2" if pool == "A" else "npc_policy2_unseen"
    out = [Sample(task, "npc", lang, "choice", state, q, options, target)]
    # does one condition hold right now?
    conds = rules[rng.randrange(len(rules))][0]
    c = conds[rng.randrange(len(conds))]
    qn = (f"Верно ли, что {cond_text(c, pool, lang)}?" if lang == "ru" else f"Is it true that {cond_text(c, pool, lang)}?")
    out.append(Sample(task + "_condition", "npc", lang, "noul", state, qn, None, float(w.holds(c))))
    return out


def gen_policy2_unseen(rng: random.Random) -> Optional[List[Sample]]:
    return gen_policy2(rng, "B")


GENERATORS: Dict[str, Tuple[Callable[[random.Random], Optional[List[Sample]]], float]] = {"policy2": (gen_policy2, 6.0)}
