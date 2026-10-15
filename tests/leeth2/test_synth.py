"""The synthetic domains are only useful if their labels are right. Each test recomputes the answer from the
TEXT of the sample, independently of the generator's internal state."""

import random
import re
from collections import deque

import pytest

from src.leeth2.data.synth import browser, games, npc, sorting, tools
from src.leeth2.schema import Sample

N = 250


def many(fn, n=N, seed=0, **kw):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        got = fn(rng, **kw)
        if got is None:
            continue
        got = got if isinstance(got, list) else [got]
        for s in got:
            s.validate()
        out.extend(got)
    return out


def support(s: Sample):
    t = s.target
    if isinstance(t, list):
        return {i for i, p in enumerate(t) if p > 0}
    return {t}


# ------------------------------------------------------------------------------------------------ games
NAME_TO_DIR = {}
for style in games.MOVE_NAMES.values():
    for d, names in style.items():
        for n in names:
            NAME_TO_DIR[n] = d


def bfs(start, goal, blocked, H, W):
    q, seen = deque([(start, 0)]), {start}
    while q:
        c, d = q.popleft()
        if c == goal:
            return d
        for dr, dc in games.DIRS.values():
            n = (c[0] + dr, c[1] + dc)
            if 0 <= n[0] < H and 0 <= n[1] < W and n not in seen and n not in blocked:
                seen.add(n)
                q.append((n, d + 1))
    return None


def test_maze_labels_match_a_bfs_on_the_printed_grid():
    ss = many(games.gen_maze)
    assert len(ss) > 100
    for s in ss:
        lines = s.state.split("\n")
        wall = re.search(r"(?:wall|стена)", lines[0])
        wall_c = re.search(r"(\S) (?:=|—) (?:wall|стена)", lines[0]).group(1)
        grid = lines[1:]
        H, W = len(grid), len(grid[0])
        pos = {ch: (r, c) for r, row in enumerate(grid) for c, ch in enumerate(row) if ch in "SE"}
        walls = {(r, c) for r, row in enumerate(grid) for c, ch in enumerate(row) if ch == wall_c}
        total = bfs(pos["S"], pos["E"], walls, H, W)
        good = set()
        for i, opt in enumerate(s.options):
            dr, dc = games.DIRS[NAME_TO_DIR[opt]]
            nxt = (pos["S"][0] + dr, pos["S"][1] + dc)
            if 0 <= nxt[0] < H and 0 <= nxt[1] < W and nxt not in walls and bfs(nxt, pos["E"], walls, H, W) == total - 1:
                good.add(i)
        assert good == support(s), s.state


def test_treasure_labels_match_a_bfs_on_the_printed_map():
    ss = many(games.gen_treasure)
    assert len(ss) > 100
    for s in ss:
        grid = s.state.split("\n")[1:]
        H, W = len(grid), len(grid[0])
        me = next((r, c) for r, row in enumerate(grid) for c, ch in enumerate(row) if ch == "P")
        chests = [(r, c) for r, row in enumerate(grid) for c, ch in enumerate(row) if ch == "T"]
        blocked = {(r, c) for r, row in enumerate(grid) for c, ch in enumerate(row) if ch in "^#"}
        dm = min(d for ch in chests for d in [bfs(me, ch, blocked, H, W)] if d is not None)
        good = set()
        for i, opt in enumerate(s.options):
            dr, dc = games.DIRS[NAME_TO_DIR[opt]]
            nxt = (me[0] + dr, me[1] + dc)
            if 0 <= nxt[0] < H and 0 <= nxt[1] < W and nxt not in blocked and any(bfs(nxt, ch, blocked, H, W) == dm - 1 for ch in chests):
                good.add(i)
        assert good == support(s)


def test_snake_never_picks_a_lethal_move_and_soft_labels_are_distributions():
    ss = many(games.gen_snake)
    assert len(ss) > 100
    soft = 0
    for s in ss:
        legend, *grid = s.state.split("\n")
        syms = re.findall(r"(\S) (?:=|—) (?:head|голова|food|еда|body|тело)", legend)
        head_c, food_c, body_c = syms[0], syms[1], syms[2]
        H, W = len(grid), len(grid[0])
        head = next((r, c) for r, row in enumerate(grid) for c, ch in enumerate(row) if ch == head_c)
        body = {(r, c) for r, row in enumerate(grid) for c, ch in enumerate(row) if ch == body_c}
        for i in support(s):
            dr, dc = games.DIRS[NAME_TO_DIR[s.options[i]]]
            nxt = (head[0] + dr, head[1] + dc)
            assert 0 <= nxt[0] < H and 0 <= nxt[1] < W and nxt not in body, "optimal move must not hit a wall or the body"
        if isinstance(s.target, list):
            soft += 1
            assert sum(s.target) == pytest.approx(1.0)
    assert soft > 10, "ties between equally good moves exist and must be labelled softly"


def test_tictactoe_best_moves_never_lose_value():
    ss = many(games.gen_ttt)
    assert len(ss) > 60
    for s in ss:
        me = re.search(r"(?:You play|играете за) ([XO])", s.state).group(1)
        rows = s.state.split("\n")[1:4]
        board = tuple(ch for r in rows for ch in r)
        best = set(games._ttt_best(board, me))
        shown = set()
        for i, o in enumerate(s.options):
            m = re.search(r"\((\d);?,?(\d)\)", o)
            if m:
                cell = (int(m.group(1)) - 1) * 3 + int(m.group(2)) - 1
            elif re.search(r"(\d)$", o):
                cell = int(re.search(r"(\d)$", o).group(1)) - 1
            else:
                r_names = ["top", "middle", "bottom", "верх", "середина", "низ"]
                c_names = ["left", "center", "right", "лево", "центр", "право"]
                a, b = o.split("-")
                cell = (r_names.index(a) % 3) * 3 + (c_names.index(b) % 3)
            assert board[cell] == ".", "only empty cells are offered"
            if cell in best:
                shown.add(i)
        assert shown == support(s)


# ------------------------------------------------------------------------------------------------ NPC
def parse_world(state):
    g = lambda pat: re.search(pat, state)
    return dict(
        health=int(g(r"(?:Health|Здоровье): (\d+)").group(1)), stamina=int(g(r"(?:Stamina|Выносливость): (\d+)").group(1)), hunger=int(g(r"(?:Hunger|Голод): (\d+)").group(1)),
        gold=int(g(r"(?:Gold|Золото): (\d+)").group(1)), time=g(r"(?:Time|Время): (\w+)").group(1), loc=g(r"(?:Location|Место): (\w+)").group(1),
        enemy=(int(g(r"(?:Enemy nearby: | в )(\d+)(?: tiles| клетках)").group(1)) if g(r"(?:Enemy nearby|Враг рядом)") else None),
        items=[w.strip() for w in re.search(r"(?:Inventory|Инвентарь): ([^;" + chr(10) + r"]*)", state).group(1).split(",") if w.strip() not in ("empty", "пусто")],
        relation=int(g(r"(?:Player relation|Отношение игрока): ([+-]?\d+)").group(1)),
    )


def eval_cond(cond_text, w):
    """Independent re-implementation of the condition phrases, working from the printed sentence."""
    t = cond_text
    n = re.search(r"(\d+)", t)
    n = int(n.group(1)) if n else None
    ru_item = {"potion": "зелье", "food": "еда", "sword": "меч"}
    if "health is below" in t or "здоровье ниже" in t:
        return w["health"] < n
    if "stamina is below" in t or "выносливость ниже" in t:
        return w["stamina"] < n
    if "hunger is above" in t or "голод выше" in t:
        return w["hunger"] > n
    if t in ("it is night", "сейчас ночь"):
        return w["time"] in ("night", "ночь")
    if t in ("it is daytime", "сейчас день"):
        return w["time"] in ("day", "день")
    if "an enemy is within" in t or "враг ближе чем" in t:
        return w["enemy"] is not None and w["enemy"] <= n
    if "no enemies nearby" in t or "врагов поблизости нет" in t:
        return w["enemy"] is None
    if "you have a potion" in t or "есть зелье" in t:
        return "potion" in w["items"] or "зелье" in w["items"]
    if "you have food" in t or "есть еда" in t:
        return "food" in w["items"] or "еда" in w["items"]
    if "carry a sword" in t or "есть меч" in t:
        return "sword" in w["items"] or "меч" in w["items"]
    if "at least" in t or "не меньше" in t:
        return w["gold"] >= n
    if "at the market" in t or "на рынке" in t:
        return w["loc"] in ("market", "рынок")
    if "at home" in t or "дома" in t:
        return w["loc"] in ("home", "дом")
    if "in the forest" in t or "в лесу" in t:
        return w["loc"] in ("forest", "лес")
    if "player is friendly" in t or "дружелюбно" in t:
        return w["relation"] >= 1
    if "player is hostile" in t or "враждебно" in t:
        return w["relation"] <= -1
    raise AssertionError("unknown condition: " + t)


def test_npc_policy_follows_the_rules_exactly():
    ss = [s for s in many(npc.gen_policy, n=300) if s.task == "npc_policy"]
    assert len(ss) > 150
    for s in ss:
        w = parse_world(s.state)
        body = s.state
        rules = re.findall(r"(?:\[(?:Priority|Приоритет) (\d+)\] |(\d+)\) )(?:If|Если) (.+?)(?::| —) (.+?)\.\n", body + "\n")
        assert rules
        explicit = "Priority" in body or "Приоритет" in body
        ordered = sorted(rules, key=lambda r: int(r[0] or r[1]))
        default = re.search(r"(?:Otherwise|Иначе): (.+?)\.", body).group(1)
        want = default
        for _, _, cond, act in ordered:
            if eval_cond(cond, w):
                want = act
                break
        assert s.options[s.target] == want, body


def test_npc_condition_noul_and_extract_labels():
    ss = many(npc.gen_policy, n=300)
    cond = [s for s in ss if s.task == "npc_condition"]
    assert cond
    for s in cond:
        w = parse_world(s.state)
        txt = re.search(r"[«“](.+?)[»”]", s.question).group(1)
        assert eval_cond(txt, w) == bool(s.target)
    for s in [x for x in ss if x.task == "npc_rule"]:
        span = s.state[s.target[0] : s.target[1]]
        assert re.match(r"\d\) (If|Если)", span)


def test_persona_reply_matches_persona_and_utterance():
    ss = [s for s in many(npc.gen_persona, n=300) if s.task == "npc_persona"]
    assert len(ss) > 100
    inv = {}
    for (u, p), (en, ru) in npc.REPLIES.items():
        inv[en] = (u, p, "en")
        inv[ru] = (u, p, "ru")
    utt = {en: u for u, (en, ru) in npc.UTTER.items()}
    utt.update({ru: u for u, (en, ru) in npc.UTTER.items()})
    pers = {}
    for k, (en, ru) in npc.PERSONAS.items():
        pers[en] = k
        pers[ru] = k
    for s in ss:
        u = utt[re.search(r"[«“](.+?)[»”]", s.state).group(1)]
        p = pers[re.search(r"(?:personality|характер): (\w+)", s.state).group(1)]
        assert inv[s.options[s.target]][:2] == (u, p)


# ------------------------------------------------------------------------------------------------ sorting
def test_file_sorting_follows_the_extension_rule():
    ss = many(sorting.gen_files)
    assert len(ss) > 150
    for s in ss:
        name = re.search(r"(?:File|Файл): (\S+)", s.state).group(1)
        ext = name.split(".", 1)[1] if name.endswith("tar.gz") is False else "tar.gz"
        ext = name.rsplit(".", 1)[-1] if ext != "tar.gz" else "tar.gz"
        group = next((g for g, (exts, en, ru) in sorting.EXT_GROUPS.items() if ext in exts), None)
        chosen = s.options[s.target]
        listed = [o.split(" (")[0] for o in s.options]
        gnames = {en: g for g, (e, en, ru) in sorting.EXT_GROUPS.items()}
        gnames.update({ru: g for g, (e, en, ru) in sorting.EXT_GROUPS.items()})
        listed_groups = {gnames[o] for o in listed if o in gnames}
        if group in listed_groups:
            assert chosen.split(" (")[0] in (sorting.EXT_GROUPS[group][1], sorting.EXT_GROUPS[group][2])
        else:
            assert chosen.split(" (")[0] in ("Other", "Разное")


def test_ticket_routing_picks_the_department_with_most_keywords():
    ss = many(sorting.gen_tickets)
    assert len(ss) > 100
    for s in ss:
        msg = re.search(r"[«“](.+?)[»”]", s.state).group(1).lower()
        best, best_n = None, 0
        for o in s.options[:-1]:
            name, kws = o.split(": ", 1)
            n = sum(1 for kw in kws.split(", ") if re.search(r"(?<!\w)" + re.escape(kw) + r"(?!\w)", msg))
            if n > best_n:
                best, best_n = o, n
        want = best if best_n else s.options[-1]
        assert s.options[s.target] == want, (msg, s.options)


def test_ordering_questions():
    ss = [s for s in many(sorting.gen_order, n=300) if s.task == "sort_order"]
    assert len(ss) > 150
    for s in ss:
        rows = re.findall(r"- (.+?): \S+ \$?([\d.]+)", s.state)
        vals = {n: float(v) for n, v in rows}
        q = s.question
        asc = sorted(vals, key=vals.get)
        if re.search(r"cheapest|lightest|lowest rated|дешёвый|лёгкий|низким", q):
            want = asc[0]
        elif re.search(r"second|вторым", q):
            want = asc[1]
        else:
            want = asc[-1]
        assert s.options[s.target] == want, (s.state, q)


def test_sortedness_noul():
    for s in [x for x in many(sorting.gen_order, n=200) if x.task == "sort_is_sorted"]:
        vals = [float(v) for _, v in re.findall(r"- (.+?): \S+ \$?([\d.]+)", s.state)]
        assert bool(s.target) == (vals == sorted(vals))


def test_compare_numbers():
    for s in many(sorting.gen_compare_numbers):
        a, b = s.options
        key = (lambda x: x) if "-" in a and a.count("-") == 2 else float
        want = max([a, b], key=lambda x: key(x))
        assert s.options[s.target] == want


# ------------------------------------------------------------------------------------------------ browser
def test_browser_target_is_the_unique_element_matching_the_goal():
    ss = [s for s in many(browser.gen_click, n=400) if s.task == "browser_element"]
    assert len(ss) > 250
    for s in ss:
        goal = s.state.split("\n")[1]
        picked = s.options[s.target]
        quoted = re.findall(r"[“«](.+?)[”»]", goal)
        typing = goal.startswith(("Goal: Type", "Цель: Введите"))
        if typing:  # the value is quoted, the field name is not (EN) / quoted second (RU)
            m = re.search(r"into the (.+?) field", goal)
            lab = m.group(1) if m else quoted[-1]
        elif quoted:
            lab = quoted[0]
        else:
            lab = None  # icon goals are semantic ("Open the search."): checked through uniqueness below
        if lab:
            assert lab in picked, (goal, picked)
        same = [o for o in s.options if re.sub(r"\d+", "#", o) == re.sub(r"\d+", "#", picked)]
        assert len(same) == 1, "the goal must identify exactly one element"
        assert len(set(s.options)) == len(s.options)


def test_browser_typing_goal_extracts_the_exact_value():
    ss = [s for s in many(browser.gen_click, n=400) if s.task == "browser_type_value"]
    assert ss
    for s in ss:
        v = s.state[s.target[0] : s.target[1]]
        assert f"“{v}”" in s.state or f"«{v}»" in s.state


def test_browser_risk_labels():
    for s in many(browser.gen_risk):
        label = re.search(r"'(.+?)'|«(.+?)»", s.state)
        lab = label.group(1) or label.group(2)
        assert (lab in browser.DANGEROUS_EN + browser.DANGEROUS_RU) == bool(s.target)


# ------------------------------------------------------------------------------------------------ tools
def match_tools(request, lang):
    """Every tool whose request template could have produced this text (templates overlap: 'Включи {song}' vs 'Включи свет ...')."""
    hits = []
    for i, t in enumerate(tools.TOOLS):
        for tpl in (t.tpl_ru if lang == "ru" else t.tpl_en):
            pat = re.escape(tpl)
            pat = re.sub(r"\\\{\w+\\\}", r"(.+?)", pat)
            if re.fullmatch(pat, request):
                hits.append(i)
                break
    return hits


def test_tool_pick_target_is_the_tool_whose_template_produced_the_request():
    ss = [s for s in many(tools.gen_toolpick, n=500) if s.task == "tool_pick"]
    assert len(ss) > 300
    none_hits = 0
    for s in ss:
        lang = s.lang
        request = re.search(r"[«“](.+)[»”]", s.state).group(1)
        idx = match_tools(request, lang)
        chosen = s.options[s.target]
        if not idx:  # chit-chat
            assert chosen in tools.NONE_OPT[lang]
            none_hits += 1
            continue
        mentions = [o for o in s.options[:-1] if any(tools.TOOLS[i].name in o or (tools.TOOLS[i].desc_ru if lang == "ru" else tools.TOOLS[i].desc_en) in o for i in idx)]
        if chosen in tools.NONE_OPT[lang]:
            assert not mentions, "'none' is only correct when no matching tool is in the list"
        else:
            assert chosen in mentions
    assert none_hits > 10


def test_tool_arguments_are_exact_spans_and_holdout_tools_stay_out_of_training():
    ss = many(tools.gen_toolpick, n=400)
    for s in [x for x in ss if x.task == "tool_arg"]:
        v = s.state[s.target[0] : s.target[1]]
        assert v and v in s.state
    train_names = " ".join(" ".join(s.options) for s in ss if s.task == "tool_pick")
    for i, t in enumerate(tools.TOOLS):
        if tools.is_holdout(i):
            assert t.name not in train_names, f"{t.name} is reserved for evaluation"
    un = many(tools.gen_toolpick, n=200, holdout=True)
    assert any(tools.is_holdout(tools.TOOLS.index(next(t for t in tools.TOOLS if t.name in " ".join(s.options)))) for s in un if s.task == "tool_pick")
