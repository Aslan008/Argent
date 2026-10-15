"""NPC rules with an open vocabulary: labels follow from the structure that wrote the text, and the held-out pool shares no wording."""

import random

from src.leeth2.data.synth import generate, npc2
from src.leeth2.data.synth.npc2 import POOLS, gen_policy2, gen_policy2_unseen


def test_pools_share_no_wording():
    for kind in POOLS["A"]:
        a = {w for pair in POOLS["A"][kind] for w in pair}
        b = {w for pair in POOLS["B"][kind] for w in pair}
        assert not (a & b), (kind, a & b)


def test_samples_are_valid_and_the_label_is_the_first_matching_rule():
    seen_default = seen_rule = 0
    for seed in range(300):
        rng = random.Random(seed)
        lang = None
        out = gen_policy2(rng)
        s = out[0]
        s.validate()
        assert s.ptype == "choice" and len(set(s.options)) == len(s.options) and 3 <= len(s.options) <= 11
        assert out[1].ptype == "noul" and 0.0 <= out[1].target <= 1.0
        seen_default += s.options[s.target] == s.state.rsplit("\n", 1)[1].split(": ", 1)[-1].rstrip(".")
        seen_rule += 1
    assert seen_default > 5 and seen_default < seen_rule - 20          # both outcomes occur: a rule fires, or the default applies


def test_an_independent_reading_of_the_text_gives_the_same_label():
    """Parse the rules back out of the English sentences and re-evaluate them against the facts printed in the state."""
    import re

    ok = checked = 0
    for seed in range(400):
        rng = random.Random(seed)
        smp = gen_policy2(rng)[0]
        if smp.lang != "en":
            continue
        lines = smp.state.split("\n")
        facts = lines[0]
        rules = [l for l in lines if re.match(r"^\d\) If ", l)]
        stats = {m.group(1): int(m.group(2)) for m in re.finditer(r"(?:^|; )([a-z ]+): (\d+)", facts)}
        flags = {m.group(1): m.group(2) == "yes" for m in re.finditer(r"(?:^|; )([a-z ]+): (yes|no)", facts)}
        carrying = facts.split("Carrying: ")[1].split(";")[0]
        carried = set() if carrying.strip() == "nothing" else {x.strip() for x in carrying.split(",")}
        near = {m.group(1): int(m.group(2)) for m in re.finditer(r"(?:^|; |, )([a-z]+) (\d+) steps away", facts)}

        def holds(text: str) -> bool:
            neg = text.startswith("it is not true that ")
            text = text.replace("it is not true that ", "")
            m = re.match(r"^([a-z ]+) (is below|is above|is at least|is at most|is exactly|is between) (\d+)(?: and (\d+))?$", text)
            if m and m.group(1) in stats:
                v, n = stats[m.group(1)], int(m.group(3))
                r = {"is below": v < n, "is above": v > n, "is at least": v >= n, "is at most": v <= n, "is exactly": v == n, "is between": n <= v <= int(m.group(4) or 0)}[m.group(2)]
            elif text in flags:
                r = flags[text]
            elif text.startswith("you carry the "):
                r = text[len("you carry the "):] in carried
            elif text.startswith("you do not carry the "):
                r = text[len("you do not carry the "):] not in carried
            else:
                m = re.match(r"^a ([a-z]+) is within (\d+) steps$", text)
                r = bool(m) and m.group(1) in near and near[m.group(1)] <= int(m.group(2))
            return r != neg

        picked = None
        for l in rules:
            body, action = l.split(") If ", 1)[1].rsplit(": ", 1)
            action = action.rstrip(".")
            protected = re.sub(r"between (\d+) and (\d+)", r"between  to ", body)         # "between 20 and 40" is one comparison, not a conjunction
            conds = [c.replace(" to ", " and ") if "between" in c else c for c in re.split(r" and | or ", protected)]
            vals = [holds(c) for c in conds]
            fire = all(vals) if " and " in protected else any(vals)
            if fire:
                picked = action
                break
        if picked is None:
            picked = lines[-1].split(": ", 1)[1].rstrip(".")
        checked += 1
        ok += picked == smp.options[smp.target]
    assert checked > 100 and ok / checked > 0.97, (ok, checked)


def test_unseen_generator_uses_pool_b_and_its_own_task_names():
    out = generate(gen_policy2_unseen, 50, 3)
    assert out and all(s.task.startswith("npc_policy2_unseen") for s in out)
    assert any("airlock" in s.state or "шлюз" in s.state or "battery" in s.state or "батаре" in s.state or "oxygen" in s.state or "кислород" in s.state for s in out)


def test_registered_for_training_but_the_unseen_pool_only_for_test(tmp_path):
    from src.leeth2.data import synth as SY

    assert any(m == "npc2" for m, *_ in SY.PLAN)
    res = SY.build(tmp_path, scale=0.002)
    names = [p.name for p in res["train"]]
    assert any("npc2_policy2" in n for n in names) and not any("unseen" in n for n in names)
    assert any("npc2_unseen" in p.name for p in res["test"])
