"""The honest arena's rules: shared items, each model's own rendering, dev-only selection, counted errors, the pre-written verdict."""
import random

import numpy as np

from src.leeth2 import fairarena as fa

CLINC = fa.TASKS["clinc150"]
NAMES = ["oos", "card_declined", "pay_bill", "alarm_set", "timer", "weather"]


def test_clinc_drops_out_of_scope_and_keeps_the_gold():
    labels = fa.label_set(CLINC, [], NAMES)
    assert "oos" not in labels and len(labels) == 5
    assert fa.make_item(CLINC, "test", 3, {"text": "hi", "intent": 0}, labels, NAMES) is None      # an out-of-scope row is out of the task
    it = fa.make_item(CLINC, "test", 4, {"text": "my card was declined", "intent": 1}, labels, NAMES)
    assert it["options"][it["gold"]] == "card declined" and sorted(it["options"]) == sorted(fa.humanize(x) for x in labels)


def test_items_are_deterministic_and_shuffled_per_item():
    labels = fa.label_set(CLINC, [], NAMES)
    a = fa.make_item(CLINC, "test", 4, {"text": "x", "intent": 2}, labels, NAMES)
    b = fa.make_item(CLINC, "test", 4, {"text": "x", "intent": 2}, labels, NAMES)
    orders = {tuple(fa.make_item(CLINC, "test", i, {"text": "x", "intent": 2}, labels, NAMES)["options"]) for i in range(30)}
    assert a == b and len(orders) > 5


def test_ordinal_and_nli_keep_their_order():
    sst = fa.TASKS["sst5"]
    it = fa.make_item(sst, "test", 1, {"text": "fine", "label_text": "neutral"}, fa.label_set(sst, [], None), None)
    assert it["options"] == list(fa.SST5_LEVELS) and it["gold"] == 2
    x = fa.TASKS["xnli_en"]
    it = fa.make_item(x, "test", 1, {"premise": "a", "hypothesis": "b", "label": 2}, fa.label_set(x, [], None), None)
    assert it["options"] == list(fa.NLI_LABELS) and it["gold"] == 2


def test_laya_twenty_option_protocol():
    m = fa.TASKS["massive_en"]
    labels = [f"intent_{i}" for i in range(60)]
    it = fa.make_item(m, "repro", 7, {"text": "wake me up", "label": "intent_5"}, labels, None, k=20, seed=13)
    assert len(it["options"]) == 20 and it["options"][it["gold"]] == "intent 5"
    assert it == fa.make_item(m, "repro", 7, {"text": "wake me up", "label": "intent_5"}, labels, None, k=20, seed=13)


def test_each_model_reads_its_own_rendering_of_the_same_item():
    labels = fa.label_set(CLINC, [], NAMES)
    it = fa.make_item(CLINC, "test", 4, {"text": "my card was declined", "intent": 1}, labels, NAMES)
    state, q = fa.leeth_question(CLINC, it)
    assert state == it["text"] and q == "Which user intent fits best?"                     # English, never the Russian half
    lstate, qs = fa.laya_request(CLINC, it)
    assert lstate == {"message": it["text"]} and list(qs["q"]["criteria"]) == it["options"]
    x = fa.TASKS["xnli_en"]
    nli = fa.make_item(x, "test", 1, {"premise": "p", "hypothesis": "h", "label": 0}, list(fa.NLI_LABELS), None)
    s, qs = fa.laya_request(x, nli)
    assert s == {"premise": "p", "hypothesis": "h"} and all(qs["q"]["criteria"].values())
    assert fa.leeth_question(x, nli) == ("p", fa.LEETH_NLI_Q.format(h="h"))


def test_laya_answers_map_onto_the_item_and_errors_are_counted():
    it = {"id": "t:1", "options": ["a", "b", "c", "d"], "gold": 2}
    tour = {"type": "choice", "probabilities": {"c": 0.7, "a": 0.3}}                          # a tournament: finalists only
    assert fa.laya_probs(it, tour) == [0.3, 0.0, 0.7, 0.0]
    score = {"type": "score", "probabilities": {"0": 0.1, "1": 0.1, "2": 0.6, "3": 0.2}}
    assert fa.outcome(it, fa.laya_probs(it, score))["ok"] == 1.0
    err = fa.outcome(it, None, error="ValueError: too many options")
    assert err["ok"] == 0.0 and err["error"] and err["probs"] is None


def test_selection_uses_dev_and_prefers_the_simpler_config_on_a_tie():
    assert fa.select_config({"single": 0.6, "tournament": 0.7}, ["single", "wide", "tournament"]) == "tournament"
    assert fa.select_config({"single": 0.7, "tournament": 0.7}, ["single", "wide", "tournament"]) == "single"


def test_temperature_refit_repairs_overconfidence():
    rng = np.random.default_rng(0)

    def draw(n):
        out = []
        for _ in range(n):
            g = int(rng.integers(0, 4))
            ok = rng.random() < 0.6
            top = g if ok else (g + 1) % 4
            p = np.full(4, 0.01)
            p[top] = 0.97
            out.append((list(p / p.sum()), g))
        return out

    dev, test = draw(400), draw(400)
    t = fa.fit_temperature(dev)
    assert t > 1.5                                                                              # over-confident -> soften
    rows = lambda d, pre: [{"id": f"{pre}{i}", "probs": p, "conf": max(p), "ok": float(int(np.argmax(p)) == g)} for i, (p, g) in enumerate(d)]  # noqa: E731
    gold = {**{f"d{i}": g for i, (_, g) in enumerate(dev)}, **{f"t{i}": g for i, (_, g) in enumerate(test)}}
    te = rows(test, "t")
    _, after = fa.refit_ece(rows(dev, "d"), te, gold)
    assert after < fa.ece([r["conf"] for r in te], [r["ok"] for r in te]) - 0.2


def test_bootstrap_and_the_written_verdict():
    rng = random.Random(1)
    tie = {t: [float(rng.random() < 0.6) for _ in range(500)] for t in "abc"}
    other = {t: [float(rng.random() < 0.6) for _ in range(500)] for t in "abc"}
    assert fa.verdict(fa.paired_bootstrap(tie, other, n_boot=2000)) == "на уровне"
    better = {t: [1.0] * 400 + [0.0] * 100 for t in "abc"}
    worse = {t: [1.0] * 250 + [0.0] * 250 for t in "abc"}
    assert fa.verdict(fa.paired_bootstrap(better, worse, n_boot=2000)) == "выше"
    assert fa.verdict(fa.paired_bootstrap(worse, better, n_boot=2000)) == "ниже"
    mixed_a = {"a": [1.0] * 500, "b": [1.0] * 500, "c": [0.0] * 200 + [1.0] * 300}
    mixed_b = {"a": [0.0] * 300 + [1.0] * 200, "b": [0.0] * 300 + [1.0] * 200, "c": [1.0] * 500}
    assert fa.verdict(fa.paired_bootstrap(mixed_a, mixed_b, n_boot=2000)) == "на уровне"           # ahead on average, behind on one task


def test_reproduction_tolerance():
    assert fa.repro_ok(0.80, 0.783, 300) and not fa.repro_ok(0.59, 0.783, 300)


def test_contamination_check():
    corpus = fa.shingles("we were told that the hotel shuttle leaves every hour from the main gate near the lobby")
    assert fa.contaminated("The hotel shuttle leaves every hour from the main gate near the lobby", corpus)
    assert not fa.contaminated("how do I reset my online banking password today please", corpus)
