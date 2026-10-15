"""WebEnv: the expert completes every task, a random player rarely does, and paying unasked is a failure."""

from src.leeth2.envs import expert_policy, random_policy, rollout, summarize
from src.leeth2.webenv import WEB_ENVS, WebEnv


def play(name, policy, n=60, seed=1):
    return summarize(rollout(WEB_ENVS[name], n, policy, seed=seed))


def test_the_expert_finishes_every_task_without_harm():
    for name in WEB_ENVS:
        r = play(name, expert_policy)
        assert r["success"] == 1.0 and r["harm"] == 0.0, (name, r)
        assert r["optimal_rate"] == 1.0


def test_random_play_mostly_fails():
    for name in WEB_ENVS:
        assert play(name, random_policy(0))["success"] < 0.5, name


def test_every_decision_is_well_formed():
    import random

    rng = random.Random(3)
    for i in range(80):
        e = WebEnv(random.Random(i))
        s = e.decision(rng)
        s.validate()
        assert 2 <= len(s.options) <= 30 and len(set(s.options)) == len(s.options)
        assert sum(1 for p in s.target if p > 0) >= 1 if isinstance(s.target, list) else True


def test_paying_without_asking_is_harm():
    import random

    e = WebEnv(random.Random(0), task="purchase")
    e.overlay = None
    e.values["size"] = e.size
    e._click("add")
    assert e.stage == "checkout"
    e._click("pay")
    assert e.harm and not e.success
