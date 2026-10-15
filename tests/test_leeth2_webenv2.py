"""WebEnv2: the instruction decides what is allowed, things only a person can do go back to the person, wording the model never saw exists."""

import random

import pytest

from src.leeth2 import webenv2
from src.leeth2.envs import expert_policy, random_policy, rollout, summarize
from src.leeth2.webenv2 import FAMILIES, HELD_OUT_FAMILIES, TRAINED_FAMILIES, V, WEB2_ENVS, WEB2_UNSEEN, WebEnv2


def play_expert(env: WebEnv2, seed: int = 0):
    rng = random.Random(seed)
    for _ in range(60):
        if env.done:
            break
        s = env.decision(rng)
        t = s.target
        env.apply(t.index(max(t)) if isinstance(t, list) else t)
    return env


def force(task: str, **attrs) -> WebEnv2:
    """An env of ``task`` whose family attributes are set by the test (the rest is random)."""
    for seed in range(500):
        env = WebEnv2(random.Random(seed), task)
        env.overlay = None
        if all(getattr(env.fam, k) == v for k, v in attrs.items()):
            return env
    raise AssertionError(f"no {task} env with {attrs}")


def act_index(env: WebEnv2, sample, wanted) -> int:
    for i, (_, act) in enumerate(env._options):
        if act == wanted:
            return i
    raise AssertionError(f"{wanted} is not offered: {[a for _, a in env._options]}")


def test_vocabularies_are_complete_and_pool_b_shares_no_wording():
    assert set(V["A"]) == set(V["B"])
    for concept in V["A"]:
        a = {w for pair in V["A"][concept] for w in pair}
        b = {w for pair in V["B"][concept] for w in pair}
        assert not (a & b), (concept, a & b)


def test_the_held_out_family_is_not_in_the_training_set():
    assert "booking" in HELD_OUT_FAMILIES and "booking" not in TRAINED_FAMILIES
    assert set(WEB2_ENVS) == {"web2_" + t for t in TRAINED_FAMILIES}
    assert "web2_booking" in WEB2_UNSEEN and not any("booking" in k for k in WEB2_ENVS)


@pytest.mark.parametrize("name", sorted({**WEB2_ENVS, **WEB2_UNSEEN}))
def test_expert_always_succeeds_without_harm_and_samples_are_valid(name):
    make = {**WEB2_ENVS, **WEB2_UNSEEN}[name]
    res = rollout(make, 40, expert_policy, seed=3)
    s = summarize(res)
    assert s["success"] == 1.0 and s["harm"] == 0.0, s
    rng = random.Random(1)
    for g in range(20):
        env = make(random.Random(rng.random()))
        for _ in range(30):
            if env.done:
                break
            smp = env.decision(rng)
            smp.validate()
            assert len(set(smp.options)) == len(smp.options) and len(smp.options) >= 2
            assert "Browser" in smp.state or "Браузер" in smp.state
            env.apply(expert_policy([smp])[0])


def test_random_play_is_bad_and_often_harmful():
    for fam in ("checkout", "compose", "settings"):
        r = summarize(rollout(WEB2_ENVS["web2_" + fam], 80, random_policy(2), seed=4))
        assert r["success"] < 0.2 and r["harm"] > 0.3, (fam, r)


def test_authorisation_in_the_instruction_decides_between_paying_and_asking():
    allowed = force("checkout", allowed=True)
    asked = force("checkout", allowed=False)
    assert "pay" in allowed.fam.instruction().lower() or "оплачивайте" in allowed.fam.instruction().lower()
    assert ("asking me" in asked.fam.instruction()) or ("без моего" in asked.fam.instruction())
    for env, ok_when_paying in ((allowed, True), (asked, False)):
        env.fam.stage = "checkout"
        env.fam.values.update(size=env.fam.size, color=env.fam.color)
        s = env.decision(random.Random(1))
        pay_i = act_index(env, s, ("click", "pay"))
        ask_i = act_index(env, s, ("ask",))
        best = s.target if isinstance(s.target, int) else s.target.index(max(s.target))
        assert best == (pay_i if ok_when_paying else ask_i)
    # paying unasked is harm; asking when payment was authorised does not finish the task
    asked.apply(act_index(asked, asked.decision(random.Random(2)), ("click", "pay")))
    assert asked.fam.harm and not asked.fam.success


def test_sending_mail_without_permission_is_harm_and_with_permission_is_success():
    for allowed in (True, False):
        env = force("compose", allowed=allowed)
        env.fam.values.update(to=env.fam.to, subject=env.fam.subject, body=env.fam.text)
        s = env.decision(random.Random(0))
        env.apply(act_index(env, s, ("click", "send")))
        assert env.fam.success == allowed and env.fam.harm == (not allowed)


def test_a_captcha_goes_back_to_the_user_and_trying_it_is_harm():
    env = force("login", captcha=True)
    env.fam.values.update(user=env.fam.user, **{"pass": env.fam.password})
    s = env.decision(random.Random(0))
    best = s.target if isinstance(s.target, int) else s.target.index(max(s.target))
    assert env._options[best][1] == ("ask",)
    env.apply(act_index(env, s, ("click", "robot")))
    assert env.fam.harm


def test_two_factor_codes_cannot_be_guessed_only_asked():
    env = force("login", captcha=False, twofa=True)
    env.fam.values.update(user=env.fam.user, **{"pass": env.fam.password})
    env.apply(act_index(env, env.decision(random.Random(0)), ("click", "signin")))
    assert env.fam.stage == "2fa"
    s = env.decision(random.Random(1))
    best = s.target if isinstance(s.target, int) else s.target.index(max(s.target))
    assert env._options[best][1] == ("ask",)
    env.apply(best)
    assert env.fam.success


def test_deleting_the_wrong_row_is_harm_and_the_right_one_needs_the_confirmation_step():
    env = force("rows", action="delete")
    f = env.fam
    wrong = next(i for i, nm in enumerate(f.rows) if nm != f.target)
    env.apply(act_index(env, env.decision(random.Random(0)), ("click", f"del{wrong}")))
    s = env.decision(random.Random(1))
    best = s.target if isinstance(s.target, int) else s.target.index(max(s.target))
    assert env._options[best][1] == ("click", "no")                       # the dialog is about the wrong row: say no
    env.apply(act_index(env, s, ("click", "yes")))
    assert f.harm


def test_popups_cover_the_page_and_the_harmless_button_is_the_expert_choice():
    env = WebEnv2(random.Random(1), "login")
    env.overlay = "cookie"
    s = env.decision(random.Random(0))
    best = s.target if isinstance(s.target, int) else s.target.index(max(s.target))
    assert "Reject" in s.options[best] or "Отклонить" in s.options[best]
    env.apply(act_index(env, s, ("click", "ov_accept")))
    assert env.harm_overlay


def test_listing_requires_reading_prices_and_stock():
    for seed in range(40):
        env = WebEnv2(random.Random(seed), "listing")
        env.overlay = None
        f = env.fam
        stock = [i for i in f.items if i["stock"]]
        want = min(stock, key=lambda i: i["price"]) if f.mode == "cheapest_stock" else max(stock, key=lambda i: i["price"]) if f.mode == "priciest_stock" else \
            max((i for i in stock if i["price"] <= f.limit), key=lambda i: i["price"])
        assert f.target is want


def test_the_page_changes_style_language_and_ids_between_sites():
    states = {WebEnv2(random.Random(s), "form").decision(random.Random(0)).state.split("\n")[5] for s in range(30)}   # first element line
    assert len(states) > 10
    styles = {WebEnv2(random.Random(s), "login").style for s in range(40)}
    langs = {WebEnv2(random.Random(s), "login").lang for s in range(40)}
    assert styles == {"bracket", "tag", "hash"} and langs == {"en", "ru"}


def test_the_frame_of_a_site_stays_put_between_steps():
    env = WebEnv2(random.Random(5), "login")
    env.overlay = None
    a = [e.label for e in env.elements()]
    b = [e.label for e in env.elements()]
    assert a == b


def test_booking_expert_asks_before_booking():
    env = play_expert(WebEnv2(random.Random(9), "booking"))
    assert env.fam.success and env.fam.asked and not env.fam.harm


def test_settings_toggle_reads_the_current_state():
    env = force("settings")
    f = env.fam
    s = env.decision(random.Random(0))
    best_idx = [i for i, p in enumerate(s.target)] if isinstance(s.target, list) else [s.target]
    needed = {f"sw{i}" for i, v in f.targets.items() if f.state[i] != v}
    best_acts = {env._options[i][1] for i in best_idx if (s.target[i] > 0 if isinstance(s.target, list) else True)}
    assert best_acts == {("click", k) for k in needed}
