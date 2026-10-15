import json
import random

from src.leeth2.data import specs
from src.leeth2.data.kinds import KINDS
from src.leeth2.data.webtools import _json_objects, _node_text, describe_candidate

HTML = '<html backend_node_id="1"><body backend_node_id="2"><div backend_node_id="3"><li backend_node_id="1250" class="item">Pickup <span>reservation</span></li><li backend_node_id="1251">Delivery</li></div></body></html>'


def cand(i, tag="li", **attrs):
    return json.dumps({"tag": tag, "backend_node_id": str(i), "attributes": json.dumps({"backend_node_id": str(i), **attrs})})


def test_node_text_is_nesting_aware():
    assert _node_text(HTML, "1250") == "Pickup reservation"
    assert _node_text(HTML, "1251") == "Delivery"
    assert _node_text(HTML, "999") == ""


def test_candidate_description_uses_text_and_key_attributes():
    d = describe_candidate(cand(1250, **{"class": "item big wide extra", "aria_label": "Pickup"}), HTML)
    assert d.startswith("<li>") and "text='Pickup reservation'" in d and "aria_label='Pickup'" in d
    assert "extra" not in d  # classes are cut to the first three


def test_mind2web_row_becomes_a_choice_with_one_correct_candidate():
    sp = specs.by_name("mind2web")
    row = {
        "confirmed_task": "Reserve a pickup order in Boston", "website": "exploretock", "domain": "Travel", "cleaned_html": HTML,
        "pos_candidates": [cand(1250, aria_label="Pickup")], "neg_candidates": [cand(1251 + i, id=f"x{i}", title=f"Other {i}") for i in range(20)],
        "action_reprs": ["[combobox] Reservation type -> SELECT: Pickup", "[svg] -> CLICK", "[li] Pickup -> CLICK"], "target_action_index": "2",
        "operation": json.dumps({"op": "CLICK", "value": ""}),
    }
    for seed in range(15):
        out = KINDS["mind2web"](sp, row, random.Random(seed), {})
        assert len(out) == 1
        s = out[0]
        s.validate()
        assert "aria_label='Pickup'" in s.options[s.target]
        assert 6 <= len(s.options) <= 30
        assert "Previous steps: [combobox] Reservation type -> SELECT: Pickup | [svg] -> CLICK" in s.state
    assert KINDS["mind2web"](sp, {**row, "neg_candidates": []}, random.Random(0), {}) == []


SYSTEM = 'SYSTEM: You are a helpful assistant with access to the following functions. Use them if required -\n{\n    "name": "get_exchange_rate",\n    "description": "Get the exchange rate between two currencies",\n    "parameters": {"type": "object", "properties": {"from": {"type": "string"}, "to": {"type": "string"}}}\n}\n\n{"name": "book_flight", "description": "Book a flight", "parameters": {"type": "object", "properties": {}}}'


def test_function_objects_are_found_with_nested_braces():
    objs = _json_objects(SYSTEM)
    assert [o["name"] for o in objs] == ["get_exchange_rate", "book_flight"]


def test_glaive_function_call_and_direct_answer():
    sp = specs.by_name("glaive_fc")
    call = SYSTEM and {"system": SYSTEM, "chat": "USER: What is the exchange rate from USD to EUR?\n\n\nASSISTANT: <functioncall> {\"name\": \"get_exchange_rate\", \"arguments\": '{\"from\": \"USD\", \"to\": \"EUR\"}'} <|endoftext|>\n\n\nFUNCTION RESPONSE: {\"rate\": 0.9}"}
    outs = []
    for seed in range(20):
        outs += KINDS["glaive"](sp, call, random.Random(seed), {})
    ch = [s for s in outs if s.ptype == "choice"]
    assert ch and all(s.options[s.target].startswith("get_exchange_rate") for s in ch)
    ex = [s for s in outs if s.ptype == "extract"]
    assert ex and all(s.state[s.target[0] : s.target[1]] in ("USD", "EUR") for s in ex)
    direct = {"system": SYSTEM, "chat": "USER: Tell me a joke.\n\n\nASSISTANT: Why did the chicken cross the road? <|endoftext|>"}
    s = KINDS["glaive"](sp, direct, random.Random(1), {})[0]
    assert s.options[s.target] in ("no function needed — answer directly", "none of the functions", "just reply")
    assert KINDS["glaive"](sp, {"system": "no tools here", "chat": "USER: hi\n\n\nASSISTANT: hello"}, random.Random(1), {}) == []
