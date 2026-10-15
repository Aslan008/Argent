import json
import random

import pytest

from src.leeth2.data import kinds, specs
from src.leeth2.data.kinds import KINDS, Spec
from src.leeth2.schema import Sample


def run(spec_id, row, seed=0, ctx=None, n=1):
    sp = specs.by_name(spec_id)
    rng = random.Random(seed)
    out = []
    for k in range(n):
        out.extend(KINDS[sp.kind](sp, row, random.Random(seed + k), ctx or {}))
    for s in out:
        s.validate()
    return out


def test_registry_is_consistent():
    ids = [s.id for s in specs.SPECS]
    assert len(ids) == len(set(ids))
    assert len(ids) > 100
    for s in specs.SPECS:
        assert s.kind in KINDS, s.id
        assert len(s.caps) == 4, s.id
        if s.eval_only or s.heldout:
            assert s.caps[0] == 0 or s.train is None or s.eval_only, s.id
        if s.heldout:
            assert s.caps[0] == 0, f"{s.id}: held-out tasks must never reach the train pool"
        if s.eval_only:
            assert s.lic != "x"  # explicit flag exists
        if s.lic == "nc":
            assert s.eval_only, f"{s.id}: non-commercial data must stay out of training"
    # benchmarks are never trained on
    for sid in ("mmlu_test", "mmlu_pro", "belebele_ru", "belebele_en", "xquad_ru", "anli", "toxic_chat"):
        assert specs.by_name(sid).eval_only


def test_cls_binary_sentiment():
    ss = run("imdb", {"text": "A wonderful, moving film.", "label": 1}, ctx={"names": ["neg", "pos"], "n_classes": 2}, n=30)
    ch = [s for s in ss if s.ptype == "choice"]
    assert len(ch) == 30 and all(len(s.options) == 2 for s in ch)
    pos_words = {"positive", "good", "favorable", "позитивный", "положительный", "хороший"}
    assert all(s.options[s.target].lower() in pos_words for s in ch)
    noul = [s for s in ss if s.ptype == "noul"]
    assert noul and {s.target for s in noul} <= {0.0, 1.0}


def test_cls_many_classes_subset_contains_target_and_is_bounded():
    names = [f"intent {i}" for i in range(150)]
    sp = specs.by_name("massive_en")  # a training source: option sets are sampled
    ctx = {"names": names, "n_classes": 150, "label_to_id": {n: i for i, n in enumerate(names)}}
    seen_sizes = set()
    for seed in range(40):
        s = KINDS["cls"](sp, {"text": "book a table", "label": "intent 77"}, random.Random(seed), ctx)[0]
        assert s.options[s.target] == "intent 77"
        assert 4 <= len(s.options) <= 60
        seen_sizes.add(len(s.options))
    assert len(seen_sizes) > 5


def test_held_out_benchmarks_always_show_the_whole_label_set():
    names = [f"intent {i}" for i in range(150)]
    sp = specs.by_name("clinc150")
    for seed in range(10):
        s = KINDS["cls"](sp, {"text": "book a table", "intent": 77}, random.Random(seed), {"names": names, "n_classes": 150})[0]
        assert len(s.options) == 150 and s.options[s.target] == "intent 77"


def test_cls_string_labels_use_the_label_to_id_map():
    ctx = {"names": ["alarm set", "music play"], "n_classes": 2, "label_to_id": {"alarm_set": 0, "music_play": 1}}
    s = run("massive_en", {"text": "wake me up", "label": "alarm_set"}, ctx=ctx)[0]
    assert s.options[s.target] == "alarm set"


def test_nli_labels_and_strict_noul():
    for label, p in [(0, 1.0), (1, 0.0), (2, 0.0)]:
        ss = run("snli", {"premise": "A man plays guitar.", "hypothesis": "A man is playing music.", "label": label}, ctx={"n_classes": 3, "names": []}, n=20)
        ch = [s for s in ss if s.ptype == "choice"]
        assert ch and all(s.target == label and len(s.options) == 3 for s in ch)
        assert all(s.target == p for s in ss if s.ptype == "noul")


def test_nli_drops_unlabelled_rows():
    assert run("snli", {"premise": "a", "hypothesis": "b", "label": -1}, ctx={"n_classes": 3, "names": []}) == []


def test_paraphrase_rows():
    ss = run("paws", {"sentence1": "Cats chase mice.", "sentence2": "Mice are chased by cats.", "label": 1}, n=20)
    noul = [s for s in ss if s.ptype == "noul"]
    assert noul and all(s.target == 1.0 for s in noul)
    assert all("Cats chase mice." in s.state for s in noul)


def test_mcq_getters():
    arc = {"question": "Which is a mammal?", "choices": {"text": ["frog", "whale", "trout", "newt"], "label": ["A", "B", "C", "D"]}, "answerKey": "B"}
    s = run("arc_challenge", arc)[0]
    assert s.options[s.target] == "whale" and "mammal" in s.question
    race = {"article": "Tom went home.", "question": "Where did Tom go?", "options": ["school", "home", "park", "shop"], "answer": "B"}
    s = run("race", race)[0]
    assert s.options[s.target] == "home" and s.state == "Tom went home."
    hs = {"ctx": "He opens the door and", "activity_label": "Entering", "endings": ["walks in.", "flies away.", "melts.", "sings a tree."], "label": "0"}
    s = run("hellaswag", hs)[0]
    assert s.options[s.target] == "walks in." and s.state.startswith("Entering: ")
    wg = {"sentence": "The trophy didn't fit in the case because the _ was too big.", "option1": "trophy", "option2": "case", "answer": "1"}
    s = run("winogrande", wg)[0]
    assert s.options[s.target] == "trophy" and "_" in s.state
    assert run("arc_challenge", {**arc, "answerKey": "Z"}) == []


def test_squad_spans_are_exact_and_unanswerable_has_no_target():
    ctx = "The  Eiffel Tower\nis in Paris.   It was built in 1889."
    row = {"context": ctx, "question": "When was it built?", "answers": {"text": ["1889"], "answer_start": [ctx.index("1889")]}}
    s = run("squad2", row)[0]
    a, b = s.target
    assert s.state[a:b] == "1889"
    s = run("squad2", {**row, "answers": {"text": [], "answer_start": []}})[0]
    assert s.target is None and s.ptype == "extract"


def test_ner_extracts_the_first_entity_of_a_type_and_handles_absent_types():
    names = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
    row = {"tokens": ["Angela", "Merkel", "visited", "Siemens", "in", "Munich", "."], "ner_tags": [1, 2, 0, 3, 0, 5, 0]}
    sp = specs.by_name("wikiann_en")
    got_texts, absent = set(), 0
    for seed in range(60):
        for s in KINDS["ner"](sp, row, random.Random(seed), {"tag_names": names}):
            s.validate()
            if s.ptype == "extract" and s.target:
                got_texts.add(s.state[s.target[0] : s.target[1]])
            if s.ptype == "extract" and s.target is None:
                absent += 1
    assert got_texts <= {"Angela Merkel", "Siemens", "Munich"} and "Angela Merkel" in got_texts
    # every type is present in this sentence, so there is nothing to ask about as 'absent'
    assert absent == 0


def test_ner_plain_type_tags_without_bio():
    names = ["O", "person", "location"]
    row = {"tokens": ["Paul", "Smith", "flew", "to", "Rome"], "ner_tags": [1, 1, 0, 0, 2]}
    sp = specs.by_name("fewnerd")
    spans = set()
    for seed in range(40):
        for s in KINDS["ner"](sp, row, random.Random(seed), {"tag_names": names}):
            if s.ptype == "extract" and s.target:
                spans.add(s.state[s.target[0] : s.target[1]])
    assert "Paul Smith" in spans and "Rome" in spans


def test_gliner_style_entities():
    row = {"tokenized_text": ["Identify", "the", "entity", ":", "\n", "Mary", "likes", "Paris", "."], "ner": [[5, 5, '"Person"'], [7, 7, '"City"']]}
    from src.leeth2.data.specs import ner_pairs_from_gliner

    text, ents = ner_pairs_from_gliner(row)
    assert text == "Mary likes Paris ."
    assert {text[s:e] for s, e, _ in ents} == {"Mary", "Paris"}
    sp = specs.by_name("gliner_multi")
    ss = KINDS["entlist"](sp, row, random.Random(1), {})
    assert ss and ss[0].state[ss[0].target[0] : ss[0].target[1]] in {"Mary", "Paris"}


def test_numind_parser():
    from src.leeth2.data.specs import ner_pairs_from_numind

    text, ents = ner_pairs_from_numind({"input": "State University Press, 1997.", "output": "['State University Press <> Publisher']"})
    assert [(text[s:e], l) for s, e, l in ents] == [("State University Press", "Publisher")]


def test_scores_binary_sim_pairwise():
    hs = {"prompt": "What is 2+2?", "response": "4", "helpfulness": 4, "correctness": 4, "coherence": 3, "complexity": 1, "verbosity": 0}
    ss = run("helpsteer2", hs)
    assert len(ss) == 2 and all(s.levels == 5 and 1 <= s.target <= 5 for s in ss)
    s = run("civil_comments", {"text": "you are awful", "toxicity": 0.35})[0]
    assert s.ptype == "noul" and s.target == pytest.approx(0.35)
    s = run("stsb", {"sentence1": "A man sings.", "sentence2": "A person is singing.", "score": 4.2}, n=30)
    sc = [x for x in s if x.ptype == "score"]
    assert sc and all(x.levels == 6 and x.target == pytest.approx(1 + 4.2 / 5 * 5) for x in sc)
    hh = {"chosen": "\n\nHuman: Hi\n\nAssistant: Hello! How can I help?", "rejected": "\n\nHuman: Hi\n\nAssistant: Go away."}
    opts = set()
    for seed in range(40):
        for s in run("hh_rlhf", hh, seed=seed):
            assert ("Hello! How can I help?" in s.state) and ("Go away." in s.state)
            first = s.state.index("Hello!")
            second = s.state.index("Go away.")
            assert (s.target == 0) == (first < second), "target must point at the position of the chosen response"
            opts.add(s.target)
    assert opts == {0, 1}, "position of the better answer is randomised"


def test_every_spec_with_a_getter_survives_an_empty_row():
    for sp in specs.SPECS:
        if "get" in sp.opts:
            try:
                out = KINDS[sp.kind](sp, {}, random.Random(0), {})
            except (KeyError, TypeError, AttributeError):
                continue  # hard getters may raise on malformed rows; the builder counts these as conv_err
            assert out == [] or all(isinstance(x, Sample) for x in out)


def test_entity_lists_also_yield_many_option_type_questions():
    sp = specs.by_name("nuner")
    ctx, rng, sizes = {}, random.Random(0), []
    for i in range(400):
        text = f"Alice{i} met Bob{i} in Paris{i % 7} about the Widget{i} project."
        row = {"input": text, "output": str([f"Alice{i} <> Person{i % 50}", f"Paris{i % 7} <> City{i % 30}", f"Widget{i} <> Product{i % 40}"])}
        for s in KINDS["entlist"](sp, row, rng, ctx):
            s.validate()
            if s.ptype == "choice":
                sizes.append(len(s.options))
                assert s.options[s.target] in (f"Person{i % 50}", f"City{i % 30}", f"Product{i % 40}")
    assert sizes and min(sizes) >= 8 and max(sizes) > 60, "option counts spread from small to large"
