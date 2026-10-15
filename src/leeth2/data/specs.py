"""The dataset registry. Schemas were checked against the live Hub from a Kaggle kernel (datasets 4.x:
no loading scripts, so script-only repos are either re-pointed to parquet mirrors or left out).

Licence classes (``lic``): permissive = MIT/Apache/CC-BY/CC0/ODC-BY; sa = CC-BY-SA (share-alike);
unknown = no explicit licence on the Hub (standard academic benchmarks); nc = non-commercial -> eval only.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any, Dict, List, Optional, Tuple

from .common import clean, humanize
from .kinds import Spec

SPECS: List[Spec] = []


def add(**kw: Any) -> Spec:
    s = Spec(**kw)
    SPECS.append(s)
    return s


# ---------------------------------------------------------------------------------------------- helpers
def _letters(ans: str, n: int) -> Optional[int]:
    ans = str(ans).strip()
    if ans[:1].upper() in "ABCDEFGHIJ" and len(ans) == 1:
        i = "ABCDEFGHIJ".index(ans.upper())
        return i if i < n else None
    if ans.isdigit():
        i = int(ans)
        return i if 0 <= i < n else None
    return None


def _choices_labels(row: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    c = row.get("choices") or {}
    return list(c.get("text", [])), list(c.get("label", []))


def mcq_arc(row):  # ARC / QASC / CommonsenseQA / OpenBookQA share one layout
    texts, labels = _choices_labels(row)
    key = row.get("answerKey")
    if key in labels:
        return None, clean(row.get("question") or row.get("question_stem")), texts, labels.index(key)
    return None


def mcq_obqa(row):
    texts, labels = _choices_labels(row)
    key = row.get("answerKey")
    if key in labels:
        return None, clean(row["question_stem"]), texts, labels.index(key)
    return None


def mcq_race(row):
    opts = list(row.get("options") or [])
    a = _letters(row.get("answer", ""), len(opts))
    return (clean(row["article"], 5000), clean(row["question"]), opts, a) if a is not None else None


def mcq_mmlu(row):
    opts = list(row.get("choices") or [])
    a = row.get("answer")
    if isinstance(a, str):
        a = _letters(a, len(opts))
    subj = humanize(row.get("subject") or "").replace("-", " ")
    q = clean(row.get("question"))
    return (None, q, opts, a) if a is not None and 0 <= a < len(opts) else None


def mcq_hellaswag(row):
    ends = list(row.get("endings") or [])
    lab = row.get("label")
    if lab in (None, "") or not ends:
        return None
    ctx = clean(row.get("ctx") or (row.get("ctx_a", "") + " " + row.get("ctx_b", "")))
    act = clean(row.get("activity_label"))
    state = (act + ": " if act else "") + ctx
    return state, "", ends, int(lab), "ending"


def mcq_winogrande(row):
    ans = str(row.get("answer", "")).strip()
    if ans not in ("1", "2"):
        return None
    return clean(row["sentence"]), "", [clean(row["option1"]), clean(row["option2"])], int(ans) - 1, "blank"


def mcq_xstory(row):
    a = row.get("answer_right_ending")
    if a not in (1, 2):
        return None
    ctx = " ".join(clean(row.get(f"input_sentence_{i}")) for i in range(1, 5))
    return ctx, "", [clean(row["sentence_quiz1"]), clean(row["sentence_quiz2"])], int(a) - 1, "ending"


def mcq_belebele(row):
    opts = [clean(row.get(f"mc_answer{i}")) for i in range(1, 5)]
    a = str(row.get("correct_answer_num", "")).strip()
    return (clean(row["flores_passage"], 5000), clean(row["question"]), opts, int(a) - 1) if a.isdigit() else None


def mcq_mmlupro(row):
    opts = list(row.get("options") or [])
    a = row.get("answer_index")
    return (None, clean(row["question"]), opts, int(a)) if a is not None and 0 <= int(a) < len(opts) else None


def span_squad(row):
    ans = row.get("answers") or {}
    texts, starts = list(ans.get("text", [])), list(ans.get("answer_start", []))
    return row.get("context"), row.get("question"), [(texts[0], starts[0])] if texts and starts else []


def span_mrqa(row):
    if row.get("subset") == "SQuAD":
        return None  # already covered by squad_v2
    da = row.get("detected_answers") or {}
    texts = list(da.get("text", []))
    spans = list(da.get("char_spans", []))
    if texts and spans and spans[0].get("start"):
        return row.get("context"), row.get("question"), [(texts[0], spans[0]["start"][0])]
    return None


def span_ropes(row):
    ans = (row.get("answers") or {}).get("text", [])
    ctx = clean((row.get("background") or "") + "\n\n" + (row.get("situation") or ""), 8000)
    return (ctx, row.get("question"), [(ans[0], None)]) if ans else None


def ner_pairs_from_numind(row):
    """NuNER: output = "['span <> label', ...]"."""
    text = row.get("input") or ""
    try:
        items = ast.literal_eval(row.get("output") or "[]")
    except Exception:
        return None
    ents = []
    for it in items:
        if " <> " not in it:
            continue
        span, lab = it.rsplit(" <> ", 1)
        i = text.find(span)
        if i >= 0 and span.strip():
            ents.append((i, i + len(span), lab))
    return text, ents


def ner_pairs_from_gliner(row):
    toks = row.get("tokenized_text") or []
    ner = row.get("ner") or []
    # the multi-task set prepends an instruction ("Identify the following entity classes ... \n") — keep only the real text
    if "\n" in toks and toks[:1] == ["Identify"]:
        cut = len(toks) - 1 - toks[::-1].index("\n")
        shift = cut + 1
        toks = toks[shift:]
        ner = [[int(s) - shift, int(e) - shift, l] for s, e, l in ner if int(s) - shift >= 0]
    offs, pos = [], 0
    for i, t in enumerate(toks):
        if i:
            pos += 1
        offs.append((pos, pos + len(t)))
        pos += len(t)
    text = " ".join(toks)
    ents = []
    for s, e, lab in ner:
        s, e = int(s), int(e)
        if 0 <= s <= e < len(offs):
            ents.append((offs[s][0], offs[e][1], str(lab).strip('"')))
    return text, ents


# ---------------------------------------------------------------------------------------------- classification
EN_NEG_POS = [["negative", "bad", "unfavorable"], ["positive", "good", "favorable"]]
RU_NEG_POS = [["негативный", "отрицательный", "плохой"], ["позитивный", "положительный", "хороший"]]
SENT_WHAT = dict(what_en=["sentiment", "sentiment of the text", "polarity"], what_ru=["тональность", "тональность текста"])

add(id="imdb", repo="stanfordnlp/imdb", kind="cls", family="sentiment", lang="en", dev="test", caps=(30000, 150, 300, 1000), lic="unknown",
    opts=dict(text="text", names=["neg", "pos"], names_en=EN_NEG_POS, names_ru=RU_NEG_POS, **SENT_WHAT, noul_frac=0.2))
add(id="sst2", repo="stanfordnlp/sst2", kind="cls", family="sentiment", lang="en", dev="validation", caps=(40000, 150, 300, 800), lic="unknown",
    opts=dict(text="sentence", names=["negative", "positive"], names_en=EN_NEG_POS, names_ru=RU_NEG_POS, **SENT_WHAT))
add(id="amazon_polarity", repo="fancyzhx/amazon_polarity", kind="cls", family="sentiment", lang="en", dev="test", caps=(40000, 100, 200, 600), lic="permissive",
    opts=dict(text=["title", "content"], names=["negative", "positive"], names_en=EN_NEG_POS, names_ru=RU_NEG_POS, **SENT_WHAT))
add(id="yelp_polarity", repo="fancyzhx/yelp_polarity", kind="cls", family="sentiment", lang="en", dev="test", caps=(30000, 100, 200, 600), lic="unknown",
    opts=dict(text="text", names=["negative", "positive"], names_en=EN_NEG_POS, names_ru=RU_NEG_POS, **SENT_WHAT))
add(id="sentiment3", repo="Sp1786/multiclass-sentiment-analysis-dataset", kind="cls", family="sentiment", lang="en", dev="validation", test="test", caps=(25000, 100, 200, 600), lic="permissive",
    opts=dict(text="text", names=["negative", "neutral", "positive"], names_en=[["negative", "bad"], ["neutral", "mixed or neutral"], ["positive", "good"]],
              names_ru=[["негативный", "плохой"], ["нейтральный", "смешанный"], ["позитивный", "хороший"]], **SENT_WHAT))
for sub, names, what in [
    ("sentiment", ["negative", "neutral", "positive"], "sentiment"),
    ("hate", ["not hateful", "hateful"], "hate-speech label"),
    ("offensive", ["not offensive", "offensive"], "offensiveness label"),
    ("irony", ["not ironic", "ironic"], "irony label"),
]:
    add(id=f"tweet_{sub}", repo="cardiffnlp/tweet_eval", cfg=sub, kind="cls", family="sentiment" if sub == "sentiment" else "safety", lang="en", dev="validation", test="test",
        caps=(20000, 100, 200, 500), lic="unknown", opts=dict(text="text", names=names, what_en=[what], what_ru=[{"sentiment": "тональность", "hate-speech label": "язык вражды", "offensiveness label": "оскорбительность", "irony label": "ирония"}[what]]))
add(id="ag_news", repo="fancyzhx/ag_news", kind="cls", family="topic", lang="en", dev="test", caps=(40000, 150, 300, 1000), lic="unknown",
    opts=dict(text="text", names=["World", "Sports", "Business", "Sci/Tech"],
              names_en=[["world news", "World"], ["sports", "Sports"], ["business", "Business"], ["science and technology", "Sci/Tech"]],
              names_ru=[["мировые новости", "мир"], ["спорт"], ["бизнес"], ["наука и технологии"]], what_en=["topic", "news category"], what_ru=["тема", "категория новости"]))
add(id="dbpedia14", repo="fancyzhx/dbpedia_14", kind="cls", family="topic", lang="en", dev="test", caps=(0, 150, 300, 1000), lic="sa", heldout=True,
    opts=dict(text=["title", "content"], names=["Company", "EducationalInstitution", "Artist", "Athlete", "OfficeHolder", "MeanOfTransportation", "Building", "NaturalPlace", "Village", "Animal", "Plant", "Album", "Film", "WrittenWork"],
              what_en=["category of the entity"], what_ru=["категория объекта"]))
add(id="banking77", repo="PolyAI/banking77", revision="refs/convert/parquet", kind="cls", family="intent", lang="en", train=None, dev=None, test="test", caps=(0, 0, 0, 1500), lic="permissive",
    eval_only=True, heldout=True, opts=dict(text="text", names=None, what_en=["banking intent"], what_ru=["банковский запрос"], k_max=77, full_prob=1.0), note="77 classes, never trained on")
add(id="banking77_hf", repo="mteb/banking77", kind="cls", family="intent", lang="en", train=None, dev=None, test="test", caps=(0, 0, 0, 1500), lic="permissive",
    eval_only=True, heldout=True, opts=dict(text="text", label="label_text", names=None, string_labels=True, what_en=["banking intent"], what_ru=["банковский запрос"], k_max=77, full_prob=1.0),
    note="the Hub parquet mirror of Banking77 (the original script repo cannot be loaded): 77 classes, never trained on; Laya reports 0.425 here")
add(id="sst5", repo="SetFit/sst5", kind="cls", family="sentiment", lang="en", train=None, dev=None, test="test", caps=(0, 0, 0, 1200), lic="unknown",
    eval_only=True, heldout=True, opts=dict(text="text", label="label_text", names=None, string_labels=True, what_en=["sentiment (five grades)"], what_ru=["тональность (пять градаций)"], k_max=5, full_prob=1.0),
    note="five-grade sentiment: Laya's weakest primitive (0.372)")
add(id="emotion", repo="dair-ai/emotion", kind="cls", family="emotion", lang="en", dev="validation", test="test", caps=(0, 150, 300, 1000), lic="unknown", heldout=True,
    opts=dict(text="text", names=["sadness", "joy", "love", "anger", "fear", "surprise"], what_en=["emotion"], what_ru=["эмоция"]))
add(id="clinc150", repo="clinc/clinc_oos", cfg="plus", kind="cls", family="intent", lang="en", dev="validation", test="test", caps=(0, 300, 300, 1500), lic="permissive", heldout=True,
    opts=dict(text="text", label="intent", names=None, what_en=["user intent"], what_ru=["намерение пользователя"], k_max=60))
for lg, code in [("en", "en"), ("ru", "ru"), ("de", "de"), ("fr", "fr"), ("es", "es"), ("pt", "pt"), ("it", "it"), ("pl", "pl"), ("tr", "tr"), ("ja", "ja"), ("zh-CN", "zh"), ("ko", "ko"), ("nl", "nl"), ("sv", "sv")]:
    add(id=f"massive_{code}", repo="mteb/MassiveIntentClassification", cfg=lg, kind="cls", family="intent", lang=code, dev="validation", test="test",
        caps=(12000 if code in ("en", "ru") else 4000, 80, 150, 300), lic="permissive",
        opts=dict(text="text", names=None, string_labels=True, what_en=["user intent", "intent of the request"], what_ru=["намерение пользователя", "цель запроса"], k_max=60))
add(id="langid", repo="papluca/language-identification", kind="cls", family="langid", lang="mix", dev="validation", test="test", caps=(30000, 150, 300, 1000), lic="unknown",
    opts=dict(text="text", label="labels", string_labels=True, label_names_map={
        "ar": "Arabic", "bg": "Bulgarian", "de": "German", "el": "Greek", "en": "English", "es": "Spanish", "fr": "French", "hi": "Hindi", "it": "Italian", "ja": "Japanese",
        "nl": "Dutch", "pl": "Polish", "pt": "Portuguese", "ru": "Russian", "sw": "Swahili", "th": "Thai", "tr": "Turkish", "ur": "Urdu", "vi": "Vietnamese", "zh": "Chinese"},
        names_ru_map={"ar": "арабский", "bg": "болгарский", "de": "немецкий", "el": "греческий", "en": "английский", "es": "испанский", "fr": "французский", "hi": "хинди", "it": "итальянский", "ja": "японский",
                      "nl": "нидерландский", "pl": "польский", "pt": "португальский", "ru": "русский", "sw": "суахили", "th": "тайский", "tr": "турецкий", "ur": "урду", "vi": "вьетнамский", "zh": "китайский"},
        what_en=["language"], what_ru=["язык"], q_en=["What language is this text written in?", "Identify the language of the text.", "Which language is this?"],
        q_ru=["На каком языке написан текст?", "Определите язык текста.", "Какой это язык?"], q_prob=1.0, noul_frac=0.1))
add(id="yahoo_topics", repo="community-datasets/yahoo_answers_topics", kind="cls", family="topic", lang="en", dev="test", caps=(30000, 100, 200, 600), lic="unknown",
    opts=dict(text=lambda r: clean((r.get("question_title") or "") + "\n" + (r.get("question_content") or "")), label="topic",
              names=["Society & Culture", "Science & Mathematics", "Health", "Education & Reference", "Computers & Internet", "Sports", "Business & Finance", "Entertainment & Music", "Family & Relationships", "Politics & Government"],
              what_en=["topic of the question", "forum category"], what_ru=["тема вопроса", "категория форума"]))
add(id="newsgroups20", repo="SetFit/20_newsgroups", kind="cls", family="topic", lang="en", dev="test", caps=(11000, 100, 200, 500), lic="unknown",
    opts=dict(text="text", label="label_text", string_labels=True, what_en=["newsgroup", "discussion topic"], what_ru=["тематическая группа", "тема обсуждения"], k_max=20))
add(id="sms_spam", repo="ucirvine/sms_spam", kind="cls", family="spam", lang="en", dev=None, caps=(4000, 60, 100, 300), lic="unknown",
    opts=dict(text="sms", names=["ham", "spam"], names_en=[["not spam", "legitimate message"], ["spam", "unwanted advertising"]], names_ru=[["не спам", "обычное сообщение"], ["спам", "нежелательная реклама"]],
              what_en=["message type"], what_ru=["тип сообщения"]), note="train only: carved by hash")
add(id="enron_spam", repo="SetFit/enron_spam", kind="cls", family="spam", lang="en", dev="test", caps=(20000, 80, 150, 400), lic="unknown",
    opts=dict(text=lambda r: clean((r.get("subject") or "") + "\n" + (r.get("message") or r.get("text") or "")), names=["ham", "spam"],
              names_en=[["not spam", "legitimate email"], ["spam", "unwanted email"]], names_ru=[["не спам", "обычное письмо"], ["спам", "нежелательное письмо"]], what_en=["email type"], what_ru=["тип письма"]))
# --- Russian
add(id="ru_reviews", repo="ai-forever/ru-reviews-classification", kind="cls", family="sentiment", lang="ru", dev="validation", test="test", caps=(30000, 100, 200, 500), lic="permissive",
    opts=dict(text="text", label="label_text", string_labels=True, names_ru=None, **SENT_WHAT))
add(id="ru_kinopoisk", repo="ai-forever/kinopoisk-sentiment-classification", kind="cls", family="sentiment", lang="ru", dev="validation", test="test", caps=(25000, 100, 200, 500), lic="permissive",
    opts=dict(text="text", label="label_text", string_labels=True, **SENT_WHAT))
add(id="ru_headlines", repo="ai-forever/headline-classification", kind="cls", family="topic", lang="ru", dev="validation", test="test", caps=(15000, 100, 200, 500), lic="permissive",
    opts=dict(text="text", label="label_text", string_labels=True, what_en=["topic of the headline"], what_ru=["тема заголовка", "рубрика"]))

# ---------------------------------------------------------------------------------------------- NLI / pairs
add(id="mnli", repo="nyu-mll/multi_nli", kind="nli", family="nli", lang="en", dev="validation_matched", test="validation_mismatched", caps=(90000, 200, 300, 1500), lic="sa", opts={})
add(id="snli", repo="stanfordnlp/snli", kind="nli", family="nli", lang="en", dev="validation", test="test", caps=(50000, 150, 300, 1000), lic="sa", opts={})
add(id="xnli_ru", repo="facebook/xnli", cfg="ru", kind="nli", family="nli", lang="ru", dev="validation", test="test", caps=(60000, 150, 300, 1500), lic="unknown", opts={})
for lg in ["de", "fr", "es", "zh", "ar", "tr", "vi", "th", "hi", "bg", "el", "sw", "ur"]:
    add(id=f"xnli_{lg}", repo="facebook/xnli", cfg=lg, kind="nli", family="nli", lang=lg, dev="validation", test="test", caps=(5000, 50, 100, 500), lic="unknown", opts={})
add(id="xnli_en", repo="facebook/xnli", cfg="en", kind="nli", family="nli", lang="en", train=None, dev=None, test="test", caps=(0, 0, 0, 800), lic="unknown", eval_only=True, opts={},
    note="English XNLI test: the number Laya publishes (0.860 / 0.843)")
add(id="nli_rus", repo="cointegrated/nli-rus-translated-v2021", kind="nli", family="nli", lang="ru", dev="dev", test="test", caps=(40000, 150, 300, 800), lic="unknown",
    opts=dict(premise="premise_ru", hypothesis="hypothesis_ru", label="label", string_labels=True, label_order=["entailment", "neutral", "contradiction"]))
add(id="anli", repo="facebook/anli", kind="nli", family="nli", lang="en", train=None, dev="dev_r3", test="test_r3", caps=(0, 100, 200, 800), lic="nc", eval_only=True, opts={}, note="NC licence: evaluation only")
add(id="vitaminc", repo="tals/vitaminc", kind="nli", family="nli", lang="en", dev="validation", test="test", caps=(40000, 100, 200, 600), lic="sa",
    opts=dict(premise="evidence", hypothesis="claim", label="label", label_fn=lambda v, r: {"SUPPORTS": 0, "NOT ENOUGH INFO": 1, "REFUTES": 2}.get(v)))
add(id="fever_nli", repo="pietrolesci/nli_fever", kind="nli", family="nli", lang="en", dev="dev", test="test", caps=(30000, 100, 200, 600), lic="unknown",
    opts=dict(premise="hypothesis", hypothesis="premise", label="fever_gold_label", label_fn=lambda v, r: {"SUPPORTS": 0, "NOT ENOUGH INFO": 1, "REFUTES": 2}.get(v)))
add(id="rte", repo="nyu-mll/glue", cfg="rte", kind="nli", family="nli", lang="en", dev="validation", caps=(2490, 60, 100, 100), lic="unknown",
    opts=dict(premise="sentence1", hypothesis="sentence2", label_fn=lambda v, r: {0: 0, 1: 1}.get(v), choice=False))
add(id="paws", repo="google-research-datasets/paws", cfg="labeled_final", kind="para", family="paraphrase", lang="en", dev="validation", test="test", caps=(25000, 100, 200, 500), lic="unknown", opts={})
for lg in ["en", "de", "fr", "es", "ja", "ko", "zh"]:
    add(id=f"pawsx_{lg}", repo="google-research-datasets/paws-x", cfg=lg, kind="para", family="paraphrase", lang=lg, dev="validation", test="test", caps=(6000, 60, 100, 300), lic="unknown", opts={})
add(id="mrpc", repo="nyu-mll/glue", cfg="mrpc", kind="para", family="paraphrase", lang="en", dev="validation", caps=(3668, 60, 100, 100), lic="unknown", opts={})
add(id="qqp", repo="nyu-mll/glue", cfg="qqp", kind="para", family="paraphrase", lang="en", dev="validation", caps=(40000, 100, 200, 500), lic="unknown",
    opts=dict(a="question1", b="question2"))


def _cola(row):
    return [(clean(row["sentence"]), "Is this sentence grammatically acceptable?", "Грамматически ли допустимо это предложение?", float(row["label"]))]


add(id="cola", repo="nyu-mll/glue", cfg="cola", kind="binary", family="grammar", lang="en", dev="validation", caps=(8500, 80, 150, 300), lic="unknown", opts=dict(get=_cola))


def _boolq(row):
    return [(clean(row["passage"], 4000), clean(row["question"]) + "?", clean(row["question"]) + "?", float(bool(row["answer"])))]


add(id="boolq", repo="google/boolq", kind="binary", family="qa_bool", lang="en", dev="validation", caps=(9400, 100, 200, 600), lic="sa", opts=dict(get=_boolq))

# ---------------------------------------------------------------------------------------------- multiple choice
add(id="arc_challenge", repo="allenai/ai2_arc", cfg="ARC-Challenge", kind="mcq", family="mcq", lang="en", dev="validation", test="test", caps=(1119, 80, 100, 400), lic="sa", opts=dict(get=mcq_arc))
add(id="arc_easy", repo="allenai/ai2_arc", cfg="ARC-Easy", kind="mcq", family="mcq", lang="en", dev="validation", test="test", caps=(2251, 80, 100, 400), lic="sa", opts=dict(get=mcq_arc))
add(id="openbookqa", repo="allenai/openbookqa", cfg="main", kind="mcq", family="mcq", lang="en", dev="validation", test="test", caps=(4957, 80, 100, 400), lic="unknown", opts=dict(get=mcq_obqa))
add(id="qasc", repo="allenai/qasc", kind="mcq", family="mcq", lang="en", dev="validation", caps=(8134, 80, 150, 150), lic="permissive", opts=dict(get=mcq_arc))
add(id="commonsenseqa", repo="tau/commonsense_qa", kind="mcq", family="mcq", lang="en", dev="validation", caps=(9741, 100, 200, 400), lic="permissive", opts=dict(get=mcq_arc))
add(id="race", repo="ehovy/race", cfg="all", kind="mcq", family="mcq_reading", lang="en", dev="validation", test="test", caps=(70000, 150, 300, 800), lic="unknown", opts=dict(get=mcq_race))
add(id="mmlu_aux", repo="cais/mmlu", cfg="all", train="auxiliary_train", kind="mcq", family="mcq", lang="en", dev=None, caps=(70000, 0, 0, 0), lic="permissive", opts=dict(get=mcq_mmlu),
    note="train only (never the test split)")
add(id="hellaswag", repo="Rowan/hellaswag", kind="mcq", family="mcq_commonsense", lang="en", dev="validation", caps=(39000, 150, 300, 600), lic="unknown", opts=dict(get=mcq_hellaswag))
add(id="winogrande", repo="allenai/winogrande", cfg="winogrande_xl", kind="mcq", family="mcq_commonsense", lang="en", dev="validation", caps=(40000, 150, 300, 600), lic="unknown", opts=dict(get=mcq_winogrande))
# --- evaluation only (benchmarks, never trained on)
add(id="mmlu_test", repo="cais/mmlu", cfg="all", train=None, dev="validation", test="test", kind="mcq", family="bench_knowledge", lang="en", caps=(0, 0, 0, 2500), lic="permissive", eval_only=True, opts=dict(get=mcq_mmlu), note="benchmark: test split only")
add(id="mmlu_pro", repo="TIGER-Lab/MMLU-Pro", train=None, dev="validation", test="test", kind="mcq", family="bench_knowledge", lang="en", caps=(0, 0, 0, 1500), lic="permissive", eval_only=True, opts=dict(get=mcq_mmlupro))
add(id="belebele_ru", repo="facebook/belebele", cfg="rus_Cyrl", train=None, dev=None, test="test", kind="mcq", family="bench_reading", lang="ru", caps=(0, 0, 0, 900), lic="sa", eval_only=True, opts=dict(get=mcq_belebele))
add(id="belebele_en", repo="facebook/belebele", cfg="eng_Latn", train=None, dev=None, test="test", kind="mcq", family="bench_reading", lang="en", caps=(0, 0, 0, 900), lic="sa", eval_only=True, opts=dict(get=mcq_belebele))
add(id="xstory_ru", repo="juletxara/xstory_cloze", cfg="ru", train=None, dev=None, test="eval", kind="mcq", family="bench_commonsense", lang="ru", caps=(0, 0, 0, 1500), lic="sa", eval_only=True, heldout=True, opts=dict(get=mcq_xstory))
add(id="xstory_en", repo="juletxara/xstory_cloze", cfg="en", train=None, dev=None, test="eval", kind="mcq", family="bench_commonsense", lang="en", caps=(0, 0, 0, 1500), lic="sa", eval_only=True, heldout=True, opts=dict(get=mcq_xstory))
add(id="xwinograd_ru", repo="Muennighoff/xwinograd", cfg="ru", train=None, dev=None, test="test", kind="mcq", family="bench_commonsense", lang="ru", caps=(0, 0, 0, 400), lic="permissive", eval_only=True, opts=dict(get=mcq_winogrande))

# ---------------------------------------------------------------------------------------------- extractive QA / NER
add(id="squad2", repo="rajpurkar/squad_v2", kind="span", family="qa_extract", lang="en", dev="validation", caps=(90000, 150, 300, 1500), lic="sa", opts=dict(get=span_squad))
add(id="mrqa", repo="mrqa-workshop/mrqa", kind="span", family="qa_extract", lang="en", dev="validation", caps=(45000, 100, 200, 800), lic="unknown", opts=dict(get=span_mrqa))
add(id="sberquad", repo="kuznetsoffandrey/sberquad", kind="span", family="qa_extract", lang="ru", dev="validation", test="test", caps=(45000, 150, 300, 1000), lic="unknown", opts=dict(get=span_squad))
add(id="tydiqa", repo="google-research-datasets/tydiqa", cfg="secondary_task", kind="span", family="qa_extract", lang="mix", dev="validation", caps=(45000, 100, 200, 600), lic="permissive", opts=dict(get=span_squad))
add(id="ropes", repo="allenai/ropes", kind="span", family="qa_extract", lang="en", dev="validation", test="test", caps=(10000, 60, 100, 300), lic="permissive", opts=dict(get=span_ropes))
add(id="xquad_ru", repo="google/xquad", cfg="xquad.ru", train=None, dev=None, test="validation", kind="span", family="bench_qa", lang="ru", caps=(0, 0, 0, 1190), lic="sa", eval_only=True, heldout=True, opts=dict(get=span_squad))
for lg in ["ru", "en", "de", "es", "fr", "nl", "pl", "it", "pt", "zh"]:
    add(id=f"wikiann_{lg}", repo="unimelb-nlp/wikiann", cfg=lg, kind="ner", family="ner", lang=lg, dev="validation", test="test", caps=(15000 if lg in ("ru", "en") else 2500, 80, 150, 300), lic="unknown",
        opts=dict(types=["PER", "ORG", "LOC"]))
add(id="fewnerd", repo="DFKI-SLT/few-nerd", cfg="supervised", kind="ner", family="ner", lang="en", dev="validation", test="test", caps=(50000, 150, 300, 800), lic="sa", opts=dict(bio=False))
add(id="nuner", repo="numind/NuNER", kind="entlist", family="ner", lang="en", train="entity", dev=None, caps=(120000, 0, 0, 0), lic="permissive", opts=dict(get=ner_pairs_from_numind), note="carved by hash")
add(id="gliner_multi", repo="knowledgator/GLINER-multi-task-synthetic-data", kind="entlist", family="ner", lang="en", dev=None, caps=(60000, 0, 0, 0), lic="permissive", opts=dict(get=ner_pairs_from_gliner), note="carved by hash")
add(id="pile_mistral_ner", repo="urchade/pile-mistral-v0.1", kind="entlist", family="ner", lang="en", dev=None, caps=(60000, 0, 0, 0), lic="permissive", opts=dict(get=ner_pairs_from_gliner), note="carved by hash")

# ---------------------------------------------------------------------------------------------- scoring
add(id="stsb", repo="mteb/stsbenchmark-sts", kind="sim", family="similarity", lang="en", dev="validation", test="test", caps=(5749, 100, 200, 1000), lic="unknown", opts={})
for lg in ["en", "ru", "de", "fr", "es", "it", "nl", "pl", "pt", "zh"]:
    add(id=f"stsb_{lg}", repo="PhilipMay/stsb_multi_mt", cfg=lg, kind="sim", family="similarity", lang=lg, dev="dev", test="test", caps=(5749, 60, 100, 300), lic="unknown",
        opts=dict(score="similarity_score"))


def score_helpsteer(row):
    out = []
    names = {"helpfulness": ("helpfulness of the response", "полезность ответа"), "correctness": ("factual correctness of the response", "фактическая верность ответа"),
             "coherence": ("coherence of the response", "связность ответа"), "complexity": ("complexity of the response", "сложность ответа"), "verbosity": ("verbosity of the response", "многословность ответа")}
    keys = list(names)
    import random as _r

    rr = _r.Random(hash(row.get("prompt", "")) & 0xFFFF)
    for k in rr.sample(keys, 2):
        v = row.get(k)
        if v is None:
            continue
        state = "Request:\n" + clean(row.get("prompt"), 1500) + "\n\nResponse:\n" + clean(row.get("response"), 2500)
        out.append((state, names[k][0], names[k][1], int(v) + 1, 5))
    return out


add(id="helpsteer2", repo="nvidia/HelpSteer2", kind="score", family="quality", lang="en", dev="validation", caps=(40000, 150, 300, 600), lic="permissive", opts=dict(get=score_helpsteer))
add(id="helpsteer", repo="nvidia/HelpSteer", kind="score", family="quality", lang="en", dev="validation", caps=(30000, 100, 200, 400), lic="permissive", opts=dict(get=score_helpsteer))


def score_ultrafeedback(row):
    import random as _r

    comps = row.get("completions") or []
    if not comps:
        return []
    rr = _r.Random(hash(row.get("instruction", "")) & 0xFFFFF)
    c = rr.choice(comps)
    ann = c.get("annotations") or {}
    aspects = {"helpfulness": ("helpfulness of the response", "полезность ответа"), "honesty": ("honesty of the response", "честность ответа"),
               "instruction_following": ("how well the response follows the instruction", "насколько ответ следует инструкции"), "truthfulness": ("truthfulness of the response", "правдивость ответа")}
    a = rr.choice([k for k in aspects if k in ann])
    rating = (ann[a] or {}).get("Rating") if isinstance(ann[a], dict) else None
    if isinstance(ann[a], list) and ann[a]:
        rating = ann[a][0].get("Rating")
    try:
        lvl = int(rating)
    except (TypeError, ValueError):
        return []
    if not 1 <= lvl <= 5:
        return []
    state = "Instruction:\n" + clean(row.get("instruction"), 1500) + "\n\nResponse:\n" + clean(c.get("response"), 2500)
    return [(state, aspects[a][0], aspects[a][1], lvl, 5)]


add(id="ultrafeedback", repo="openbmb/UltraFeedback", kind="score", family="quality", lang="en", dev=None, caps=(80000, 0, 0, 0), lic="permissive", opts=dict(get=score_ultrafeedback), note="carved by hash")


def score_kinopoisk(row):
    g = str(row.get("grade10") or "").strip()
    if not g.isdigit() or not 1 <= int(g) <= 10:
        return []
    return [(clean((row.get("title") or "") + "\n" + (row.get("content") or ""), 3500), "overall rating of the movie in this review", "общая оценка фильма в этом отзыве", int(g), 10)]


add(id="kinopoisk_grade", repo="blinoff/kinopoisk", kind="score", family="sentiment", lang="ru", dev=None, caps=(40000, 0, 0, 0), lic="permissive", opts=dict(get=score_kinopoisk), note="carved by hash")

# ---------------------------------------------------------------------------------------------- safety / moderation
def bin_civil(row):
    t = row.get("toxicity")
    return [(clean(row.get("text")), "Is this comment toxic?", "Является ли этот комментарий токсичным?", float(t))] if t is not None else []


add(id="civil_comments", repo="google/civil_comments", kind="binary", family="safety", lang="en", dev="validation", test="test", caps=(80000, 150, 300, 1000), lic="permissive", opts=dict(get=bin_civil))


def bin_jigsaw(row):
    import random as _r

    rr = _r.Random(hash(row.get("comment_text", "")) & 0xFFFFF)
    cats = {"toxic": ("Is this comment toxic?", "Является ли комментарий токсичным?"), "obscene": ("Does this comment contain obscene language?", "Содержит ли комментарий непристойную лексику?"),
            "threat": ("Does this comment contain a threat?", "Содержит ли комментарий угрозу?"), "insult": ("Is this comment insulting?", "Оскорбляет ли этот комментарий?"),
            "identity_hate": ("Does this comment attack a group of people?", "Нападает ли комментарий на группу людей?")}
    k = rr.choice(list(cats))
    v = row.get(k)
    return [(clean(row.get("comment_text")), cats[k][0], cats[k][1], float(v))] if v is not None else []


add(id="jigsaw", repo="Arsive/toxicity_classification_jigsaw", kind="binary", family="safety", lang="en", dev="validation", test="test", caps=(60000, 150, 300, 800), lic="permissive", opts=dict(get=bin_jigsaw))


def bin_textdetox(row):
    return [(clean(row.get("text")), "Is this text toxic or offensive?", "Является ли этот текст токсичным или оскорбительным?", float(row["toxic"]))]


for lg in ["en", "ru", "uk", "de", "es", "zh", "ar", "hi", "it", "fr", "ja"]:
    add(id=f"textdetox_{lg}", repo="textdetox/multilingual_toxicity_dataset", train=lg, kind="binary", family="safety", lang=lg, dev=None, caps=(4000 if lg != "en" else 6000, 40, 80, 250), lic="other",
        opts=dict(get=bin_textdetox), note="carved by hash; OpenRAIL++")


def bin_aegis(row):
    lab = row.get("prompt_label")
    if lab not in ("safe", "unsafe"):
        return []
    return [(clean(row.get("prompt")), "Is this request unsafe or harmful?", "Является ли этот запрос небезопасным или вредным?", float(lab == "unsafe"))]


add(id="aegis2", repo="nvidia/Aegis-AI-Content-Safety-Dataset-2.0", kind="binary", family="safety", lang="en", dev="validation", test="test", caps=(25000, 100, 200, 500), lic="permissive", opts=dict(get=bin_aegis))
add(id="prompt_injection_deepset", repo="deepset/prompt-injections", kind="binary", family="injection", lang="mix", dev="test", caps=(550, 30, 50, 80), lic="permissive",
    opts=dict(get=lambda r: [(clean(r["text"]), "Does this text try to override or hijack the assistant's instructions?", "Пытается ли этот текст переопределить или перехватить инструкции ассистента?", float(r["label"]))]))
add(id="prompt_injection_safeguard", repo="xTRam1/safe-guard-prompt-injection", kind="binary", family="injection", lang="en", dev="test", caps=(8000, 80, 150, 400), lic="unknown",
    opts=dict(get=lambda r: [(clean(r["text"]), "Does this text try to override or hijack the assistant's instructions?", "Пытается ли этот текст переопределить или перехватить инструкции ассистента?", float(r["label"]))]))
add(id="jailbreak", repo="jackhhao/jailbreak-classification", kind="binary", family="injection", lang="en", dev="test", caps=(1200, 40, 60, 100), lic="permissive",
    opts=dict(get=lambda r: [(clean(r["prompt"]), "Is this prompt a jailbreak attempt?", "Является ли этот промпт попыткой джейлбрейка?", float(r["type"] == "jailbreak"))]))
add(id="toxigen", repo="skg/toxigen-data", cfg="train", kind="binary", family="safety", lang="en", dev=None, caps=(40000, 0, 0, 0), lic="unknown",
    opts=dict(get=lambda r: [(clean(r["generation"]), "Is this text hateful toward a group of people?", "Выражает ли этот текст ненависть к группе людей?", float(r["prompt_label"]))]), note="carved by hash")
add(id="toxic_chat", repo="lmsys/toxic-chat", cfg="toxicchat0124", train=None, dev=None, test="test", kind="binary", family="safety", lang="en", caps=(0, 0, 0, 1000), lic="nc", eval_only=True, heldout=True,
    opts=dict(get=lambda r: [(clean(r["user_input"]), "Is this user message toxic?", "Является ли это сообщение пользователя токсичным?", float(r["toxicity"]))]), note="NC licence: evaluation only")


def pw_hh(row):
    ch, rj = row.get("chosen") or "", row.get("rejected") or ""
    i = ch.rfind("\n\nAssistant:")
    j = rj.rfind("\n\nAssistant:")
    if i < 0 or j < 0 or ch[:i] != rj[:j]:
        return None
    return ch[:i].replace("\n\nHuman:", "\nUser:").replace("\n\nAssistant:", "\nAssistant:").strip(), ch[i + 12 :].strip(), rj[j + 12 :].strip()


add(id="hh_rlhf", repo="Anthropic/hh-rlhf", kind="pairwise", family="preference", lang="en", dev="test", caps=(50000, 100, 200, 600), lic="permissive", opts=dict(get=pw_hh))


def pw_ultra(row):
    def last(m):
        if isinstance(m, list) and m:
            return m[-1].get("content", "")
        return str(m)

    return clean(row.get("prompt"), 1800), last(row.get("chosen")), last(row.get("rejected"))


add(id="ultrafeedback_pairs", repo="argilla/ultrafeedback-binarized-preferences-cleaned", kind="pairwise", family="preference", lang="en", dev=None, caps=(30000, 0, 0, 0), lic="permissive", opts=dict(get=pw_ultra), note="carved by hash")


def score_summeval(row):
    import random as _r

    summaries = row.get("machine_summaries") or []
    if not summaries:
        return []
    rr = _r.Random(hash(row.get("text", "")[:60]) & 0xFFFFF)
    i = rr.randrange(len(summaries))
    asp = rr.choice([("relevance", "relevance of the summary", "релевантность резюме"), ("coherence", "coherence of the summary", "связность резюме"),
                     ("fluency", "fluency of the summary", "беглость резюме"), ("consistency", "factual consistency of the summary", "фактическая согласованность резюме")])
    vals = row.get(asp[0]) or []
    if i >= len(vals):
        return []
    v = max(1.0, min(5.0, float(vals[i])))
    state = "Article:\n" + clean(row.get("text"), 3000) + "\n\nSummary:\n" + clean(summaries[i], 800)
    return [(state, asp[1], asp[2], v, 5)]


add(id="summeval", repo="mteb/summeval", train=None, dev=None, test="test", kind="score", family="bench_quality", lang="en", caps=(0, 0, 0, 600), lic="permissive", eval_only=True, opts=dict(get=score_summeval))

# ---------------------------------------------------------------------------------------------- browser + function calling
add(id="mind2web", repo="osunlp/Multimodal-Mind2Web", kind="mind2web", family="browser", lang="en", train="train", dev=None, test="test_website", caps=(6000, 0, 0, 500), lic="other",
    opts=dict(drop_cols=["screenshot", "raw_html"]), note="element ranking over cleaned-HTML candidates; test_website = unseen websites")
add(id="glaive_fc", repo="glaiveai/glaive-function-calling-v2", kind="glaive", family="tools", lang="en", dev=None, caps=(60000, 0, 0, 0), lic="permissive", note="carved by hash")

from .kinds import register_extra  # noqa: E402

register_extra()


def by_name(name: str) -> Spec:
    for s in SPECS:
        if s.id == name:
            return s
    raise KeyError(name)
