"""Validate DiverseUniversalGenerator: speed, span validity, class coverage, token lengths."""
import sys, time, json, collections
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.leeth.diverse_generator import DiverseUniversalGenerator
from src.leeth.tokenizer import LeethTokenizer

tok = LeethTokenizer.load(ROOT / "src/leeth/data")
c2i = json.load(open(ROOT / "src/leeth/data/choice_classes.json", encoding="utf-8"))

g = DiverseUniversalGenerator(seed=1234)
t0 = time.perf_counter()
samples = []
for i, s in enumerate(g.generate_stream()):
    samples.append(s)
    if i >= 1999:
        break
dt = time.perf_counter() - t0
print(f"generated 2000 in {dt:.1f}s ({2000/dt:.0f}/s)")

ok_span = bad_span = 0
types = collections.Counter()
targets = collections.Counter()
for s in samples:
    types[s.question_type] += 1
    if s.target_choice:
        targets[s.target_choice] += 1
    if s.question_type == "extract" and s.target_span_text:
        text = f"[STATE] {s.state} [SEP] [QUESTION] [EXTRACT] {s.question}"
        if s.target_span_text in text:
            ok_span += 1
        else:
            bad_span += 1
print("types:", dict(types))
print(f"span substring match: {ok_span} ok, {bad_span} bad")
unknown = {t: c for t, c in targets.items() if t not in c2i}
print("targets outside 26-class universe:", unknown)
print("choice targets:", dict(targets))

lens = []
for s in samples[:300]:
    text = f"[STATE] {s.state} [SEP] [QUESTION] [{s.question_type.upper()}] {s.question}"
    lens.append(len(tok.encode(text, truncation=False)))
lens.sort()
print(f"token len P50={lens[len(lens)//2]} P95={lens[int(len(lens)*0.95)]} max={lens[-1]}")

# full pipeline check: LeethTorchDataset span alignment on 300 extract samples
from src.leeth.train import LeethTorchDataset
ds = LeethTorchDataset(samples[:300], tok, max_length=384, class_to_idx=c2i)
em_ok = 0
n_extract = 0
for i, s in enumerate(samples[:300]):
    if s.question_type != "extract" or not s.target_span_text:
        continue
    n_extract += 1
    item = ds[i]
    ids = item["input_ids"].tolist()
    st, en = item["start_target"].item(), item["end_target"].item()
    decoded = tok.decode(ids[st:en + 1], skip_special_tokens=True)
    tgt = s.target_span_text
    if tgt.strip().lower() in decoded.lower() or decoded.strip().lower() in tgt.strip().lower():
        em_ok += 1
print(f"dataset span alignment: {em_ok}/{n_extract} close match")
print("OK")