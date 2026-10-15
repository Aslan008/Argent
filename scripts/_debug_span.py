"""Debug span alignment failures."""
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.leeth.diverse_generator import DiverseUniversalGenerator
from src.leeth.tokenizer import LeethTokenizer
from src.leeth.train import LeethTorchDataset

tok = LeethTokenizer.load(ROOT / "src/leeth/data")
c2i = json.load(open(ROOT / "src/leeth/data/choice_classes.json", encoding="utf-8"))
g = DiverseUniversalGenerator(seed=1234)
samples = []
for i, s in enumerate(g.generate_stream()):
    samples.append(s)
    if i >= 299:
        break

ds = LeethTorchDataset(samples, tok, max_length=384, class_to_idx=c2i)
shown = 0
for i, s in enumerate(samples):
    if s.question_type != "extract" or not s.target_span_text:
        continue
    item = ds[i]
    ids = item["input_ids"].tolist()
    st, en = item["start_target"].item(), item["end_target"].item()
    decoded = tok.decode(ids[st:en + 1], skip_special_tokens=True)
    tgt = s.target_span_text
    close = tgt.strip().lower() in decoded.lower() or decoded.strip().lower() in tgt.strip().lower()
    if not close and shown < 6:
        shown += 1
        print(f"--- case {i}")
        print(f"  target: {tgt!r}")
        print(f"  decoded[{st}:{en}]: {decoded!r}")
        print(f"  state head: {s.state[:90]!r}")
        span_toks = tok.tokenize(tgt)
        print(f"  span tokens: {span_toks[:6]}")
        text = f"[STATE] {s.state} [SEP] [QUESTION] [EXTRACT] {s.question}"
        print(f"  target in composed text: {tgt in text}")
print("done")