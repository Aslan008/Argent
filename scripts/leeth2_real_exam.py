"""The exam that uses real traffic: what Leeth 2 does with the requests people actually typed into Argent.

    python scripts/leeth2_real_exam.py <model_dir> [--queries .argent/real_queries.json] [--labels scripts/real_exam_v4.json]

Real requests are almost never one tool call ("проанализируй план", "продолжай", pasted errors, weather).  The costly mistake
is therefore not a wrong tool but a FALSE DISPATCH: the fast path grabbing a request that needed the LLM.  Per request the
script plays the dispatcher's gates (need_tool, tool confidence, allowed tools; same thresholds) and reports

  false-dispatch rate   share of "delegate" requests that the fast path would have taken      (lower is better)
  tool recall           share of labelled one-tool requests routed to the right tool          (higher is better)
  context robustness    the same requests re-asked inside a production session context (a running server, a last file,
                        a previous turn): the answer must not change for requests that do not depend on it

The labels file lists only the requests whose right answer is a tool; everything else in the query file is "delegate".
Nothing here is trained on (finetune.py reads the same files to keep them out of the training data).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.leeth.dispatcher import ALLOWED_FAST_TOOLS  # noqa: E402
from src.leeth.primitives import ChoiceQuestion, NoulQuestion  # noqa: E402
from src.leeth2.data.argent import TOOL_QUESTION  # noqa: E402
from src.leeth2.data.synth import argent_context as CX  # noqa: E402
from src.leeth2.data.synth.argent_router import NEED_TOOL_QUESTION  # noqa: E402
from src.leeth2.runtime import Leeth2Runtime  # noqa: E402

THRESHOLD = 0.65          # dispatcher.confidence_threshold
EXEC_THRESHOLD = 0.80
WORKDIR = "C:/Users/user/Desktop/Argent"
DELEGATE = {"NO_TOOL", "UNFAMILIAR"}

# The labels file names the requests whose right answer is a tool; every other request counts as "delegate".  That is the
# conservative reading and it stays the headline number.  But some unlabelled requests ARE one clean tool call, and the fast path
# taking them is not a mistake.  The rules below are deliberately narrow and mechanical (no judgement per query), and the adjusted rate
# is reported NEXT to the headline one, for every model alike.
import re  # noqa: E402

_ARITH = re.compile(r"^\s*(?:!fast\s*)?(?:сколько будет|посчитай|вычисли|calculate|what is|чему равно)?\s*[-+*/().,^%×÷=?\s]*\d[-+*/().,^%×÷=?\s\d]*[-+*/^%×÷][-+*/().,^%×÷=?\s\d]*\d[-+*/().,^%×÷=?\s\d]*$", re.I)
_GIT = re.compile(r"^\s*(?:покажи|выведи|что в|show|give me)?\s*(?:статус\s+(?:git|гита?|гит)|git status)\s*\??\s*$", re.I)
_ADDRESS = re.compile(r"https?://|\b[\w-]+\.(?:com|org|net|dev|io|kz|ru|app)\b", re.I)


def defensible(query: str, tool: str) -> bool:
    """A fast-path pick on an unlabelled request that is nevertheless exactly one clean call."""
    if tool == "calculate":
        return bool(_ARITH.match(query))
    if tool == "run_command":
        return bool(_GIT.match(query))
    if tool == "browser_open":
        return bool(_ADDRESS.search(query)) and len(query.split()) <= 6
    return False


def decide(rt: Leeth2Runtime, queries: List[str], ctxs: List[Any]) -> List[Dict[str, Any]]:
    out = []
    qs = [NoulQuestion("need", NEED_TOOL_QUESTION), ChoiceQuestion("tool", TOOL_QUESTION, list(CX.PROD_OPTIONS))]
    for q, ctx in zip(queries, ctxs):
        r = rt.decide(CX.build_state(q, ctx, WORKDIR), qs)
        need, tool = r.results["need"], r.results["tool"]
        need_ok = need.value and need.probability >= THRESHOLD
        floor = EXEC_THRESHOLD if tool.choice == "run_command" else THRESHOLD
        taken = bool(need_ok and tool.choice in ALLOWED_FAST_TOOLS and tool.confidence >= floor)
        out.append({"query": q, "tool": tool.choice, "conf": round(tool.confidence, 3), "need": round(need.probability, 3), "taken": taken})
    return out


def production_path(rt: Leeth2Runtime, queries: List[str]) -> List[Dict[str, Any]]:
    """What the DISPATCHER itself does with each request: its own triage -- the model's gates plus its regex guards, the argument checks
    (a site name is not an address, a sentence is not a file name), the deny list and the safety fan-out for shell commands.  The numbers
    above are the model's; this one is what the owner meets in Argent."""
    import types

    try:
        import config as cfg
    except Exception:                                    # the slice of the project on Kaggle has no config.py
        cfg = types.ModuleType("config")
        sys.modules["config"] = cfg
    cfg.get_leeth_backend = lambda: "leeth2"
    from src.leeth.dispatcher import LeethDispatcher
    from src.leeth.runtime import leeth

    leeth._v2 = rt
    d = LeethDispatcher()
    out = []
    for i in range(0, len(queries), 24):
        for info in d.triage(queries[i:i + 24]):
            disp = info.get("dispatch")
            out.append({"query": info["query"], "taken": bool(disp), "tool": disp["tool"] if disp else (info.get("tool") or {}).get("choice", ""), "skip": info.get("skip_reason") or ""})
    return out


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--queries", default=str(ROOT / ".argent" / "real_queries.json"))
    ap.add_argument("--labels", default=str(ROOT / "scripts" / "real_exam_v4.json"))
    ap.add_argument("--out", default="")
    ap.add_argument("--device", default="")
    ap.add_argument("--production-only", action="store_true", help="only the dispatcher-path measurement (skips the model-only runs)")
    a = ap.parse_args(argv)

    queries: List[str] = json.loads(Path(a.queries).read_text(encoding="utf-8"))
    lab = json.loads(Path(a.labels).read_text(encoding="utf-8")) if Path(a.labels).exists() else {"tools": {}, "skip": []}
    tools: Dict[str, str] = lab["tools"]
    skip = set(lab.get("skip", []))
    queries = [q for q in queries if q not in skip]
    rt = Leeth2Runtime.load(a.model_dir, device=a.device or None, max_len=4096)

    prod = production_path(rt, queries)
    p_delegate = [r for r in prod if r["query"] not in tools]
    p_false = [r for r in p_delegate if r["taken"]]
    p_labelled = [r for r in prod if r["query"] in tools]
    p_right = [r for r in p_labelled if r["taken"] and r["tool"] == {"find_by_name": "search_files"}.get(tools[r["query"]], tools[r["query"]])]
    p_wrong = [r for r in p_labelled if r["taken"] and r not in p_right]
    print(f"PRODUCTION PATH (dispatcher.triage: regex guards, argument checks, deny list, shell safety)  false-dispatch {len(p_false) / max(1, len(p_delegate)):.2%} "
          f"({len(p_false)}/{len(p_delegate)}); adjusted {sum(1 for r in p_false if not defensible(r['query'], r['tool']))}; "
          f"tool recall {len(p_right) / max(1, len(p_labelled)):.1%} ({len(p_right)}/{len(p_labelled)}), wrong tool: {len(p_wrong)}")
    for r in p_false[:40]:
        print(f"  PROD-FALSE  {r['tool']:<22} {r['query'][:100]!r}")
    if a.production_only:
        if a.out:
            Path(a.out).write_text(json.dumps({"production": prod}, ensure_ascii=False, indent=1), encoding="utf-8")
        return 0

    bare = decide(rt, queries, [None] * len(queries))
    delegate = [r for r in bare if r["query"] not in tools]
    labelled = [r for r in bare if r["query"] in tools]
    false_taken = [r for r in delegate if r["taken"]]
    right = [r for r in labelled if r["taken"] and r["tool"] == tools[r["query"]]]
    wrong_tool = [r for r in labelled if r["taken"] and r["tool"] != tools[r["query"]]]
    print(f"real requests {len(bare)}: {len(delegate)} delegate, {len(labelled)} one-tool")
    print(f"false-dispatch rate  {len(false_taken) / max(1, len(delegate)):.2%}  ({len(false_taken)}/{len(delegate)})")
    adj = [r for r in false_taken if not defensible(r["query"], r["tool"])]
    print(f"  adjusted (a lone arithmetic sum / git status / bare address is not counted)  {len(adj) / max(1, len(delegate)):.2%}  ({len(adj)}/{len(delegate)})")
    print(f"tool recall          {len(right) / max(1, len(labelled)):.1%}  ({len(right)}/{len(labelled)}), wrong tool taken: {len(wrong_tool)}")
    for r in false_taken[:40]:
        print(f"  FALSE  {r['tool']:<22} p={r['conf']:.2f}  {r['query'][:100]!r}")
    for r in wrong_tool:
        print(f"  WRONG  {r['tool']:<22} want {tools[r['query']]}  {r['query'][:100]!r}")
    for r in labelled:
        if r not in right and r not in wrong_tool:
            print(f"  missed {r['tool']:<22} want {tools[r['query']]}  {r['query'][:100]!r}")

    # the same delegate requests inside a session context: the fast path must not start grabbing them
    rng = random.Random(7)
    ctxs = [CX.make_context(rng, prev=True) for _ in delegate]
    ctx_run = decide(rt, [r["query"] for r in delegate], ctxs)
    ctx_false = [r for r in ctx_run if r["taken"]]
    print(f"false-dispatch rate with session context  {len(ctx_false) / max(1, len(ctx_run)):.2%}  ({len(ctx_false)}/{len(ctx_run)})")
    adj_ctx = [r for r in ctx_false if not defensible(r["query"], r["tool"])]
    print(f"  adjusted  {len(adj_ctx) / max(1, len(ctx_run)):.2%}  ({len(adj_ctx)}/{len(ctx_run)})")
    for r in ctx_false[:20]:
        print(f"  FALSE(ctx)  {r['tool']:<22} p={r['conf']:.2f}  {r['query'][:100]!r}")
    if a.out:
        Path(a.out).write_text(json.dumps({"bare": bare, "ctx": ctx_run, "production": prod}, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
