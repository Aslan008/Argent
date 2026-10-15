"""
scripts/_leeth_ood_probe.py — calibrates the OOD familiarity threshold.

Measures EVERY candidate familiarity signal on:
  - IN-DIST queries: routine dispatcher territory (should look familiar)
  - OOD queries: unfamiliar languages/domains/phrasings (should look alien)

Signals (per state, from the SAME decide_batch forward pass):
  - state_nll          : mean MLM NLL over content tokens (low = familiar)
  - tool_confidence    : max prob of the normalized tool-choice distribution (high = familiar)
  - tool_entropy       : Shannon entropy of that distribution in nats (low = familiar)
  - need_tool_prob     : noul head probability (informational)

Reports per-signal stats for both sets, separation quality, and a suggested
threshold for the best signal. The value is meant to be wired into
dispatcher.familiarity_threshold after human review.
"""

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from src.leeth import leeth  # noqa: E402
from src.leeth.primitives import ChoiceQuestion, ExtractQuestion, NoulQuestion  # noqa: E402

TOOL_CHOICES = [
    "read_file", "write_file", "replace_in_file", "run_command",
    "grep_search", "find_by_name", "list_directory", "browser_open",
    "browser_click", "browser_type", "browser_upload_file", "calculate",
    "ask_user_questions", "NO_TOOL", "UNFAMILIAR",
]

# Exact dispatcher question set (mirrors _build_state_and_questions).
QUESTIONS = [
    NoulQuestion(id="need_tool", question="Требуется ли для выполнения данного запроса вызов хотя бы одного системного инструмента?"),
    ChoiceQuestion(id="tool", question="Какой инструмент Argent должен быть вызван в первую очередь?", choices=TOOL_CHOICES),
    ExtractQuestion(id="param", question="Извлеките ключевой целевой параметр (путь к файлу, запрос или команду)"),
]


def _state(query: str) -> str:
    return f'Incoming user request in Argent session:\n"{query}"\nWorking directory: C:/Users/user/Desktop/Argent'


# Routine territory: the exact query shapes the dispatcher serves daily.
IN_DIST = [
    "прочитай config.py", "запусти pytest -x", "сколько будет 512 / 8",
    "покажи что в README", "найди файл feedback.py", "поищи класс AgentState в коде",
    "покажи содержимое папки src", "выполни git status", "посчитай 2^10",
    "прочитай requirements.txt", "выполни git diff HEAD~1", "какие у тебя есть возможности?",
    "а что ты умеешь?", "продолжай", "удали папку build",
]

# Unfamiliar territory: unseen languages, exotic phrasings, alien domains.
OOD = [
    "скомпилируй Main.go и запусти получившийся бинарник",
    "покажи стектрейс из NullPointerException в Java-сервисе",
    "cargo build --release и покажи warnings",
    "запусти docker compose up -d для стека мониторинга",
    "отрефактори этот Haskell-модуль под newtype",
    "kubectl get pods -n production и опиши статус",
    "переведи этот текст на французский и сохрани в файл",
    "нарисуй диаграмму архитектуры в mermaid",
    "спарси этот CSV и построй сводную таблицу продаж",
    "настрой nginx reverse proxy для двух бэкендов",
    "напиши SQL-миграцию для добавления индекса",
    "оптимизируй этот pandas-пайплайн по памяти",
]


def _entropy(probs: dict) -> float:
    vals = [p for p in probs.values() if p > 0]
    total = sum(vals) or 1.0
    return -sum((v / total) * math.log(v / total) for v in vals)


def measure(queries):
    states = [_state(q) for q in queries]
    batches = leeth.decide_batch(states, QUESTIONS)
    rows = []
    for q, b in zip(queries, batches):
        tool = b.results.get("tool")
        need = b.results.get("need_tool")
        rows.append({
            "query": q,
            "state_nll": b.state_nll,
            "tool_confidence": tool.confidence if tool else None,
            "tool_choice": tool.choice if tool else None,
            "tool_entropy": _entropy(tool.probabilities) if tool and tool.probabilities else None,
            "need_tool_prob": need.probability if need else None,
        })
    return rows


def stats(rows, key):
    vals = sorted(r[key] for r in rows if r[key] is not None)
    if not vals:
        return {"n": 0}
    n = len(vals)
    return {
        "n": n,
        "min": round(vals[0], 3),
        "p01": round(vals[max(0, int(n * 0.01))], 3),
        "p50": round(vals[n // 2], 3),
        "p90": round(vals[int(n * 0.9)], 3),
        "p99": round(vals[min(n - 1, int(n * 0.99))], 3),
        "max": round(vals[-1], 3),
        "mean": round(sum(vals) / n, 3),
    }


def main():
    in_dist = measure(IN_DIST)
    ood = measure(OOD)

    signals = [
        # (key, familiar_direction, in_thr_quantile, ood_caught_cmp)
        ("state_nll", "low", 0.99, "gt"),      # familiar = low; thr = in-dist p99; OOD caught if > thr
        ("tool_confidence", "high", 0.01, "lt"),  # familiar = high; thr = in-dist p01; OOD caught if < thr
        ("tool_entropy", "low", 0.99, "gt"),
        ("need_tool_prob", "high", 0.01, "lt"),
    ]

    report = {"in_dist": {}, "ood": {}}
    for key, *_ in signals:
        s_in, s_ood = stats(in_dist, key), stats(ood, key)
        report["in_dist"][key] = s_in
        report["ood"][key] = s_ood

    print("=== IN-DIST (routine dispatcher queries) ===")
    for r in in_dist:
        print(f"  nll={r['state_nll']} conf={r['tool_confidence']:.3f} H={r['tool_entropy']:.3f} need={r['need_tool_prob']:.3f}  {r['query']}")
    print("\n=== OOD (unseen languages/domains) ===")
    for r in ood:
        print(f"  nll={r['state_nll']} conf={r['tool_confidence']:.3f} H={r['tool_entropy']:.3f} need={r['need_tool_prob']:.3f}  {r['query']}")

    print("\n=== SIGNAL SEPARATION ===")
    best = None
    for key, direction, q, cmp_op in signals:
        s_in, s_ood = report["in_dist"][key], report["ood"][key]
        if not s_in.get("n") or not s_ood.get("n"):
            continue
        thr = s_in[f"p{int(q * 100):02d}"] if f"p{int(q * 100):02d}" in s_in else (s_in["p99"] if q == 0.99 else s_in["p01"])
        if cmp_op == "gt":
            caught = sum(1 for r in ood if r[key] is not None and r[key] > thr)
            false_pos = sum(1 for r in in_dist if r[key] is not None and r[key] > thr)
        else:
            caught = sum(1 for r in ood if r[key] is not None and r[key] < thr)
            false_pos = sum(1 for r in in_dist if r[key] is not None and r[key] < thr)
        sep = s_ood["mean"] - s_in["mean"] if direction_meaning(key) == "ood_higher" else s_in["mean"] - s_ood["mean"]
        print(f"  {key:16s} in_mean={s_in['mean']:7.3f} ood_mean={s_ood['mean']:7.3f} sep={sep:+7.3f} | thr={thr:7.3f} caught={caught}/{s_ood['n']} fp={false_pos}/{s_in['n']}")
        if best is None or caught > best[1] or (caught == best[1] and false_pos < best[2]):
            best = (key, caught, false_pos, thr, cmp_op)

    if best:
        print(f"\nBEST SIGNAL: {best[0]} (caught {best[1]}/{len(OOD)}, false positives {best[2]}/{len(IN_DIST)})")
        print(f"suggested threshold: {best[3]} ({'OOD if value <' if best[4] == 'lt' else 'OOD if value >'} threshold)")

    # Explicit UNFAMILIAR-class evaluation (phase 3 signal).
    unfam_ood = sum(1 for r in ood if r["tool_choice"] == "UNFAMILIAR")
    unfam_in = sum(1 for r in in_dist if r["tool_choice"] == "UNFAMILIAR")
    print(f"\nUNFAMILIAR class: OOD flagged {unfam_ood}/{len(OOD)} | IN-DIST false flags {unfam_in}/{len(IN_DIST)}")

    out = ROOT / ".argent" / "artifacts" / "leeth_ood_probe.json"
    out.write_text(json.dumps({"in_dist": in_dist, "ood": ood, "stats": report}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {out}")


def direction_meaning(key):
    # Which set is expected higher for this signal
    return {"state_nll": "in_high", "tool_confidence": "in_high", "tool_entropy": "in_low", "need_tool_prob": "in_high"}[key]


if __name__ == "__main__":
    main()