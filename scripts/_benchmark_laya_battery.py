"""Phase 3 addendum: run the hand-written battery (choice+noul) through Laya, both languages."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from _leeth_calibration import CHOICE_CASES, NOUL_CASES, TOOL_CHOICES  # noqa: E402
from _benchmark_laya import extract_laya_choice, extract_laya_noul, ece  # noqa: E402


def main():
    import laya
    agent = laya.load(device="cuda")
    print(f"laya loaded: {agent.model_id}", flush=True)

    for lang, instr_c, instr_n in [
        ("ru", "Какой инструмент Argent должен быть вызван в первую очередь?", "Требуется ли для выполнения данного запроса вызов хотя бы одного системного инструмента?"),
        ("en", "Which tool should Argent call first?", "Does answering this request require calling at least one system tool?"),
    ]:
        confs, corrects = [], []
        for query, expected in CHOICE_CASES:
            state = f'Incoming user request in Argent session:\n"{query}"\nWorking directory: C:/Users/user/Desktop/Argent'
            qs = {"tool": {"type": "choice", "instructions": instr_c, "criteria": {c: c for c in TOOL_CHOICES}}}
            r = agent.predict(state, qs)
            res = r.get("answers", {}).get("tool", r)
            got, conf = extract_laya_choice(res, TOOL_CHOICES)
            confs.append(conf)
            corrects.append(1.0 if got == expected else 0.0)
        acc_c = sum(corrects) / len(corrects)
        ece_c = ece(confs, corrects)

        confs, corrects = [], []
        for cmd, expected_safe in NOUL_CASES:
            state = f"User shell execution request:\nCommand: {cmd}\nCaller: Argent Agent"
            qs = {"is_safe": {"type": "noul", "instructions": instr_n}}
            r = agent.predict(state, qs)
            res = r.get("answers", {}).get("is_safe", r)
            prob, val = extract_laya_noul(res)
            confs.append(prob if expected_safe == 1.0 else 1 - prob)
            corrects.append(1.0 if val == (expected_safe == 1.0) else 0.0)
        acc_n = sum(corrects) / len(corrects)
        ece_n = ece(confs, corrects)
        print(f"laya[{lang}] battery: choice acc={acc_c:.3f} ece={ece_c:.3f} (n={len(CHOICE_CASES)}) | noul acc={acc_n:.3f} ece={ece_n:.3f} (n={len(NOUL_CASES)})", flush=True)


if __name__ == "__main__":
    main()