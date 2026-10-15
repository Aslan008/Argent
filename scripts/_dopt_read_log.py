"""Разбор лога Kaggle-кернела dopt: stdout + ошибки."""
import json

rows = json.load(open(".argent/kaggle/kernel_dopt_output/leeth-dopt-kernel.log", encoding="utf-8"))
print("всего записей:", len(rows))
for r in rows:
    d = r["data"]
    if r["stream_name"] == "stdout" or "Error" in d or "error" in d.lower():
        print(f"[{r['time']:7.1f}s] {r['stream_name'][:3]}: {d[:300]}", end="")