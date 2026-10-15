"""Разбор лога Kaggle-кернела dopt v7: stdout + ошибки."""
import json

rows = json.load(open(".argent/kaggle/kernel_dopt_output/leeth-dopt-v7-dagger.log", encoding="utf-8"))
print("всего записей:", len(rows))
picked = [r for r in rows if r["stream_name"] == "stdout" or "rror" in r["data"]]
for r in picked[-16:]:
    print(f"[{r['time']:7.1f}s] {r['stream_name'][:3]}: {r['data'][:250]}", end="")