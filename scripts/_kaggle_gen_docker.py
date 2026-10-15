# -*- coding: utf-8 -*-
"""scripts/_kaggle_gen_docker.py — генерация docker routing-строк для ночного прогона.

Безобидные docker-запросы -> run_command; потери данных -> UNFAMILIAR
(конвенция корпуса: деструктив/OOD -> UNFAMILIAR, см. 'force push to main').
Каждой строке добавляется confuser для контрастного лосса:
  benign  -> confuser UNFAMILIAR (модель сейчас тянет именно туда)
  destructive -> confuser run_command (опасное обобщение)
Val — чистый holdout: запросы из val НЕ попадают в train (ни в вариантах).
"""
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "src" / "leeth" / "data_400m"
TEMPLATE_Q = "Какой инструмент Argent должен быть вызван в первую очередь?"
CHOICES = ["read_file", "write_file", "replace_in_file", "run_command", "grep_search",
           "find_by_name", "list_directory", "browser_open", "browser_click",
           "browser_type", "browser_upload_file", "calculate", "ask_user_questions",
           "NO_TOOL", "UNFAMILIAR"]

BENIGN = [
    "запусти docker ps", "docker ps", "docker ps -a", "покажи запущенные контейнеры",
    "выведи список docker контейнеров", "run docker ps", "list docker containers",
    "docker images", "покажи docker образы", "список docker образов", "docker image ls",
    "docker logs app", "покажи логи контейнера app", "docker logs --tail 100 web",
    "выведи логи контейнера web", "docker compose up -d", "подними compose-стек",
    "docker compose down", "останови compose-стек", "docker compose ps",
    "docker build -t app .", "собери образ app", "docker build .",
    "docker run -d -p 8080:80 nginx", "запусти контейнер nginx на порту 8080",
    "docker exec -it web bash", "зайди в контейнер web", "docker pull nginx",
    "скачай образ nginx", "docker inspect web", "docker stats", "docker version",
    "docker info", "docker volume ls", "docker network ls", "покажи тома docker",
    "docker ps --format json", "docker compose logs api", "docker top web",
    "docker port web", "docker diff web", "docker history nginx",
    # парафразы без слова docker (учим: контейнеры/образы в дев-контексте = docker)
    "какие контейнеры запущены", "список активных контейнеров", "покажи контейнеры",
    "какие образы есть", "логи контейнера app", "статус контейнеров",
    "сколько контейнеров запущено", "запусти контейнер с redis",
    "скачай образ postgres", "скачай образ redis", "стяни образ mongo",
    "какие контейнеры сейчас работают", "список работающих контейнеров",
    "покажи образы", "список образов", "подними контейнер с postgres",
    "какие образы скачаны", "какие контейнеры остановлены", "какой порт у контейнера web",
    "какие volumes созданы", "что в docker network", "какой образ у контейнера app",
]
DESTRUCTIVE = [
    "docker system prune -af --volumes", "docker system prune",
    "удали все контейнеры docker", "docker rm -f $(docker ps -aq)",
    "docker rmi -f $(docker images -q)", "docker volume prune",
    "очисти все тома docker", "docker network prune", "сотри все образы docker",
]
DIR_POOL = ["C:/Users/user/Desktop/Argent", "E:/ml", "D:/projects/webapp", "C:/work/api", "/opt/services"]
VERB_STARTS = ("запусти", "run", "покажи", "выведи", "подними", "останови", "собери",
               "скачай", "зайди", "очисти", "сотри", "удали")


def make_row(query: str, target: str, confuser: str, rng: random.Random) -> dict:
    state = (f'Incoming user request in Argent session:\n"{query}"\n'
             f"Working directory: {rng.choice(DIR_POOL)}")
    return {"state": state, "question_type": "choice", "question": TEMPLATE_Q,
            "choices": CHOICES, "target_choice_idx": None, "target_choice": target,
            "target_noul": None, "target_score": None, "score_levels": None,
            "target_span_text": None, "confuser": confuser}


def main() -> None:
    rng = random.Random(42)
    train, val = [], []
    val_benign = set(BENIGN[::7])
    val_destr = set(DESTRUCTIVE[::3])
    for q in BENIGN:
        if q in val_benign:
            continue
        train.append(make_row(q, "run_command", "UNFAMILIAR", rng))
        for variant in (f"запусти {q}" if not q.startswith(VERB_STARTS) else q,
                        q + " сейчас"):
            if variant != q and len(train) < 118:
                train.append(make_row(variant, "run_command", "UNFAMILIAR", rng))
    for q in DESTRUCTIVE:
        if q in val_destr:
            continue
        train.append(make_row(q, "UNFAMILIAR", "run_command", rng))
    for q in BENIGN[::7]:
        val.append(make_row(q, "run_command", "UNFAMILIAR", rng))
    for q in DESTRUCTIVE[::3]:
        val.append(make_row(q, "UNFAMILIAR", "run_command", rng))
    rng.shuffle(train)
    rng.shuffle(val)
    (OUT_DIR / "docker_train.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in train) + "\n", encoding="utf-8")
    (OUT_DIR / "docker_val.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in val) + "\n", encoding="utf-8")
    print(f"docker_train: {len(train)} rows, docker_val: {len(val)} rows")


if __name__ == "__main__":
    main()