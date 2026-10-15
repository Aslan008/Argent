# -*- coding: utf-8 -*-
"""scripts/_e5_quant_experiment11.py — вынос с маской по ПАРАМ токенов.

Диагноз: канал 957 слоя 12 выстреливает на парах ('/' -> 'ms') и (' ' -> '2')
(19/20 текстов). exp10 с маской «id ∈ {пробел, слэш}» покрыл ВСЕ пробелы/слэши
— константа попала на лишние токены, fp32 упал до 56.9%.

Фикс: маска по паре (текущий id, следующий id). В ONNX: Equal(input_ids, cur)
И Equal(Shift(input_ids), nxt) — сдвиг реализуем через Slice(input_ids, [1], ...)
+ Concat(pad). Собираем пары по 20 текстам динамически.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import numpy_helper
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModelForMaskedLM, AutoTokenizer

BASE_DIR = Path(".argent/models/mmbert-argent")
FP32_PATH = BASE_DIR / "mmbert_route.onnx"
BIAS_PATH = BASE_DIR / "mmbert_route_bias4.onnx"
BIAS_INT8_PATH = BASE_DIR / "mmbert_route_bias4_int8.onnx"
BATTERY_PATH = Path("scripts/prod_battery.json")
CLASSES_PATH = Path("src/leeth/data_400m/choice_classes.json")
TEMPLATE_Q = "Какой инструмент Argent должен быть вызван в первую очередь?"

# слой 11: канал 924, маска <bos>; слой 12: канал 957, маска по парам
SPECS = [
    (11, 924, 16709.5, "bos"),
    (12, 957, -2570.5, "pairs"),
]


def build_state(query: str) -> str:
    normalized = query[:1].lower() + query[1:] if query[:1].isalpha() else query
    return f'Incoming user request in Argent session:\n"{normalized}"\nWorking directory: C:/Users/user/Desktop/Argent'


def collect_pairs() -> list:
    """Пары (текущий id, следующий id), на которых канал 957 слоя 12 выстреливает."""
    base = AutoModelForMaskedLM.from_pretrained(str(BASE_DIR), dtype=torch.float32).eval()
    bm = base.base_model
    tok = AutoTokenizer.from_pretrained(str(BASE_DIR))
    battery = json.loads(BATTERY_PATH.read_text(encoding="utf-8"))
    texts = [build_state(c["query"]) + "\n[QUESTION] " + TEMPLATE_Q for c in battery["cases"][:20]]
    pairs = set()
    with torch.no_grad():
        for t in texts:
            enc = tok(t, truncation=True, max_length=256, return_tensors="pt")
            ids = enc["input_ids"][0].tolist()
            h = bm.embeddings(enc["input_ids"])
            pos_ids = torch.arange(enc["input_ids"].shape[1]).unsqueeze(0)
            pos = bm.rotary_emb(h, position_ids=pos_ids, layer_type="full_attention")
            for i in range(12):
                layer = bm.layers[i]
                attn_out = layer.attn(layer.attn_norm(h), position_embeddings=pos,
                                      attention_mask=enc["attention_mask"].bool())
                if isinstance(attn_out, tuple):
                    attn_out = attn_out[0]
                h2 = h + attn_out
                h = h2 + layer.mlp(layer.mlp_norm(h2))
            layer = bm.layers[12]
            attn_out = layer.attn(layer.attn_norm(h), position_embeddings=pos,
                                  attention_mask=enc["attention_mask"].bool())
            if isinstance(attn_out, tuple):
                attn_out = attn_out[0]
            h2 = h + attn_out
            hn = layer.mlp_norm(h2)
            ff = layer.mlp.Wi(hn)
            inp, gate = ff.chunk(2, dim=-1)
            act = layer.mlp.act(inp) * gate
            a957 = act[0, :, 957]
            fire = (a957.abs() > 100).nonzero().flatten().tolist()
            for f in fire:
                nxt = ids[f + 1] if f + 1 < len(ids) else -1
                pairs.add((int(ids[f]), int(nxt)))
    return sorted(pairs)


def battery_acc(sess) -> tuple:
    tokenizer = AutoTokenizer.from_pretrained(str(BASE_DIR))
    battery = json.loads(BATTERY_PATH.read_text(encoding="utf-8"))
    class_to_idx = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    correct = 0
    t0 = time.perf_counter()
    for c in battery["cases"]:
        t = build_state(c["query"]) + "\n[QUESTION] " + TEMPLATE_Q
        enc = tokenizer(t, truncation=True, max_length=256, return_tensors="pt")
        logits = sess.run(None, {"input_ids": enc["input_ids"].numpy(),
                                 "attention_mask": enc["attention_mask"].numpy()})[0][0]
        correct += int(idx_to_class[int(logits.argmax())] == c["expected_tool"])
    wall = time.perf_counter() - t0
    n = len(battery["cases"])
    return correct, n, wall / n * 1000


def main() -> None:
    tok = AutoTokenizer.from_pretrained(str(BASE_DIR))
    pairs = collect_pairs()
    print(f"пары слоя 12: {[(tok.decode([c]), tok.decode([n]) if n >= 0 else '<end>') for c, n in pairs]}")

    model = onnx.load(str(FP32_PATH))
    g = model.graph
    inits = {i.name: i for i in g.initializer}
    added_inits: set = set()
    nodes = {n.name: n for n in g.node}

    for layer, act_ch, mean_act, mask_kind in SPECS:
        wo_node = nodes[f"/base/layers.{layer}/mlp/Wo/MatMul"]
        wo_w = inits[wo_node.input[1]]
        wo_arr = numpy_helper.to_array(wo_w).copy()
        const_vec = wo_arr[act_ch, :] * mean_act
        wo_arr[act_ch, :] = 0.0
        inits[wo_w.name].CopyFrom(numpy_helper.from_array(wo_arr, wo_w.name))
        const_name = f"/base/layers.{layer}/mlp/Wo/bias_const"
        g.initializer.append(numpy_helper.from_array(const_vec.astype(np.float32), const_name))

        # строим маску
        eq_nodes = []
        if mask_kind == "bos":
            bos_id = tok.bos_token_id
            g.initializer.append(numpy_helper.from_array(np.array([bos_id], dtype=np.int64),
                                                         f"/base/layers.{layer}/mlp/Wo/mask_id"))
            eq_nodes.append(onnx.helper.make_node(
                "Equal", ["input_ids", f"/base/layers.{layer}/mlp/Wo/mask_id"],
                [f"/base/layers.{layer}/mlp/Wo/eq_0"], name=f"/base/layers.{layer}/mlp/Wo/eq_0"))
            or_nodes = []
            mask_bool = f"/base/layers.{layer}/mlp/Wo/eq_0"
        else:
            # маска по парам: Equal(input_ids, cur) AND Equal(shift(input_ids), nxt)
            # shift: input_ids[:, 1:] -> Pad слева -1 -> input_ids_shifted
            and_nodes = []
            for j, (cur, nxt) in enumerate(pairs):
                cur_name = f"/base/layers.{layer}/mlp/Wo/cur_{j}"
                nxt_name = f"/base/layers.{layer}/mlp/Wo/nxt_{j}"
                g.initializer.append(numpy_helper.from_array(np.array([cur], dtype=np.int64), cur_name))
                g.initializer.append(numpy_helper.from_array(np.array([nxt], dtype=np.int64), nxt_name))
                eq_cur = onnx.helper.make_node("Equal", ["input_ids", cur_name],
                                               [f"/base/layers.{layer}/mlp/Wo/eqc_{j}"],
                                               name=f"/base/layers.{layer}/mlp/Wo/eqc_{j}")
                # сдвиг: Slice(input_ids, starts=[1], ends=[max], axes=[1]) -> Concat(-1 pad)
                # (starts/ends/axes_s/padval/pad — общие инициализаторы, добавляются один раз)
                starts_name = f"/base/layers.{layer}/mlp/Wo/starts"
                ends_name = f"/base/layers.{layer}/mlp/Wo/ends"
                axes_s_name = f"/base/layers.{layer}/mlp/Wo/axes_s"
                padval_name = f"/base/layers.{layer}/mlp/Wo/padval"
                pad_name = f"/base/layers.{layer}/mlp/Wo/pad"
                if starts_name not in inits and starts_name not in added_inits:
                    added_inits.update([starts_name, ends_name, axes_s_name, padval_name, pad_name])
                    g.initializer.append(numpy_helper.from_array(np.array([1], dtype=np.int64), starts_name))
                    g.initializer.append(numpy_helper.from_array(np.array([np.iinfo(np.int64).max], dtype=np.int64), ends_name))
                    g.initializer.append(numpy_helper.from_array(np.array([1], dtype=np.int64), axes_s_name))
                    g.initializer.append(numpy_helper.from_array(np.array([-1], dtype=np.int64), padval_name))
                    g.initializer.append(numpy_helper.from_array(np.array([1, 0], dtype=np.int64), pad_name))
                    # padval_u: [[-1]] формы (1,1) — для Concat по оси 1 (ранг 2, broadcast)
                    g.initializer.append(numpy_helper.from_array(np.array([[-1]], dtype=np.int64),
                                                                 f"/base/layers.{layer}/mlp/Wo/padval_u"))
                slice_node = onnx.helper.make_node("Slice", ["input_ids", starts_name, ends_name, axes_s_name],
                                                   [f"/base/layers.{layer}/mlp/Wo/shifted_{j}"],
                                                   name=f"/base/layers.{layer}/mlp/Wo/slice_{j}")
                concat_node = onnx.helper.make_node("Concat",
                                                    [f"/base/layers.{layer}/mlp/Wo/shifted_{j}",
                                                     f"/base/layers.{layer}/mlp/Wo/padval_u"],
                                                    [f"/base/layers.{layer}/mlp/Wo/shifted_pad_{j}"],
                                                    axis=1, name=f"/base/layers.{layer}/mlp/Wo/concat_{j}")
                eq_nxt = onnx.helper.make_node("Equal", [f"/base/layers.{layer}/mlp/Wo/shifted_pad_{j}", nxt_name],
                                               [f"/base/layers.{layer}/mlp/Wo/eqn_{j}"],
                                               name=f"/base/layers.{layer}/mlp/Wo/eqn_{j}")
                and_node = onnx.helper.make_node("And", [f"/base/layers.{layer}/mlp/Wo/eqc_{j}",
                                                         f"/base/layers.{layer}/mlp/Wo/eqn_{j}"],
                                                 [f"/base/layers.{layer}/mlp/Wo/and_{j}"],
                                                 name=f"/base/layers.{layer}/mlp/Wo/and_{j}")
                eq_nodes.extend([eq_cur, slice_node, concat_node, eq_nxt])
                and_nodes.append(and_node)
            # Or по парам
            or_nodes = []
            acc_name = f"/base/layers.{layer}/mlp/Wo/and_0"
            for j in range(1, len(pairs)):
                or_name = f"/base/layers.{layer}/mlp/Wo/or_{j}"
                or_nodes.append(onnx.helper.make_node("Or", [acc_name, f"/base/layers.{layer}/mlp/Wo/and_{j}"],
                                                      [or_name], name=f"/base/layers.{layer}/mlp/Wo/or_{j}"))
                acc_name = or_name
            mask_bool = acc_name
        cast_f = onnx.helper.make_node("Cast", [mask_bool],
                                       [f"/base/layers.{layer}/mlp/Wo/mask_f"], to=onnx.TensorProto.FLOAT,
                                       name=f"/base/layers.{layer}/mlp/Wo/mask_cast")
        g.initializer.append(numpy_helper.from_array(np.array([2], dtype=np.int64),
                                                     f"/base/layers.{layer}/mlp/Wo/axes_u"))
        unsq = onnx.helper.make_node("Unsqueeze", [f"/base/layers.{layer}/mlp/Wo/mask_f",
                                                   f"/base/layers.{layer}/mlp/Wo/axes_u"],
                                     [f"/base/layers.{layer}/mlp/Wo/mask_u"],
                                     name=f"/base/layers.{layer}/mlp/Wo/mask_unsq")
        mul_mask = onnx.helper.make_node("Mul", [f"/base/layers.{layer}/mlp/Wo/mask_u", const_name],
                                         [f"/base/layers.{layer}/mlp/Wo/bias_term"],
                                         name=f"/base/layers.{layer}/mlp/Wo/bias_mul")
        add = onnx.helper.make_node("Add", [wo_node.output[0], f"/base/layers.{layer}/mlp/Wo/bias_term"],
                                    [wo_node.output[0] + "/bias_out"], name=f"/base/layers.{layer}/mlp/Wo/bias_add")
        # вставляем ВСЮ цепочку перед потребителем Wo
        consumer_idx = min(i for i, n2 in enumerate(g.node)
                           if wo_node.output[0] in n2.input and n2.name != wo_node.name)
        chain = [eq_nodes, and_nodes if mask_kind == "pairs" else [], or_nodes, cast_f, unsq, mul_mask, add]
        flat = [x for grp in chain for x in (grp if isinstance(grp, list) else [grp])]
        for node in flat[::-1]:
            g.node.insert(consumer_idx, node)
        wo_out = wo_node.output[0]
        for n2 in g.node:
            for k, inp in enumerate(n2.input):
                if inp == wo_out and n2.name != wo_node.name and n2.name != add.name:
                    n2.input[k] = wo_out + "/bias_out"
    onnx.checker.check_model(model, full_check=False)
    onnx.save(model, str(BIAS_PATH))
    print(f"модель сохранена: {BIAS_PATH}")

    sess = ort.InferenceSession(str(BIAS_PATH), providers=["CPUExecutionProvider"])
    ok, n, ms = battery_acc(sess)
    print(f"[маскированный fp32] батарея: {ok}/{n} = {ok / n:.1%} | {ms:.0f} мс/кейс")

    quantize_dynamic(str(BIAS_PATH), str(BIAS_INT8_PATH), weight_type=QuantType.QInt8, per_channel=True)
    print(f"[маскированный int8] size={BIAS_INT8_PATH.stat().st_size / (1024 * 1024):.0f} MB")
    sess_q = ort.InferenceSession(str(BIAS_INT8_PATH), providers=["CPUExecutionProvider"])
    ok, n, ms = battery_acc(sess_q)
    print(f"[маскированный int8] батарея: {ok}/{n} = {ok / n:.1%} | {ms:.0f} мс/кейс")


if __name__ == "__main__":
    main()