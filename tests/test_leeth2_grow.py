"""Growing a Leeth 2 checkpoint must not change what it computes -- and must leave the new capacity trainable."""

import json
import shutil
from pathlib import Path

import pytest
import torch

from src.leeth2 import grow
from src.leeth2.model import Leeth2Config, Leeth2Model

VOCAB = 600


def tiny_backbone(layers: int = 9, hidden: int = 32, inter: int = 48) -> dict:
    from transformers import ModernBertConfig

    cfg = ModernBertConfig(
        vocab_size=VOCAB, hidden_size=hidden, intermediate_size=inter, num_hidden_layers=layers, num_attention_heads=4,
        global_attn_every_n_layers=3, local_attention=16, max_position_embeddings=256,
        pad_token_id=0, eos_token_id=1, bos_token_id=2, cls_token_id=2, sep_token_id=1,
    )
    return cfg.to_dict()


@pytest.fixture
def small(tmp_path) -> Path:
    torch.manual_seed(3)
    cfg = Leeth2Config(backbone=tiny_backbone(), vocab_size=VOCAB, hidden_size=32, freeze_embeddings=False)
    model = Leeth2Model(cfg)
    # random but non-degenerate everywhere, so an identity-by-accident cannot hide a bug
    for p in model.parameters():
        p.data.normal_(0, 0.05)
    model.save_pretrained(tmp_path / "small")
    return tmp_path / "small"


def test_block_plan_rules():
    assert grow.plan_blocks(22, [5, 11, 17, 20]) == [5, 11, 17, 20]
    with pytest.raises(ValueError):
        grow.plan_blocks(22, [4])           # would shift the global/local pattern
    with pytest.raises(ValueError):
        grow.plan_blocks(22, [2])           # would copy layer 0, which has no attention norm
    order = grow.layer_order(9, [5])
    assert [s for s, _ in order] == [0, 1, 2, 3, 4, 5, 3, 4, 5, 6, 7, 8]
    assert [c for _, c in order] == [False] * 6 + [True] * 3 + [False] * 3
    # every layer keeps its attention type: the 3-periodic pattern is intact after the insertion
    assert [(i % 3 == 0) for i in range(len(order))] == [(s % 3 == 0) for s, _ in order]


@pytest.mark.parametrize("blocks,mult", [([5], 1.0), ([2 + 3, 8], 1.0), ([5], 2.0), ([5, 8], 1.5)])
def test_growth_changes_no_output(small, tmp_path, blocks, mult):
    info = grow.grow_checkpoint(small, tmp_path / "grown", blocks, mlp_mult=mult)
    assert info["layers"] == 9 + 3 * len(blocks) and info["intermediate"] == round(48 * mult)
    diff = grow.verify_equivalent(small, tmp_path / "grown", seq=64)
    assert diff and max(diff.values()) < 1e-6, diff


def test_is_grown_tells_a_grown_checkpoint_from_an_ordinary_one(small, tmp_path):
    grow.grow_checkpoint(small, tmp_path / "g", [5], mlp_mult=1.0)
    assert grow.is_grown(tmp_path / "g") and not grow.is_grown(small) and not grow.is_grown(tmp_path / "nothing-here")
    (tmp_path / "run").mkdir()
    shutil.copytree(tmp_path / "g", tmp_path / "run" / "final")
    assert grow.is_grown(tmp_path / "run")                            # a run dir that holds final/


def test_training_a_grown_start_model_needs_its_own_architecture(small, tmp_path):
    """Loading grown weights into the original network must not pass silently: the shapes differ, and load_init refuses."""
    from src.leeth2 import train as T

    grow.grow_checkpoint(small, tmp_path / "g", [5], mlp_mult=2.0)
    original = Leeth2Model.from_pretrained(small)
    with pytest.raises(RuntimeError):
        T.load_init(original, tmp_path / "g")
    rebuilt = Leeth2Model.from_arch(tmp_path / "g")
    assert "tensors" in T.load_init(rebuilt, tmp_path / "g")


def test_grown_model_is_bigger_and_config_is_consistent(small, tmp_path):
    grow.grow_checkpoint(small, tmp_path / "g", [5, 8], mlp_mult=2.0)
    big = Leeth2Model.from_pretrained(tmp_path / "g")
    base = Leeth2Model.from_pretrained(small)
    assert big.count_parameters()["total"] > base.count_parameters()["total"] * 1.5
    cfg = json.loads((tmp_path / "g" / "leeth2_config.json").read_text(encoding="utf-8"))
    assert cfg["backbone"]["num_hidden_layers"] == 15 and cfg["backbone"]["intermediate_size"] == 96
    assert len(big.backbone.layers) == 15
    types = cfg["backbone"].get("layer_types")
    if types:
        assert types == ["full_attention" if i % 3 == 0 else "sliding_attention" for i in range(15)]


def test_new_capacity_is_trainable(small, tmp_path):
    """The zero projections must receive gradient (otherwise the copies would stay identities forever)."""
    grow.grow_checkpoint(small, tmp_path / "g", [5], mlp_mult=2.0)
    m = Leeth2Model.from_pretrained(tmp_path / "g").train()
    n, L = 3, 48
    ids = torch.randint(5, 500, (n, L))
    mask = torch.ones(n, L, dtype=torch.long)
    opt = torch.zeros(n, L, dtype=torch.long)
    opt[:, 10:15], opt[:, 20:25] = 1, 2
    out = m(input_ids=ids, attention_mask=mask, opt_index=opt, rd_pos=torch.full((n,), 3), state_mask=(opt == 0), n_opts=torch.full((n,), 2), k_max=2)
    (out["choice_logits"].sum() + out["noul_logit"].sum()).backward()
    new_layer = 6                        # the first copied layer
    assert m.backbone.layers[new_layer].attn.Wo.weight.grad.abs().sum() > 0
    assert m.backbone.layers[new_layer].mlp.Wo.weight.grad.abs().sum() > 0
    wo = m.backbone.layers[0].mlp.Wo.weight.grad                       # the widened part of an original layer
    assert wo[:, 48:].abs().sum() > 0


def test_a_few_training_steps_do_not_blow_up(small, tmp_path):
    grow.grow_checkpoint(small, tmp_path / "g", [5], mlp_mult=2.0)
    m = Leeth2Model.from_pretrained(tmp_path / "g").train()
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-3)
    ids = torch.randint(5, 500, (4, 40))
    mask = torch.ones(4, 40, dtype=torch.long)
    oi = torch.zeros(4, 40, dtype=torch.long)
    oi[:, 10:15], oi[:, 20:25] = 1, 2
    losses = []
    for _ in range(25):
        out = m(input_ids=ids, attention_mask=mask, opt_index=oi, rd_pos=torch.full((4,), 3), state_mask=(oi == 0), n_opts=torch.full((4,), 2), k_max=2)
        loss = torch.nn.functional.cross_entropy(out["choice_logits"], torch.tensor([0, 1, 0, 1]))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss))
    assert all(map(lambda x: x == x, losses)) and losses[-1] < losses[0]


def test_real_mmbert_geometry_lands_where_planned():
    """The numbers the Kaggle run relies on: 22 layers / 1152 -> 34 layers / 2304 is ~457M parameters, under the 500M ceiling."""
    layers, inter, hidden = 34, 2304, 768
    per_layer = (3 * hidden * hidden + hidden * hidden) + 2 * hidden * inter * 1 + inter * hidden     # qkv+o, Wi (2I), Wo
    per_layer += 2 * hidden                                                                          # norms
    total = 256000 * hidden + layers * per_layer + 7_000_000
    assert 440e6 < total < 500e6, total


def test_trainer_can_start_from_a_grown_model(small, tmp_path):
    """arch_from builds the grown network, load_init fills it from the grown weights: same function as the small model."""
    from src.leeth2.train import load_init

    grow.grow_checkpoint(small, tmp_path / "g", [5], mlp_mult=2.0)
    model = Leeth2Model.from_arch(tmp_path / "g")
    assert len(model.backbone.layers) == 12
    what = load_init(model, tmp_path / "g")
    assert "model.safetensors" in what
    ref = Leeth2Model.from_pretrained(small).eval()
    model.eval()
    ids = torch.randint(5, 500, (2, 40))
    mask = torch.ones(2, 40, dtype=torch.long)
    oi = torch.zeros(2, 40, dtype=torch.long)
    oi[:, 10:15], oi[:, 20:25] = 1, 2
    kw = dict(input_ids=ids, attention_mask=mask, opt_index=oi, rd_pos=torch.full((2,), 3), state_mask=(oi == 0), n_opts=torch.full((2,), 2), k_max=2)
    with torch.no_grad():
        a, b = ref(**kw), model(**kw)
    assert (a["choice_logits"] - b["choice_logits"]).abs().max() < 1e-6


def test_new_layers_get_their_own_learning_rate(small, tmp_path):
    from src.leeth2.train import TrainConfig, build_param_groups

    grow.grow_checkpoint(small, tmp_path / "g", [5], mlp_mult=1.0)
    model = Leeth2Model.from_arch(tmp_path / "g")
    assert model.cfg.grown["new_layers"] == [6, 7, 8]
    cfg = TrainConfig(lr=1e-4, llrd=1.0, new_lr_mult=4.0)
    lr_of = {}
    for g in build_param_groups(model, cfg):
        for p in g["params"]:
            lr_of[id(p)] = g["lr"]
    names = dict(model.named_parameters())
    assert lr_of[id(names["backbone.layers.7.mlp.Wi.weight"])] == pytest.approx(4e-4)
    assert lr_of[id(names["backbone.layers.2.mlp.Wi.weight"])] == pytest.approx(1e-4)
    assert lr_of[id(names["backbone.layers.9.mlp.Wi.weight"])] == pytest.approx(1e-4)       # the originals after the block
