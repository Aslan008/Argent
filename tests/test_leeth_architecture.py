"""
tests/test_leeth_architecture.py — Verification tests for Leeth neural model architecture and decision heads.
"""

import pytest
import torch
from src.leeth.model import LeethConfig, LeethModel


def test_leeth_model_shapes():
    config = LeethConfig(
        vocab_size=1000,
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        intermediate_size=128,
        max_position_embeddings=512,
        num_choices=10,
    )
    model = LeethModel(config)
    model.eval()

    bsz = 2
    seq_len = 16
    input_ids = torch.randint(0, 1000, (bsz, seq_len))
    attention_mask = torch.ones((bsz, seq_len), dtype=torch.long)
    attention_mask[:, 12:] = 0  # 4 tokens of padding

    with torch.no_grad():
        out = model(input_ids, attention_mask=attention_mask)

    assert "choice_logits" in out
    assert "noul_probs" in out
    assert "score_logits" in out
    assert "start_logits" in out
    assert "end_logits" in out
    assert "mlm_logits" in out

    # Check tensor shapes
    assert out["choice_logits"].shape == (bsz, 10)
    assert out["noul_probs"].shape == (bsz,)
    assert out["score_logits"].shape == (bsz, 10)  # ordinal rubric logits
    assert out["start_logits"].shape == (bsz, seq_len)
    assert out["end_logits"].shape == (bsz, seq_len)
    assert out["mlm_logits"].shape == (bsz, seq_len, 1000)

    # Check bounds
    assert torch.all((out["noul_probs"] >= 0.0) & (out["noul_probs"] <= 1.0))
    probs = torch.softmax(out["score_logits"], dim=-1)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(bsz), atol=1e-5)


def test_leeth_model_backward_pass():
    config = LeethConfig(
        vocab_size=500,
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        intermediate_size=128,
        num_choices=5,
    )
    model = LeethModel(config)
    model.train()

    bsz = 4
    seq_len = 20
    input_ids = torch.randint(0, 500, (bsz, seq_len))
    mask = torch.ones((bsz, seq_len), dtype=torch.long)

    out = model(input_ids, attention_mask=mask)

    # Multi-task targets
    choice_target = torch.tensor([0, 2, 4, 1])
    noul_target = torch.tensor([1.0, 0.0, 1.0, 0.0])
    score_target = torch.tensor([80.0, 20.0, 95.0, 10.0])
    score_levels = torch.tensor([5, 5, 5, 5])
    start_target = torch.tensor([2, 5, 1, 3])
    end_target = torch.tensor([4, 8, 3, 6])

    # Multi-task loss
    loss_choice = torch.nn.functional.cross_entropy(out["choice_logits"], choice_target)
    loss_noul = torch.nn.functional.binary_cross_entropy(out["noul_probs"], noul_target)
    level_targets = torch.clamp(torch.ceil(score_target / 100.0 * 5), min=1, max=10).long() - 1
    arange = torch.arange(out["score_logits"].size(-1)).unsqueeze(0)
    masked_score = out["score_logits"].masked_fill(arange >= score_levels.unsqueeze(1), -1e9)
    loss_score = torch.nn.functional.cross_entropy(masked_score, level_targets)
    loss_start = torch.nn.functional.cross_entropy(out["start_logits"], start_target)
    loss_end = torch.nn.functional.cross_entropy(out["end_logits"], end_target)

    # Calibration Brier score loss
    brier_loss = torch.mean((out["noul_probs"] - noul_target) ** 2)

    total_loss = loss_choice + loss_noul + 0.5 * brier_loss + loss_score + loss_start + loss_end
    total_loss.backward()

    # Verify gradients flowed to all multi-task modules
    # (dynamic_opt_head participates only when opt_positions are passed —
    # this test exercises the legacy heads without [OPT] markers)
    for name, param in model.named_parameters():
        if param.requires_grad and not name.startswith("lm_head") and not name.startswith("dynamic_opt_head"):
            assert param.grad is not None, f"Gradient missing for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN in grad for {name}"

    # Verify MLM head backward pass
    mlm_logits = model.forward_mlm(input_ids, attention_mask=mask)
    mlm_loss = torch.nn.functional.cross_entropy(mlm_logits.view(-1, 500), input_ids.view(-1))
    mlm_loss.backward()
    assert model.lm_head[0].weight.grad is not None


def test_cuda_execution_if_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    config = LeethConfig(
        vocab_size=1000,
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        intermediate_size=128,
        num_choices=8,
    )
    model = LeethModel(config).to("cuda")
    input_ids = torch.randint(0, 1000, (2, 16), device="cuda")
    out = model(input_ids)
    assert out["choice_logits"].is_cuda
    assert out["noul_probs"].is_cuda


class TestCheckpointConfigCompatibility:
    """An old checkpoint must be rebuilt with the rotary base it was trained with, not the new default."""

    def test_a_config_without_rope_base_is_a_legacy_one(self):
        from src.leeth.model import LEGACY_ROPE_BASE, LeethConfig, config_from_checkpoint

        old = LeethConfig(hidden_size=64, num_heads=4, num_layers=1)
        del old.__dict__["rope_base"]            # what a pickle saved before the field existed looks like
        assert old.rope_base != LEGACY_ROPE_BASE  # reading it now falls through to the NEW class default
        assert config_from_checkpoint(old).rope_base == LEGACY_ROPE_BASE
        assert config_from_checkpoint({"hidden_size": 64, "num_heads": 4, "num_layers": 1}).rope_base == LEGACY_ROPE_BASE

    def test_a_saved_rope_base_wins_and_unknown_keys_are_ignored(self):
        from src.leeth.model import LeethConfig, config_from_checkpoint

        cfg = config_from_checkpoint({"hidden_size": 64, "num_heads": 4, "num_layers": 1,
                                      "rope_base": 100000.0, "field_from_the_future": 1})
        assert cfg.rope_base == 100000.0 and isinstance(cfg, LeethConfig)

    def test_the_rotary_angles_follow_the_legacy_base(self):
        import torch
        from src.leeth.model import LEGACY_ROPE_BASE, LeethConfig, LeethModel, config_from_checkpoint

        old = LeethConfig(hidden_size=64, num_heads=4, num_layers=1, vocab_size=50)
        del old.__dict__["rope_base"]
        legacy = LeethModel(config_from_checkpoint(old)).rotary_emb.inv_freq
        expected = 1.0 / (LEGACY_ROPE_BASE ** (torch.arange(0, 16, 2).float() / 16))
        assert torch.allclose(legacy, expected)
