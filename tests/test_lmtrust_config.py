"""Blind-spot tests for config.py — edge cases that could silently corrupt
behaviour: corrupted JSON quarantine, wrong-shape JSON, model size classification
boundary checks, override validation, and cache correctness."""

import json
import pytest

import config


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _reset_config_cache():
    """Clear _CONFIG_CACHE before and after every test so state never leaks."""
    config._CONFIG_CACHE = None
    yield
    config._CONFIG_CACHE = None


@pytest.fixture
def patched_config_file(tmp_path, monkeypatch):
    """Point config.CONFIG_FILE to a tmp_path file, clear cache, return path."""
    cfg_path = tmp_path / "test_config.json"
    monkeypatch.setattr(config, "CONFIG_FILE", cfg_path)
    config._CONFIG_CACHE = None
    return cfg_path


# ─── load_config: corrupted JSON ─────────────────────────────────────────────

class TestLoadConfigCorruptedJson:
    def test_corrupted_json_quarantines_and_returns_default(self, patched_config_file):
        """A corrupted JSON file is quarantined (moved aside) and load_config
        falls back to the default config instead of silently swallowing it."""
        patched_config_file.write_text("{invalid json}", encoding="utf-8")

        result = config.load_config()

        assert result == {"model": config.DEFAULT_MODEL}
        # Original file should have been moved aside by quarantine_config
        assert not patched_config_file.exists()
        # A .broken-* backup should exist in the same directory
        backups = list(patched_config_file.parent.glob("test_config.json.broken-*"))
        assert len(backups) == 1


# ─── load_config: valid JSON, wrong shape ────────────────────────────────────

class TestLoadConfigWrongShape:
    def test_list_instead_of_dict_quarantines_and_returns_default(self, patched_config_file):
        """Valid JSON that is a list (not a dict) is quarantined just like
        broken JSON — otherwise every _get() after it would raise AttributeError."""
        patched_config_file.write_text("[1, 2, 3]", encoding="utf-8")

        result = config.load_config()

        assert result == {"model": config.DEFAULT_MODEL}
        assert not patched_config_file.exists()
        backups = list(patched_config_file.parent.glob("test_config.json.broken-*"))
        assert len(backups) == 1


# ─── get_model_size_category: edge cases ─────────────────────────────────────

class TestGetModelSizeCategory:
    """The size classifier drives prompt template selection, toolset width,
    and history budget. A wrong classification silently cripples the model."""

    @pytest.fixture(autouse=True)
    def _no_override(self, patched_config_file):
        """Ensure no model_category_override leaks in from a cached config."""
        config._CONFIG_CACHE = None

    # — Empty / None —
    def test_empty_string_returns_medium(self):
        assert config.get_model_size_category("") == "medium"

    def test_none_returns_medium(self):
        assert config.get_model_size_category(None) == "medium"

    # — Keyword boundary checks —
    def test_minimax_m3_not_tiny(self):
        """'mini' inside 'minimax' must NOT match the tiny keyword — minimax-m3
        is a large cloud model, and classifying it as tiny would cripple it."""
        result = config.get_model_size_category("minimax-m3")
        assert result != "tiny"

    def test_phi_3_mini_is_tiny(self):
        """'mini' on a token boundary (after '-') IS tiny — phi-3-mini is small."""
        assert config.get_model_size_category("phi-3-mini") == "tiny"

    # — MoE active params —
    def test_lfm_moe_classified_by_active_params(self):
        """LFM2.5-8B-A1B has 8B total but only 1B active params (A1B suffix).
        Must be classified by the 1B active count → tiny, not by 8B → medium."""
        assert config.get_model_size_category("LFM2.5-8B-A1B") == "tiny"

    # — Size-based classification —
    def test_qwen_1_5b_is_tiny(self):
        assert config.get_model_size_category("qwen2.5-1.5b") == "tiny"

    def test_something_2b_is_tiny(self):
        assert config.get_model_size_category("something-2b") == "tiny"

    def test_something_3b_is_small(self):
        assert config.get_model_size_category("something-3b") == "small"

    def test_something_7b_is_medium(self):
        assert config.get_model_size_category("something-7b") == "medium"

    def test_something_13b_is_large(self):
        assert config.get_model_size_category("something-13b") == "large"

    def test_something_14b_is_large(self):
        assert config.get_model_size_category("something-14b") == "large"

    # — Cloud detection —
    def test_gpt_4o_is_cloud(self):
        assert config.get_model_size_category("gpt-4o") == "cloud"

    def test_claude_3_opus_is_cloud(self):
        assert config.get_model_size_category("claude-3-opus") == "cloud"

    def test_gemini_1_5_pro_is_cloud(self):
        assert config.get_model_size_category("gemini-1.5-pro") == "cloud"

    def test_model_cloud_tag_is_cloud(self):
        """The explicit ':cloud' tag (used by aggregators) must be detected."""
        assert config.get_model_size_category("model:cloud") == "cloud"


# ─── set_model_category_override ────────────────────────────────────────────

class TestSetModelCategoryOverride:
    def test_invalid_category_raises_value_error(self, patched_config_file):
        with pytest.raises(ValueError, match="Invalid category"):
            config.set_model_category_override("invalid_category")

    def test_none_clears_override(self, patched_config_file):
        """Passing None removes the override so auto-detection resumes."""
        # Set an override
        config.set_model_category_override("cloud")
        assert config.get_model_category_override() == "cloud"

        # Clear it
        config.set_model_category_override(None)
        assert config.get_model_category_override() is None


# ─── _CONFIG_CACHE ───────────────────────────────────────────────────────────

class TestConfigCache:
    def test_second_load_does_not_re_read_disk(self, patched_config_file):
        """Once loaded, _CONFIG_CACHE is returned on subsequent calls without
        touching the file — modifying the file on disk must not change the
        returned config."""
        initial = {"model": "test-model", "version": 1}
        patched_config_file.write_text(json.dumps(initial), encoding="utf-8")

        # First call reads disk and populates cache
        result1 = config.load_config()
        assert result1 == initial

        # Modify the file on disk
        modified = {"model": "different-model", "version": 2}
        patched_config_file.write_text(json.dumps(modified), encoding="utf-8")

        # Second call must return the cached value, not re-read disk
        result2 = config.load_config()
        assert result2 == initial
        assert result2["version"] == 1