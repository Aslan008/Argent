"""The Claude model catalogue and what the provider does with it: live list, saved copy, request parameters."""

import json
import time
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import claude_catalog
import providers
from claude_catalog import Catalog, ModelEntry, entry_from_api, load, thinking_style
from providers import ClaudeProvider, ProviderError


def info(mid, year=2025, month=1, thinking=None, effort=None, max_in=200000, max_out=64000):
    """What the SDK's ModelInfo looks like (real capability objects are plain attribute bags here)."""
    caps = None
    if thinking is not None or effort is not None:
        sup = lambda v: SimpleNamespace(supported=v)
        caps = SimpleNamespace(
            thinking=SimpleNamespace(supported=bool(thinking), types=SimpleNamespace(
                enabled=sup("enabled" in (thinking or ())), adaptive=sup("adaptive" in (thinking or ())))),
            effort=SimpleNamespace(supported=bool(effort), low=sup("low" in (effort or ())),
                                   medium=sup("medium" in (effort or ())), high=sup("high" in (effort or ())),
                                   max=sup("max" in (effort or ())), xhigh=sup("xhigh" in (effort or ()))))
    return SimpleNamespace(id=mid, display_name=mid, created_at=datetime(year, month, 1, tzinfo=timezone.utc),
                           max_input_tokens=max_in, max_tokens=max_out, capabilities=caps)


class Pages:
    """A client whose models.list yields the given items across `pages` pages, like the SDK's SyncPage."""

    def __init__(self, items):
        self.calls = []
        client = MagicMock()

        def list_(**kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(auto_paging_iter=lambda: iter(items))
        client.models.list.side_effect = list_
        self.client = client


class TestEntries:
    def test_reads_what_the_api_says_about_a_model(self):
        e = entry_from_api(info("claude-opus-5-5", 2026, 3, thinking=("adaptive",), effort=("low", "high", "max")))
        assert e.id == "claude-opus-5-5" and e.max_input == 200000 and e.max_output == 64000
        assert e.thinking_adaptive is True and e.thinking_enabled is False
        assert e.effort_levels == ("low", "high", "max")
        assert e.created == datetime(2026, 3, 1, tzinfo=timezone.utc).timestamp()

    def test_a_model_without_capability_data_has_unknowns_not_falsehoods(self):
        e = entry_from_api(info("claude-old"))
        assert e.thinking_enabled is None and e.effort_levels is None and not e.known_capabilities

    def test_the_real_sdk_objects_are_understood(self):
        from anthropic.types import ModelInfo
        raw = {"id": "claude-sonnet-5-5", "type": "model", "display_name": "Claude Sonnet 5.5",
               "created_at": "2026-02-15T00:00:00Z", "max_input_tokens": 1000000, "max_tokens": 128000,
               "capabilities": {
                   "batch": {"supported": True}, "citations": {"supported": True}, "code_execution": {"supported": True},
                   "context_management": {"supported": True},
                   "effort": {"supported": True, "low": {"supported": True}, "medium": {"supported": True},
                              "high": {"supported": True}, "max": {"supported": False},
                              "xhigh": {"supported": True}},
                   "image_input": {"supported": True}, "pdf_input": {"supported": True},
                   "structured_outputs": {"supported": True},
                   "thinking": {"supported": True, "types": {"adaptive": {"supported": True},
                                                             "enabled": {"supported": False}}}}}
        e = entry_from_api(ModelInfo.model_validate(raw))
        assert (e.max_input, e.max_output) == (1000000, 128000)
        assert e.thinking_adaptive is True and e.thinking_enabled is False
        assert e.effort_levels == ("low", "medium", "high", "xhigh")


class TestLoad:
    def test_newest_first_across_all_pages_asked_for_at_the_maximum_page_size(self, tmp_path):
        pages = Pages([info("claude-3-5-sonnet-20241022", 2024, 10), info("claude-opus-5-5", 2026, 3),
                       info("claude-sonnet-4-5-20250929", 2025, 9), info("not-a-chat-model", 2030, 1)])
        cat = load(lambda: pages.client, path=tmp_path / "c.json")
        assert cat.source == "live" and cat.authoritative
        assert cat.ids()[:3] == ["claude-opus-5-5", "claude-sonnet-4-5-20250929", "claude-3-5-sonnet-20241022"]
        assert "not-a-chat-model" not in cat.ids()
        assert pages.calls == [{"limit": 1000}]

    def test_a_new_model_appears_on_refresh_with_nothing_edited(self, tmp_path):
        pages = Pages([info("claude-sonnet-5-5", 2026, 2)])
        first = load(lambda: pages.client, path=tmp_path / "c.json")
        assert first.ids() == ["claude-sonnet-5-5"]
        pages2 = Pages([info("claude-sonnet-5-5", 2026, 2), info("claude-opus-6-0", 2026, 9)])
        assert load(lambda: pages2.client, path=tmp_path / "c.json").ids() == ["claude-sonnet-5-5"]   # fresh copy reused
        assert load(lambda: pages2.client, path=tmp_path / "c.json", refresh=True).ids()[0] == "claude-opus-6-0"

    def test_the_last_good_answer_survives_a_dead_network_and_a_restart(self, tmp_path):
        path = tmp_path / "c.json"
        load(lambda: Pages([info("claude-opus-5-5", 2026, 3, thinking=("adaptive",))]).client, path=path)
        claude_catalog.forget()                                              # a new process
        dead = MagicMock()
        dead.models.list.side_effect = OSError("no network")
        cat = load(lambda: dead, path=path, refresh=True)
        assert cat.source == "cache" and not cat.authoritative
        assert cat.get("claude-opus-5-5").thinking_adaptive is True          # capabilities survive the round trip

    def test_an_old_copy_is_used_when_the_network_is_down_not_discarded(self, tmp_path):
        path = tmp_path / "c.json"
        load(lambda: Pages([info("claude-opus-5-5")]).client, path=path)
        data = json.loads(path.read_text(encoding="utf-8"))
        data["fetched_at"] = time.time() - 30 * 86400
        path.write_text(json.dumps(data), encoding="utf-8")
        claude_catalog.forget()
        dead = MagicMock()
        dead.models.list.side_effect = OSError("no network")
        assert load(lambda: dead, path=path).ids() == ["claude-opus-5-5"]

    def test_with_nothing_at_all_the_seed_is_used_and_does_not_pin_the_failure(self, tmp_path):
        dead = MagicMock()
        dead.models.list.side_effect = OSError("no network")
        cat = load(lambda: dead, path=tmp_path / "c.json")
        assert cat.source == "seed" and cat.ids() == list(claude_catalog.SEED_IDS) and not cat.authoritative
        assert load(lambda: Pages([info("claude-opus-5-5")]).client, path=tmp_path / "c.json", refresh=True).source == "live"

    def test_a_broken_cache_file_is_ignored(self, tmp_path):
        path = tmp_path / "c.json"
        path.write_text("{not json", encoding="utf-8")
        dead = MagicMock()
        dead.models.list.side_effect = OSError("down")
        assert load(lambda: dead, path=path).source == "seed"


class TestCapabilities:
    def test_the_catalogue_decides_how_thinking_is_asked_for(self):
        assert thinking_style("m", ModelEntry("m", thinking_enabled=True, thinking_adaptive=False)) == "enabled"
        assert thinking_style("m", ModelEntry("m", thinking_enabled=False, thinking_adaptive=True)) == "adaptive"
        assert thinking_style("m", ModelEntry("m", thinking_enabled=False, thinking_adaptive=False)) is None

    def test_without_the_catalogue_the_version_decides(self):
        assert thinking_style("claude-3-5-sonnet-latest", None) is None
        assert thinking_style("claude-3-7-sonnet-latest", None) == "enabled"
        assert thinking_style("claude-sonnet-5-5", None) == "enabled"
        assert thinking_style("gpt-4o", None) is None

    def test_default_is_the_newest_sonnet(self):
        entries = [ModelEntry("claude-opus-5-5"), ModelEntry("claude-sonnet-5-5"), ModelEntry("claude-sonnet-4-5")]
        assert claude_catalog.default_model(entries) == "claude-sonnet-5-5"
        assert claude_catalog.default_model([ModelEntry("claude-opus-5-5")]) == "claude-opus-5-5"
        assert claude_catalog.default_model([]) is None


# ── the provider on top of it ──────────────────────────────────────────────────

@pytest.fixture
def provider(monkeypatch):
    import config
    store = {}
    monkeypatch.setattr(config, "load_config", lambda: store)
    monkeypatch.setattr(config, "save_config", lambda c: store.update(c))
    p = ClaudeProvider(api_key="sk-test")
    entries = [
        ModelEntry("claude-opus-5-5", created=3, max_output=32000, thinking_enabled=False, thinking_adaptive=True,
                   effort_levels=("low", "medium", "high")),
        ModelEntry("claude-sonnet-4-5-20250929", created=2, max_output=64000, thinking_enabled=True,
                   thinking_adaptive=False, effort_levels=()),
        ModelEntry("claude-3-5-sonnet-20241022", created=1, max_output=8192, thinking_enabled=False,
                   thinking_adaptive=False, effort_levels=()),
    ]
    monkeypatch.setattr(p, "catalog", lambda refresh=False: Catalog(entries, "live"))
    return p


def sent(provider, model, **config_values):
    """The keyword arguments stream_chat puts on messages.create."""
    import config
    for name, value in config_values.items():
        getattr(config, f"set_{name}")(value)
    client = MagicMock()
    client.messages.create.return_value = []
    with patch.object(provider, "_get_client", return_value=client):
        list(provider.stream_chat(model=model, messages=[{"role": "user", "content": "hi"}], temperature=0.3))
    return client.messages.create.call_args.kwargs


class TestRequests:
    def test_the_sampling_parameters_the_installed_sdk_cannot_take_are_not_sent(self, provider):
        """anthropic >= 1.x dropped temperature/top_p/top_k from messages.create: sending one failed every
        request with 'unexpected keyword argument' before anything left the machine."""
        import config
        config.set_top_p(0.9)
        config.set_top_k(40)
        kwargs = sent(provider, "claude-sonnet-4-5-20250929")
        assert not {"temperature", "top_p", "top_k"} & set(kwargs)
        # and the real SDK really does reject them, so this is not a hypothetical
        import anthropic
        with pytest.raises(TypeError):
            anthropic.Anthropic(api_key="x", base_url="http://127.0.0.1:9").messages.create(
                model="m", max_tokens=1, messages=[], temperature=0.3)

    def test_an_sdk_that_still_has_them_gets_them(self, provider, monkeypatch):
        monkeypatch.setattr(ClaudeProvider, "_sdk_accepts",
                            staticmethod(lambda: frozenset({"temperature", "top_p", "top_k", "thinking"})))
        assert sent(provider, "claude-3-5-sonnet-20241022")["temperature"] == 0.3

    def test_a_typed_name_is_corrected_before_it_goes_out(self, provider):
        assert sent(provider, "Claude Sonnet 4.5")["model"] == "claude-sonnet-4-5-20250929"
        assert sent(provider, "claude-opus-5.5")["model"] == "claude-opus-5-5"
        assert sent(provider, "opus")["model"] == "claude-opus-5-5"

    def test_the_model_decides_how_thinking_is_requested(self, provider):
        kwargs = sent(provider, "claude-sonnet-4-5-20250929", thinking_mode="auto", thinking_budget=4096)
        assert kwargs["thinking"] == {"type": "enabled", "budget_tokens": 4096}
        kwargs = sent(provider, "claude-opus-5-5", thinking_mode="auto", thinking_budget=4096)
        assert kwargs["thinking"] == {"type": "adaptive"}
        kwargs = sent(provider, "claude-3-5-sonnet-20241022", thinking_mode="on", thinking_budget=4096)
        assert "thinking" not in kwargs                       # a model that cannot think is not asked to

    def test_thinking_is_off_when_the_person_turned_it_off(self, provider):
        assert "thinking" not in sent(provider, "claude-opus-5-5", thinking_mode="off")

    def test_effort_is_sent_only_at_a_level_the_model_lists(self, provider):
        assert sent(provider, "claude-opus-5-5", reasoning_effort="high")["output_config"] == {"effort": "high"}
        assert "output_config" not in sent(provider, "claude-opus-5-5", reasoning_effort="xhigh")
        assert "output_config" not in sent(provider, "claude-sonnet-4-5-20250929", reasoning_effort="high")

    def test_the_output_ceiling_of_the_model_is_respected(self, provider):
        import config
        config.set_max_generation_tokens(100000)
        assert sent(provider, "claude-opus-5-5")["max_tokens"] == 32000
        assert sent(provider, "claude-3-5-sonnet-20241022")["max_tokens"] == 8192

    def test_a_thinking_budget_leaves_room_for_the_answer(self, provider):
        kwargs = sent(provider, "claude-sonnet-4-5-20250929", thinking_mode="on", thinking_budget=60000)
        assert kwargs["max_tokens"] >= 61000 or kwargs["max_tokens"] == 64000

    def test_a_model_from_another_provider_is_refused_with_the_way_out(self, provider):
        client = MagicMock()
        with patch.object(provider, "_get_client", return_value=client):
            with pytest.raises(ProviderError) as e:
                list(provider.stream_chat(model="nvidia/nemotron-3-ultra-550b-a55b:free",
                                          messages=[{"role": "user", "content": "hi"}]))
        assert "/model" in str(e.value) and "nemotron" in str(e.value)
        client.messages.create.assert_not_called()

    def test_sync_chat_sends_no_sampling_and_the_right_name(self, provider):
        client = MagicMock()
        client.messages.create.return_value = SimpleNamespace(content=[SimpleNamespace(type="text", text="ok")])
        with patch.object(provider, "_get_client", return_value=client):
            assert provider.sync_chat("claude 3.5 sonnet", [{"role": "user", "content": "x"}]) == "ok"
        kwargs = client.messages.create.call_args.kwargs
        assert kwargs["model"] == "claude-3-5-sonnet-20241022" and "temperature" not in kwargs

    def test_a_404_says_to_pick_a_model(self, provider):
        err = MagicMock()
        err.status_code = 404
        with pytest.raises(ProviderError) as e:
            provider._handle_api_status_error(err)
        assert "/model" in str(e.value)


class TestResolving:
    def test_resolve_suggest_default(self, provider):
        assert provider.resolve_model("Claude Sonnet 4.5") == "claude-sonnet-4-5-20250929"
        assert provider.resolve_model("claude-banana-4") is None
        assert "claude-opus-5-5" in provider.suggest_models("claude-opus-55")
        assert provider.default_model() == "claude-sonnet-4-5-20250929"

    def test_when_the_list_is_only_a_guess_a_well_formed_name_is_still_accepted(self, monkeypatch):
        p = ClaudeProvider(api_key="sk-test")
        monkeypatch.setattr(p, "catalog", lambda refresh=False: Catalog([ModelEntry("claude-opus-5-5")], "cache"))
        assert p.resolve_model("claude-sonnet-9.1") == "claude-sonnet-9-1"        # not listed, not disproved
        assert p.resolve_model("gpt-4o") is None

    def test_starting_a_provider_never_inherits_another_providers_model(self, provider, monkeypatch):
        monkeypatch.setattr(providers, "create_provider", lambda name=None: provider)
        assert providers.starting_model("claude", None, "nvidia/nemotron-3-ultra-550b-a55b:free") \
            == "claude-sonnet-4-5-20250929"
        assert providers.starting_model("claude", "claude-opus-5.5", "anything") == "claude-opus-5-5"

    def test_other_providers_keep_their_names(self):
        assert providers.starting_model("ollama", None, "qwen2.5:3b") == "qwen2.5:3b"
        assert providers.create_provider("ollama").resolve_model("qwen2.5:3b") == "qwen2.5:3b"


class TestAggregators:
    def test_openrouter_names_are_spelled_as_listed(self, monkeypatch):
        p = providers.OpenRouterProvider(api_key="k", base_url="http://x")
        monkeypatch.setattr(p, "list_models", lambda: ["anthropic/claude-sonnet-4.5", "openai/gpt-4o"])
        assert p.resolve_model("Anthropic/Claude Sonnet 4.5") == "anthropic/claude-sonnet-4.5"
        assert p.resolve_model("some/unlisted-model") == "some/unlisted-model"        # never rejected: gateways have more
        assert "openai/gpt-4o" in p.suggest_models("openai/gpt-4")

    def test_a_dead_catalogue_keeps_what_was_typed(self, monkeypatch):
        p = providers.OpenRouterProvider(api_key="k", base_url="http://x")
        monkeypatch.setattr(p, "list_models", MagicMock(side_effect=OSError))
        assert p.resolve_model("anthropic/claude-sonnet-4.5") == "anthropic/claude-sonnet-4.5"


class TestSwitchModel:
    class Agent:
        provider = "claude"
        model_name = "claude-opus-5-5"
        max_history_messages = 40
        max_context_tokens = 100000
        messages = []

        def set_model(self, name):
            self.model_name = name

    def test_the_name_is_corrected_and_saved_corrected(self, provider, monkeypatch):
        import command_handler
        import config
        said = []
        monkeypatch.setattr(command_handler, "print_system", lambda m, *a, **k: said.append(str(m)))
        monkeypatch.setattr(providers, "create_provider", lambda name=None: provider)
        agent = self.Agent()
        command_handler.switch_model(agent, "Claude Sonnet 4.5")
        assert agent.model_name == "claude-sonnet-4-5-20250929"
        assert config.get_model_for_provider("claude") == "claude-sonnet-4-5-20250929"
        assert any("исправлено" in line for line in said)

    def test_an_unknown_name_changes_nothing_and_offers_alternatives(self, provider, monkeypatch):
        import command_handler
        said = []
        monkeypatch.setattr(command_handler, "print_system", lambda m, *a, **k: said.append(str(m)))
        monkeypatch.setattr(providers, "create_provider", lambda name=None: provider)
        agent = self.Agent()
        command_handler.switch_model(agent, "claude-opus-55")
        assert agent.model_name == "claude-opus-5-5"
        assert any("не найдена" in line and "claude-opus-5-5" in line for line in said)

    def test_context_window_of_a_claude_model_comes_from_the_catalogue(self, provider, monkeypatch):
        import model_limits
        entry = ModelEntry("claude-opus-5-5", max_input=1000000)
        monkeypatch.setattr(provider, "catalog", lambda refresh=False: Catalog([entry], "live"))
        monkeypatch.setattr(providers, "create_provider", lambda name=None: provider)
        assert model_limits.detect_context_length("claude-opus-5-5", "claude") == 1000000
        assert model_limits.detect_context_length("claude-unknown", "claude") is None


class TestWorkspace:
    """An organisation-level key is refused by every call unless the workspace is named."""

    def test_the_workspace_travels_as_a_header_on_every_call(self, monkeypatch):
        seen = {}

        class FakeAnthropic:
            def __init__(self, **kwargs):
                seen.update(kwargs)

        p = ClaudeProvider(api_key="sk", base_url="https://x", workspace_id=" wrkspc_123 ")
        monkeypatch.setattr(p._anthropic, "Anthropic", FakeAnthropic)
        p._get_client()
        assert seen["default_headers"] == {"anthropic-workspace-id": "wrkspc_123"}
        seen.clear()
        monkeypatch.setattr(ClaudeProvider(api_key="sk")._anthropic, "Anthropic", FakeAnthropic)
        ClaudeProvider(api_key="sk")._get_client()
        assert "default_headers" not in seen

    def test_config_and_environment(self, monkeypatch):
        import config
        store = {}
        monkeypatch.setattr(config, "load_config", lambda: store)
        monkeypatch.setattr(config, "save_config", lambda c: store.update(c))
        assert config.get_claude_workspace_id() is None
        monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", "wrkspc_env")
        assert config.get_claude_workspace_id() == "wrkspc_env"
        config.set_claude_workspace_id("wrkspc_saved")
        assert config.get_claude_workspace_id() == "wrkspc_saved"
        assert providers.create_provider("claude")._workspace_id == "wrkspc_saved"
        config.set_claude_workspace_id("  ")
        assert config.get_claude_workspace_id() == "wrkspc_env"

    def test_the_refusal_is_explained_with_the_way_out(self):
        err = MagicMock()
        err.status_code = 400
        err.message = "This API key is not scoped to a workspace, so this request must include the anthropic-workspace-id header"
        with pytest.raises(ProviderError) as e:
            ClaudeProvider(api_key="sk")._handle_api_status_error(err)
        assert "/key workspace" in str(e.value) and "ANTHROPIC_WORKSPACE_ID" in str(e.value)

    def test_the_catalogue_remembers_why_it_is_not_live(self, tmp_path):
        dead = MagicMock()
        dead.models.list.side_effect = RuntimeError("Error code: 400 - key is not scoped to a workspace")
        cat = load(lambda: dead, path=tmp_path / "c.json")
        assert cat.source == "seed" and "workspace" in cat.problem

    def test_the_menu_tells_the_person_which_list_they_are_looking_at(self, monkeypatch):
        import ui
        said = []
        monkeypatch.setattr(ui, "print_system", lambda m, *a, **k: said.append(str(m)))
        p = ClaudeProvider(api_key="sk")
        dead = MagicMock()
        dead.models.list.side_effect = RuntimeError("key is not scoped to a workspace")
        monkeypatch.setattr(p, "_get_client", lambda: dead)
        monkeypatch.setattr(providers, "create_provider", lambda name=None: p)
        monkeypatch.setattr(ui, "_select_from_list", lambda *a, **k: None)
        ui.select_model("claude-opus-5-5")
        assert any("недоступен" in line and "/key workspace" in line for line in said)


class TestDisplayNames:
    """"Claude Sonnet 5.5" for the person, `claude-sonnet-5-5` for the config and the API."""

    @pytest.mark.parametrize("model_id, shown", [
        ("claude-sonnet-5-5", "Claude Sonnet 5.5"),
        ("claude-opus-5-5-20260301", "Claude Opus 5.5"),
        ("claude-haiku-4-5", "Claude Haiku 4.5"),
        ("claude-fable-5-1", "Claude Fable 5.1"),
        ("claude-sonnet-4-0", "Claude Sonnet 4.0"),
        ("claude-3-5-sonnet-20241022", "Claude 3.5 Sonnet"),
        ("claude-3-haiku-20240307", "Claude 3 Haiku"),
        ("gpt-4o", "gpt-4o"),
        ("anthropic/claude-sonnet-4.5", "anthropic/claude-sonnet-4.5"),
        ("", ""),
    ])
    def test_built_from_the_id_when_the_catalogue_has_no_name(self, model_id, shown):
        assert claude_catalog.display_name(model_id, entries=[]) == shown

    def test_anthropics_own_name_wins(self):
        entries = [ModelEntry("claude-sonnet-5-5", display_name="Claude Sonnet 5.5 (preview)")]
        assert claude_catalog.display_name("CLAUDE-SONNET-5-5", entries) == "Claude Sonnet 5.5 (preview)"

    def test_only_claude_ids_are_rewritten(self):
        assert claude_catalog.label_for("claude", "claude-sonnet-5-5") == "Claude Sonnet 5.5"
        assert claude_catalog.label_for("anthropic", "claude-sonnet-5-5") == "Claude Sonnet 5.5"
        assert claude_catalog.label_for("openrouter", "anthropic/claude-sonnet-5.5") == "anthropic/claude-sonnet-5.5"
        assert claude_catalog.label_for("ollama", "qwen2.5:3b") == "qwen2.5:3b"

    def test_a_toolbar_never_waits_for_the_network(self, tmp_path, monkeypatch):
        calls = []
        monkeypatch.setattr(claude_catalog, "fetch_live", lambda client: calls.append(1) or [])
        assert claude_catalog.display_name("claude-opus-5-5") == "Claude Opus 5.5"
        assert calls == []

    def test_a_saved_catalogue_supplies_the_name(self):
        item = info("claude-opus-5-5", 2026, 3)
        item.display_name = "Claude Opus Five Point Five"
        claude_catalog.load(lambda: Pages([item]).client)
        claude_catalog.forget()                                  # a new process: only the disk copy remains
        assert claude_catalog.display_name("claude-opus-5-5") == "Claude Opus Five Point Five"

    def test_the_menu_shows_names_and_saves_ids(self, provider, monkeypatch):
        import ui
        entries = [ModelEntry("claude-opus-5-5", display_name="Claude Opus 5.5"),
                   ModelEntry("claude-sonnet-4-5-20250929", display_name="Claude Sonnet 4.5")]
        monkeypatch.setattr(provider, "catalog", lambda refresh=False: Catalog(entries, "live"))
        monkeypatch.setattr(providers, "create_provider", lambda name=None: provider)
        shown = {}

        def choose(prompt, choices, default=None):
            if not shown:                                  # the model menu; later questions (context size) are not ours
                shown["choices"], shown["default"] = list(choices), default
                return choices[1]
            return None
        monkeypatch.setattr(ui, "_select_from_list", choose)
        monkeypatch.setattr(ui, "print_system", lambda *a, **k: None)
        result = ui.select_model("claude-sonnet-4-5-20250929")
        assert shown["choices"] == ["Claude Opus 5.5  (claude-opus-5-5)",
                                    "Claude Sonnet 4.5  (claude-sonnet-4-5-20250929)"]
        assert shown["default"] == "Claude Sonnet 4.5  (claude-sonnet-4-5-20250929)"
        assert result == "claude-sonnet-4-5-20250929"

    def test_the_switch_message_uses_the_name(self, provider, monkeypatch):
        import command_handler
        said = []
        monkeypatch.setattr(command_handler, "print_system", lambda m, *a, **k: said.append(str(m)))
        monkeypatch.setattr(providers, "create_provider", lambda name=None: provider)
        agent = TestSwitchModel.Agent()
        command_handler.switch_model(agent, "opus")
        assert any("Модель изменена: Claude Opus 5.5 (claude-opus-5-5)" in line for line in said)
