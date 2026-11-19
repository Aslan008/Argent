"""Groups in the Telegram bridge: who may talk, who may press, what a group turn is.

The rules themselves live in ``src/bridge/policy.py``; these tests check that the bridge
actually applies them — that is where the dangerous mistakes are (a button anyone in a
group could press, a group turn that quietly gets tools).
"""

import time

from src.bridge.session import AgentSession, _ApprovalDecider
from src.telegram import bot as bot_module

OWNER = "42"
MEMBER = "777"
GROUP = "-1001234567890"
OTHER_GROUP = "-100999"


class Recorder:
    """Stands in for TelegramClient: records what would have been sent."""

    def __init__(self):
        self.sent = []
        self.answers = []
        self.reactions = []
        self.edits = []
        self.last_error = ""

    def send_message(self, chat_id, text, **kw):
        self.sent.append((str(chat_id), text, kw))
        return {"message_id": len(self.sent)}

    def answer_callback_query(self, query_id, text=""):
        self.answers.append((query_id, text))

    def send_chat_action(self, chat_id, action):
        return True

    def set_reaction(self, chat_id, message_id, emoji):
        self.reactions.append((chat_id, message_id, emoji))
        return True

    def edit_message_text(self, chat_id, message_id, text, **kw):
        self.edits.append((chat_id, message_id, text))
        return True

    def send_photo(self, *a, **kw):
        return {}

    def send_document(self, *a, **kw):
        return {}


def make_bridge(monkeypatch, groups=(), agent_mode=(), owner=OWNER, mode="isolated"):
    groups, agent_mode = set(groups), set(agent_mode)
    monkeypatch.setattr(bot_module, "get_telegram_chat_id", lambda: owner)
    monkeypatch.setattr(bot_module, "is_telegram_group_allowed", lambda cid: str(cid) in groups)
    monkeypatch.setattr(bot_module, "get_telegram_group_agent", lambda cid: str(cid) in agent_mode)
    monkeypatch.setattr(bot_module, "get_telegram_groups", lambda: sorted(groups))
    monkeypatch.setattr(bot_module, "get_telegram_mode", lambda: mode)
    bridge = bot_module.TelegramBridge()
    bridge.client = Recorder()
    bridge._bot_info = {"id": 7, "username": "argent_bot"}
    return bridge


def message(text, sender=OWNER, chat_id=GROUP, chat_type="supergroup", **extra):
    msg = {
        "message_id": 1,
        "date": int(time.time()),
        "chat": {"id": int(chat_id), "type": chat_type},
        "from": {"id": int(sender), "first_name": "Иван"},
        "text": text,
    }
    msg.update(extra)
    return msg


# ─── who may do what ─────────────────────────────────────────────────────────

def test_the_owners_private_chat_keeps_all_tools_and_buttons(monkeypatch):
    policy = make_bridge(monkeypatch)._policy_for(OWNER, "private", OWNER)

    assert policy is not None
    assert policy.tools_allowed and policy.approvals_allowed
    assert not policy.is_group


def test_a_stranger_in_a_private_chat_is_ignored(monkeypatch):
    bridge = make_bridge(monkeypatch)
    assert bridge._policy_for("555", "private", "555") is None


def test_a_group_is_ignored_until_it_is_allowed(monkeypatch):
    bridge = make_bridge(monkeypatch)
    assert bridge._policy_for(GROUP, "supergroup", OWNER) is None


def test_an_allowed_group_is_chat_only(monkeypatch):
    policy = make_bridge(monkeypatch, groups=[GROUP])._policy_for(GROUP, "supergroup", OWNER)

    assert policy is not None and policy.is_group
    assert policy.tools_allowed is False
    assert policy.approvals_allowed is False, "nobody in a group may answer for the machine"


def test_agent_mode_turns_on_tools_for_the_owner_only(monkeypatch):
    bridge = make_bridge(monkeypatch, groups=[GROUP], agent_mode=[GROUP])

    owner = bridge._policy_for(GROUP, "supergroup", OWNER)
    member = bridge._policy_for(GROUP, "supergroup", MEMBER)

    assert owner.tools_allowed is True
    assert owner.approvals_allowed is False, "tools never come with approvals in a group"
    assert member.tools_allowed is False


# ─── being addressed ─────────────────────────────────────────────────────────

def test_mention_and_reply_are_both_recognised(monkeypatch):
    bridge = make_bridge(monkeypatch)

    assert bridge._mentions_bot(message("@argent_bot привет"))
    assert bridge._mentions_bot(message("смотри", reply_to_message={"from": {"id": 7}}))
    assert not bridge._mentions_bot(message("привет всем"))


def test_the_mention_does_not_reach_the_model(monkeypatch):
    bridge = make_bridge(monkeypatch)
    assert bridge._strip_mention("@argent_bot   привет") == "привет"


def test_a_mention_entity_is_enough(monkeypatch):
    """Telegram marks a mention with an entity; that mark is the authority, not the text."""
    bridge = make_bridge(monkeypatch)
    msg = message("@argent_botПривет",
                  entities=[{"type": "mention", "offset": 0, "length": 11}])
    assert bridge._mentions_bot(msg)


def test_a_mention_without_a_space_is_still_us(monkeypatch):
    """A username holds no Cyrillic, so `@botПривет` is still addressed to the bot."""
    bridge = make_bridge(monkeypatch)
    assert bridge._mentions_bot(message("@argent_botПривет"))


def test_a_mention_of_another_bot_in_the_group_is_not_us(monkeypatch):
    """Groups may hold several bots: it is not ours to answer for somebody else."""
    bridge = make_bridge(monkeypatch)
    assert not bridge._mentions_bot(message("@other_bot привет"))


def test_a_text_mention_of_the_bot_counts(monkeypatch):
    """Some clients insert the name as a text_mention (no @username) — the user id is the fact."""
    bridge = make_bridge(monkeypatch)
    msg = message("Argent привет",
                  entities=[{"type": "text_mention", "offset": 0, "length": 6, "user": {"id": 7}}])
    assert bridge._mentions_bot(msg)


def test_stripping_the_mention_leaves_no_invisible_characters(monkeypatch):
    bridge = make_bridge(monkeypatch)
    assert bridge._strip_mention("@argent_bot\u2060 привет") == "привет"


# ─── routing ─────────────────────────────────────────────────────────────────

def test_a_group_message_nobody_addressed_is_ignored(monkeypatch):
    bridge = make_bridge(monkeypatch, groups=[GROUP])
    calls = []
    monkeypatch.setattr(bridge, "_run_prompt_turn", lambda *a, **kw: calls.append((a, kw)))

    bridge._handle_message(message("привет всем"))

    assert calls == []
    assert bridge.client.sent == [], "silence is the answer to an unaddressed group message"


def test_a_group_turn_gets_no_tools_and_names_the_speaker(monkeypatch):
    bridge = make_bridge(monkeypatch, groups=[GROUP])
    calls = []
    monkeypatch.setattr(bridge, "_run_prompt_turn", lambda *a, **kw: calls.append((a, kw)))

    bridge._handle_message(message("@argent_bot как дела?"))

    assert len(calls) == 1
    (chat_id, text), kw = calls[0]
    assert str(chat_id) == GROUP
    assert kw["allowed_tools"] == []
    assert text.startswith("[Групповой чат")
    assert "Иван: как дела?" in text


def test_the_owners_private_turn_keeps_every_tool(monkeypatch):
    bridge = make_bridge(monkeypatch)
    calls = []
    monkeypatch.setattr(bridge, "_run_prompt_turn", lambda *a, **kw: calls.append((a, kw)))

    bridge._handle_message(message("почисти логи", chat_id=OWNER, chat_type="private"))

    assert calls and calls[0][1]["allowed_tools"] is None


def test_an_unbound_group_is_told_once_how_to_allowed(monkeypatch):
    bridge = make_bridge(monkeypatch)

    bridge._handle_message(message("привет"))
    bridge._handle_message(message("привет ещё раз"))

    assert len(bridge.client.sent) == 1
    text = bridge.client.sent[0][1]
    assert GROUP in text and "/group allow" in text


# ─── buttons and commands ────────────────────────────────────────────────────

def test_a_member_cannot_press_a_button_in_a_group(monkeypatch):
    bridge = make_bridge(monkeypatch, groups=[GROUP])
    session = bridge.session_for(GROUP)
    answered = []
    monkeypatch.setattr(session, "reply_approval", lambda *a: answered.append(a))

    bridge._handle_callback_query({
        "id": "cb1",
        "from": {"id": int(MEMBER)},
        "data": "appr:1:once",
        "message": {"message_id": 5, "chat": {"id": int(GROUP), "type": "supergroup"}},
    })

    assert answered == [], "a group member must not be able to approve anything"
    assert bridge.client.answers and "владелец" in bridge.client.answers[0][1]


def test_the_group_command_surface_is_small(monkeypatch):
    bridge = make_bridge(monkeypatch, groups=[GROUP])
    policy = bridge._policy_for(GROUP, "supergroup", MEMBER)

    bridge._handle_command(GROUP, "/mode shared", policy)

    assert any("В группе доступны" in text for _, text, _ in bridge.client.sent)


def test_only_the_owner_switches_tools_on_in_a_group(monkeypatch):
    store = {}
    bridge = make_bridge(monkeypatch, groups=[GROUP])
    monkeypatch.setattr(bot_module, "get_telegram_group_agent", lambda cid: bool(store.get(str(cid))))
    monkeypatch.setattr(bot_module, "set_telegram_group_agent",
                        lambda cid, on: store.__setitem__(str(cid), on))

    bridge._handle_command(GROUP, "/agent", bridge._policy_for(GROUP, "supergroup", MEMBER))
    assert store == {}, "a member may not turn tools on"

    bridge._handle_command(GROUP, "/agent", bridge._policy_for(GROUP, "supergroup", OWNER))
    assert store == {GROUP: True}


# ─── history and sessions ────────────────────────────────────────────────────

def test_reset_clears_only_its_own_group(monkeypatch):
    bridge = make_bridge(monkeypatch, groups=[GROUP, OTHER_GROUP])
    first = bridge.session_for(GROUP)
    second = bridge.session_for(OTHER_GROUP)

    bridge._reset_dialog(GROUP)

    assert bridge.session_for(GROUP) is not first
    assert bridge.session_for(OTHER_GROUP) is second


def test_a_group_session_never_asks_for_an_approval(monkeypatch):
    bridge = make_bridge(monkeypatch, groups=[GROUP])
    session = bridge.session_for(GROUP)
    events = []
    session.set_event_sink(events.append)

    assert session.approvals_allowed is False
    assert session._approval_backend("run: rm -rf build", True, "run") == "deny"
    assert any("подтверждения недоступны" in e.get("text", "") for e in events)


def test_a_group_decider_declares_that_nobody_is_watching(monkeypatch):
    """`unattended` is what switches off the gate's shortcuts (session grants, /vibe)."""
    bridge = make_bridge(monkeypatch, groups=[GROUP])

    assert _ApprovalDecider(bridge.session_for(GROUP)).unattended is True
    assert _ApprovalDecider(AgentSession()).unattended is False


def test_vibe_at_the_pc_cannot_approve_work_in_a_group(monkeypatch):
    """The gate's shortcuts are consent from a present human. A group chat has none."""
    import argent.approval as approval

    bridge = make_bridge(monkeypatch, groups=[GROUP], agent_mode=[GROUP])
    session = bridge.session_for(GROUP)
    previous = approval.get_policy()
    approval.set_policy(approval.POLICY_AUTO)
    approval.set_approval_backend(_ApprovalDecider(session))
    try:
        allowed = approval.request_approval(
            "run: echo group-check", destructive=False, grant_key="test:group:vibe")
    finally:
        approval.reset_approval_backend()
        approval.set_policy(previous)

    assert allowed is False, "vibe at the PC must not run commands for a group chat"


# ─── persona ─────────────────────────────────────────────────────────────────

def persona_store(monkeypatch, initial=None):
    """A stand-in for the config: the role is text per chat, cleared by an empty value."""
    store = dict(initial or {})

    def _get(chat_id):
        return store.get(str(chat_id), "")

    def _set(chat_id, text):
        if text:
            store[str(chat_id)] = text
        else:
            store.pop(str(chat_id), None)

    monkeypatch.setattr(bot_module, "get_chat_persona", _get)
    monkeypatch.setattr(bot_module, "set_chat_persona", _set)
    return store


def test_a_member_cannot_set_the_role_in_a_group(monkeypatch):
    store = persona_store(monkeypatch)
    bridge = make_bridge(monkeypatch, groups=[GROUP])

    bridge._handle_command(GROUP, "/persona rude", bridge._policy_for(GROUP, "supergroup", MEMBER))

    assert store == {}
    assert any("только владелец" in text for _, text, _ in bridge.client.sent)


def test_the_owner_can_set_a_preset_in_a_group(monkeypatch):
    from src.bridge.persona import PRESETS

    store = persona_store(monkeypatch)
    bridge = make_bridge(monkeypatch, groups=[GROUP])

    bridge._handle_command(GROUP, "/persona rude", bridge._policy_for(GROUP, "supergroup", OWNER))

    assert store[GROUP] == PRESETS["rude"]


def test_a_role_is_shown_and_can_be_cleared(monkeypatch):
    store = persona_store(monkeypatch, {GROUP: "будь пиратом"})
    bridge = make_bridge(monkeypatch, groups=[GROUP])
    owner = bridge._policy_for(GROUP, "supergroup", OWNER)

    bridge._handle_command(GROUP, "/persona", owner)
    assert any("будь пиратом" in text for _, text, _ in bridge.client.sent)

    bridge._handle_command(GROUP, "/persona off", owner)
    assert store == {}


def test_a_role_rides_on_that_chats_session_only(monkeypatch):
    store = persona_store(monkeypatch, {GROUP: "роль группы", OWNER: "роль лички"})
    bridge = make_bridge(monkeypatch, groups=[GROUP])

    assert bridge._persona_for(GROUP) == "роль группы"
    assert bridge._persona_for(OWNER) == "роль лички"


def test_a_companion_private_chat_never_carries_a_role(monkeypatch):
    """In companion mode the private chat IS the terminal's agent — a role would leak to the PC."""
    persona_store(monkeypatch, {OWNER: "роль лички"})
    bridge = make_bridge(monkeypatch, groups=[GROUP], mode="shared")

    assert bridge._persona_for(OWNER) == ""
    assert bridge._persona_for(GROUP) == "", "the group has no role in this test"
    # And the command says why instead of silently doing nothing.
    bridge._handle_command(OWNER, "/persona rude", bridge._policy_for(OWNER, "private", OWNER))
    assert any("изолированном режиме" in text for _, text, _ in bridge.client.sent)


# ─── the bot's own model ─────────────────────────────────────────────────────

class _ModelAgent:
    """Just the model surface `_apply_bot_model` touches."""

    def __init__(self, provider="zai", model="zai-model"):
        self.provider = provider
        self.model_name = model
        self.preferred_model = ""
        self.preferred_provider = ""

    def set_provider(self, name):
        self.provider = name
        self.model_name = f"{name}-default"

    def set_model(self, name):
        self.model_name = name


def test_the_bot_model_is_pushed_onto_its_sessions(monkeypatch):
    monkeypatch.setattr("argent.config.get_telegram_model", lambda: "bot:free")
    monkeypatch.setattr("argent.config.get_telegram_provider", lambda: "openrouter")
    bridge = make_bridge(monkeypatch, groups=[GROUP])
    session = bridge.session_for(GROUP)
    agent = _ModelAgent()
    session._agent = agent

    bridge._apply_bot_model(GROUP, session)

    assert agent.preferred_model == "bot:free"
    assert agent.provider == "openrouter"
    assert agent.model_name == "bot:free"


def test_without_a_bot_model_the_agent_is_left_alone(monkeypatch):
    monkeypatch.setattr("argent.config.get_telegram_model", lambda: "")
    monkeypatch.setattr("argent.config.get_telegram_provider", lambda: "")
    bridge = make_bridge(monkeypatch, groups=[GROUP])
    session = bridge.session_for(GROUP)
    agent = _ModelAgent()
    session._agent = agent

    bridge._apply_bot_model(GROUP, session)

    assert (agent.provider, agent.model_name) == ("zai", "zai-model")
    assert agent.preferred_model == ""


def test_a_companion_private_chat_keeps_the_terminal_model(monkeypatch):
    """In companion mode the private chat IS the terminal's agent: its model is not ours to change."""
    monkeypatch.setattr("argent.config.get_telegram_model", lambda: "bot:free")
    monkeypatch.setattr("argent.config.get_telegram_provider", lambda: "openrouter")
    bridge = make_bridge(monkeypatch, mode="shared")
    session = bridge.session
    agent = _ModelAgent()
    session._agent = agent

    bridge._apply_bot_model(OWNER, session)

    assert (agent.provider, agent.model_name) == ("zai", "zai-model")
    assert agent.preferred_model == ""


# ─── the prompt profile of a chat ──────────────────────────────────────────

def test_a_group_turn_uses_the_chat_prompt(monkeypatch):
    bridge = make_bridge(monkeypatch, groups=[GROUP])
    session = bridge.session_for(GROUP)

    bridge._prepare_session(GROUP, session, allowed_tools=[])

    assert session.prompt_mode == "chat"


def test_the_owners_private_chat_keeps_the_agent_prompt(monkeypatch):
    bridge = make_bridge(monkeypatch)
    session = bridge.session_for(OWNER)

    bridge._prepare_session(OWNER, session, allowed_tools=None)

    assert session.prompt_mode == "agent"


def test_tools_on_in_a_group_bring_the_agent_prompt_back(monkeypatch):
    """With tools the model needs the tool protocol; the owner asked for an agent there."""
    bridge = make_bridge(monkeypatch, groups=[GROUP], agent_mode=[GROUP])
    session = bridge.session_for(GROUP)

    bridge._prepare_session(GROUP, session, allowed_tools=None)

    assert session.prompt_mode == "agent"


# ─── reactions ─────────────────────────────────────────────────────────────

def test_the_model_chooses_the_reaction_in_a_group(monkeypatch):
    bridge = make_bridge(monkeypatch, groups=[GROUP])

    assert bridge._turn_reaction(GROUP, "done", "🔥") == "🔥"
    assert bridge._turn_reaction(GROUP, "done", None) is None, "no tag means no reaction"
    assert bridge._turn_reaction(GROUP, "failed", "🔥") is None, "a failed turn has no verdict to react to"


def test_the_private_chat_keeps_the_outcome_reaction(monkeypatch):
    bridge = make_bridge(monkeypatch)

    assert bridge._turn_reaction(OWNER, "done", None) == "👍"
    assert bridge._turn_reaction(OWNER, "failed", None) == "😢"


def test_reactions_off_silences_the_chosen_one(monkeypatch):
    bridge = make_bridge(monkeypatch, groups=[GROUP])
    monkeypatch.setattr(bot_module, "get_telegram_reactions", lambda: "off")

    assert bridge._turn_reaction(GROUP, "done", "🔥") is None


def test_only_the_owner_learns_the_model_in_a_group(monkeypatch):
    """The name must not travel in the group: either it goes to the owner's DM, or nothing."""
    bridge = make_bridge(monkeypatch, groups=[GROUP])

    bridge._handle_command(GROUP, "/model", bridge._policy_for(GROUP, "supergroup", MEMBER))
    assert any("только владелец" in text for _, text, _ in bridge.client.sent)


def test_the_model_answer_goes_to_the_owner_privately(monkeypatch):
    bridge = make_bridge(monkeypatch, groups=[GROUP])
    sent_to = []
    monkeypatch.setattr(bridge.client, "send_message",
                        lambda chat_id, text, **kw: sent_to.append((str(chat_id), text)) or {"message_id": 1})

    bridge._handle_command(GROUP, "/model", bridge._policy_for(GROUP, "supergroup", OWNER))

    assert any(chat == OWNER for chat, _ in sent_to), "the owner gets it in a private chat"
    assert not any(chat == GROUP and "отвечает" in text for chat, text in sent_to), \
        "the model's name must never appear in the group message"


def test_a_chat_only_turn_reaches_the_agent_with_no_tools():
    class FakeAgent:
        def __init__(self):
            self.calls = []
            self._turn_end_reason = "final answer"

        def process_user_input(self, text, allowed_tools=None):
            self.calls.append((text, allowed_tools))
            yield {"type": "content_stream", "content": "ok"}

        def get_context_usage(self):
            return {"tokens": 1, "max": 10, "percent": 10.0}

    agent = FakeAgent()
    session = AgentSession(agent=agent)

    session.start("привет", allowed_tools=[])
    for _ in range(100):
        if session.get_event().get("type") == "done":
            break

    assert agent.calls == [("привет", [])]
