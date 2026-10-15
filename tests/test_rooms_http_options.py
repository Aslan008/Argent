"""Rooms: the `http` node (a local test server), and questions with answers to pick from (`options`)."""

import http.server
import json
import threading

import pytest

from src.rooms import httpnode
from src.rooms.engine import Engine, replayable
from src.rooms.integrations import Integrations
from src.rooms.models import Node, Room, State
from src.rooms.rules import Rules
from src.rooms.runner import NodeRunner, match_option
from src.rooms.session import describe_call
from src.rooms.validator import room_warnings, validate_room


class _Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        raw = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(raw)

    def do_GET(self):
        if self.path.startswith("/missing"):
            return self._send(404, b"no such thing", "text/plain")
        if self.path.startswith("/to-file"):
            self.send_response(302)
            self.send_header("Location", "file:///etc/passwd")
            self.end_headers()
            return
        self._send(200, {"path": self.path, "auth": self.headers.get("Authorization"), "n": 3})

    do_HEAD = do_GET

    def do_POST(self):
        size = int(self.headers.get("Content-Length") or 0)
        data = self.rfile.read(size)
        self._send(201, {"type": self.headers.get("Content-Type"), "body": data.decode()})


@pytest.fixture(scope="module")
def base():
    server = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def call(**kw):
    return httpnode.run(Node(id="h", type="http", **kw))


class TestHttpNode:
    def test_get_with_a_query_and_the_answer_split_into_fields(self, base):
        r = call(args={"url": base + "/x", "query": {"a": "1", "b": "два"}})
        assert r.values["status"] == 200 and r.values["ok"] is True
        assert r.values["json"]["n"] == 3 and r.values["json"]["path"].startswith("/x?a=1&b=")
        assert json.loads(str(r))["n"] == 3

    def test_post_json_and_form(self, base):
        r = call(mode="post", args={"url": base, "json": {"k": [1, 2], "t": "текст"}})
        assert r.values["status"] == 201
        assert r.values["json"]["type"].startswith("application/json")
        assert json.loads(r.values["json"]["body"]) == {"k": [1, 2], "t": "текст"}
        f = call(mode="post", args={"url": base, "form": {"a": "1", "b": "x y"}})
        assert f.values["json"]["body"] == "a=1&b=x+y"

    def test_an_error_code_is_an_answer_not_a_failure(self, base):
        r = call(args={"url": base + "/missing"})
        assert r.values["status"] == 404 and r.values["ok"] is False and str(r) == "no such thing"

    def test_a_secret_comes_from_the_environment_and_is_blanked_out(self, base, monkeypatch):
        monkeypatch.setenv("ROOMS_TEST_TOKEN", "s3cr3t-value")
        r = call(args={"url": base, "headers": {"Authorization": "Bearer ${ENV:ROOMS_TEST_TOKEN}"}})
        assert r.values["json"]["auth"] == "Bearer ***"           # the server echoed it; the room never sees it
        assert "s3cr3t" not in str(r) and "s3cr3t" not in json.dumps(r.values)
        monkeypatch.delenv("ROOMS_TEST_TOKEN")
        missing = call(args={"url": base, "headers": {"Authorization": "${ENV:ROOMS_TEST_TOKEN}"}})
        assert str(missing).startswith("Error:") and "ROOMS_TEST_TOKEN" in str(missing) and missing.values["ok"] is False

    def test_only_web_addresses_are_reachable_also_after_a_redirect(self, base):
        for bad in ("file:///etc/passwd", "ftp://example.com/x"):
            with pytest.raises(ValueError):
                Node(id="h", type="http", args={"url": bad})
        r = call(args={"url": base + "/to-file"})
        assert str(r).startswith("Error:") and r.values["ok"] is False

    def test_unreachable_is_an_error_text_not_a_crash(self):
        r = call(args={"url": "http://127.0.0.1:1/", "timeout": 2})
        assert str(r).startswith("Error:") and r.values["status"] == 0


class TestHttpModel:
    def test_what_is_required_and_what_is_refused(self):
        with pytest.raises(ValueError):
            Node(id="h", type="http", args={})
        with pytest.raises(ValueError):
            Node(id="h", type="http", mode="fetch", args={"url": "https://x.example"})
        with pytest.raises(ValueError):
            Node(id="h", type="http", mode="post", args={"url": "https://x.example", "json": {"a": 1}, "body": "x"})
        with pytest.raises(ValueError):
            Node(id="h", type="http", args={"url": "https://x.example", "timeout": 999})

    def test_looking_is_repeatable_and_changing_is_not(self):
        get = Node(id="h", type="http", args={"url": "https://x.example"})
        post = Node(id="h", type="http", mode="post", args={"url": "https://x.example"})
        assert get.idempotent and replayable(get)
        assert not post.idempotent and not replayable(post)

    def test_result_naming_and_derived_values_are_allowed(self):
        n = Node(id="h", type="http", args={"url": "https://x.example"}, save_as="reply", sets={"n": "json:n"}, confirm=True)
        assert n.save_as == "reply" and n.confirm

    def test_rules_and_the_confirmation_text(self):
        post = Node(id="h", type="http", mode="post", args={"url": "https://x.example", "headers": {"Authorization": "Bearer abc"}})
        get = Node(id="h", type="http", args={"url": "https://x.example"})
        assert Rules({"read_only": True}).node_block(post) is not None
        assert Rules({"read_only": True}).node_block(get) is None
        assert Rules({"deny_tools": ["http"]}).node_block(get) is not None
        assert Rules({"no_tools": True}).node_block(get) is not None
        text = describe_call(post)
        assert "POST" in text and "https://x.example" in text and "abc" not in text      # a header never reaches a question

    def test_the_question_before_a_git_step_names_the_action(self):
        assert describe_call(Node(id="g", type="git", mode="commit", message="ship it")).startswith("git commit")


class TestHttpInARoom:
    def room(self, url):
        return Room(room="t", budget={"max_iterations": 10, "max_tokens": 1000},
                    nodes=[Node(id="ping", type="http", mode="head", args={"url": url}, save_as="pong")],
                    edges=[{"from": "ping", "if": "state.ping.ok", "to": "EXIT:up"},
                           {"from": "ping", "else": True, "to": "EXIT:down"}])

    def test_a_room_branches_on_the_answer(self, base):
        runner = NodeRunner(integrations=Integrations())
        up = self.room(base)
        assert validate_room(up, set()) == []
        assert Engine().run(up, State(), runner.run_node).exit == "up"
        down = self.room(base + "/missing")
        assert Engine().run(down, State(), runner.run_node).exit == "down"

    def test_warnings_for_a_change_without_asking_and_for_a_secret_in_the_file(self):
        room = Room(room="t", budget={"max_iterations": 10, "max_tokens": 1000},
                    nodes=[Node(id="p", type="http", mode="post",
                                args={"url": "https://x.example", "headers": {"Authorization": "Bearer abc123"}})],
                    edges=[{"from": "p", "to": "EXIT:done"}])
        notes = room_warnings(room)
        assert any("without asking" in w for w in notes)
        assert any("literal value" in w for w in notes)
        safe = Room(room="t", budget={"max_iterations": 10, "max_tokens": 1000},
                    nodes=[Node(id="p", type="http", mode="post", confirm=True,
                                args={"url": "https://x.example", "headers": {"Authorization": "Bearer ${ENV:TOKEN}"}})],
                    edges=[{"from": "p", "to": "EXIT:done"}])
        assert not any("literal value" in w or "without asking" in w for w in room_warnings(safe))


class TestOptions:
    def test_matching_a_typed_answer_to_an_offer(self):
        offers = ["Продолжить", "Остановиться", "Позже"]
        assert match_option("продолжить", offers) == "Продолжить"
        assert match_option("2", offers) == "Остановиться"
        assert match_option("ост", offers) == "Остановиться"
        assert match_option("п", offers) == ""                 # two start with it: not a guess
        assert match_option("", offers) == "" and match_option("9", offers) == ""

    def test_the_model_checks_the_offers(self):
        assert Node(id="q", type="human_pause", prompt="?", options=["да ", "нет"]).options == ["да", "нет"]
        for bad in (["один"], ["a", "A"], ["a", ""], [str(i) for i in range(13)], ["x" * 81, "y"]):
            with pytest.raises(ValueError):
                Node(id="q", type="human_pause", prompt="?", options=bad)
        with pytest.raises(ValueError):
            Node(id="q", type="agent", role="Coder", options=["a", "b"], instruction="x",
                 budget={"max_iterations": 1, "max_tokens": 10})

    def test_the_choice_is_stored_and_the_room_can_branch_on_it(self):
        def human(node, state):
            state.data[f"{node.id}.output"] = "2"             # the person typed the number of an offer

        room = Room(room="t", budget={"max_iterations": 10, "max_tokens": 1000},
                    nodes=[Node(id="ask", type="human_pause", prompt="Что делаем?", into="decision",
                                options=["Выпустить", "Отложить"])],
                    edges=[{"from": "ask", "if": "state.ask.choice == 'Выпустить'", "to": "EXIT:release"},
                           {"from": "ask", "else": True, "to": "EXIT:later"}])
        assert validate_room(room, set()) == []
        state = State()
        result = Engine().run(room, state, NodeRunner(human_prompt=human).run_node)
        assert result.exit == "later" and state.data["decision"] == "Отложить" and state.data["ask.choice"] == "Отложить"

    def test_the_editor_run_offers_buttons(self):
        from src.rooms.editor_runs import Run, RunManager
        manager = RunManager(runner_factory=lambda human, confirm: None)
        run = Run("r1", "t", "room", {})
        node = Node(id="ask", type="human_pause", prompt="?", options=["Да", "Нет"])
        seen = {}

        def answer():
            while run.prompt is None:
                pass
            seen.update(run.prompt)
            run._answer = "Да"
            run._answered.set()

        t = threading.Thread(target=answer)
        t.start()
        got = manager._ask(run, "ask", node, "Продолжать?")
        t.join()
        assert got == "Да" and seen["options"] == ["Да", "Нет"]
