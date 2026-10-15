import socket
from unittest.mock import patch, MagicMock

import pytest

import config
import doctor
from command_handler import _mcp_add
from src.agent.mcp_prompt import build_mcp_section


class TestBlenderMCPPreset:
    def test_mcp_add_blender_preset(self, monkeypatch):
        saved_servers = []
        registered = {}

        def fake_add_mcp_server(name, **kwargs):
            saved_servers.append((name, kwargs))

        fake_mcp_client = MagicMock()
        fake_mcp_client.register_and_start.side_effect = lambda name, cfg: registered.update({name: cfg}) or "OK"

        monkeypatch.setattr("command_handler.add_mcp_server", fake_add_mcp_server)
        monkeypatch.setattr("mcp_client.mcp_client", fake_mcp_client)

        _mcp_add(["/mcp", "add", "blender"])

        assert len(saved_servers) == 1
        name, kwargs = saved_servers[0]
        assert name == "blender"
        assert kwargs["server_type"] == "stdio"
        assert kwargs["command"] == "uvx"
        assert kwargs["args"] == ["blender-mcp"]

        assert "blender" in registered
        assert registered["blender"]["command"] == "uvx"

    def test_mcp_add_unity_preset(self, monkeypatch):
        saved_servers = []

        def fake_add_mcp_server(name, **kwargs):
            saved_servers.append((name, kwargs))

        fake_mcp_client = MagicMock()
        fake_mcp_client.register_and_start.return_value = "OK"

        monkeypatch.setattr("command_handler.add_mcp_server", fake_add_mcp_server)
        monkeypatch.setattr("mcp_client.mcp_client", fake_mcp_client)

        _mcp_add(["/mcp", "add", "unity"])

        assert len(saved_servers) == 1
        name, kwargs = saved_servers[0]
        assert name == "unity"
        assert kwargs["command"] == "unity"


class TestBlenderDoctorCheck:
    def test_doctor_blender_online(self, monkeypatch):
        monkeypatch.setattr(
            doctor, "get_mcp_servers",
            lambda: [{"name": "blender", "type": "stdio", "command": "uvx", "args": ["blender-mcp"]}]
        )
        monkeypatch.setattr("shutil.which", lambda cmd: "/fake/uvx")

        fake_socket = MagicMock()
        fake_socket.connect_ex.return_value = 0  # 0 means success / listening
        monkeypatch.setattr(socket, "socket", lambda *args, **kwargs: fake_socket)

        status, detail = doctor._check_mcp()
        assert status == doctor.OK
        assert "Blender online :9876" in detail
        # CP1251 safe
        detail.encode("cp1251", errors="strict")

    def test_doctor_blender_offline(self, monkeypatch):
        monkeypatch.setattr(
            doctor, "get_mcp_servers",
            lambda: [{"name": "blender", "type": "stdio", "command": "uvx", "args": ["blender-mcp"]}]
        )
        monkeypatch.setattr("shutil.which", lambda cmd: "/fake/uvx")

        fake_socket = MagicMock()
        fake_socket.connect_ex.return_value = 10061  # Connection refused
        monkeypatch.setattr(socket, "socket", lambda *args, **kwargs: fake_socket)

        status, detail = doctor._check_mcp()
        assert status == doctor.OK
        assert "Blender not running on :9876" in detail
        detail.encode("cp1251", errors="strict")

    def test_doctor_command_not_in_path(self, monkeypatch):
        monkeypatch.setattr(
            doctor, "get_mcp_servers",
            lambda: [{"name": "blender", "type": "stdio", "command": "uvx", "args": ["blender-mcp"]}]
        )
        monkeypatch.setattr("shutil.which", lambda cmd: None)

        status, detail = doctor._check_mcp()
        assert status == doctor.WARN
        assert "not in PATH" in detail
        detail.encode("cp1251", errors="strict")


class TestBlenderMCPPromptIntegration:
    def test_prompt_includes_blender_tools(self):
        class FakeClient:
            def list_tools(self, name):
                if name == "blender":
                    return [
                        {"name": "execute_blender_code", "description": "Execute Python in Blender"},
                        {"name": "get_scene_info", "description": "Get scene hierarchy"},
                    ]
                return []

        servers = [{"name": "blender", "type": "stdio"}]
        prompt_section = build_mcp_section(servers, client=FakeClient())
        assert "blender" in prompt_section
        assert "execute_blender_code" in prompt_section
        assert "get_scene_info" in prompt_section
