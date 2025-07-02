import pytest
from typer.testing import CliRunner
from vllmctl.cli import app
import subprocess
import sys

runner = CliRunner()

@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda x: None)

@pytest.fixture(autouse=True)
def no_real_requests(monkeypatch):
    monkeypatch.setattr("requests.get", lambda *a, **kw: type("Resp", (), {"status_code": 200, "text": "{\"models\": []}"})())

@pytest.fixture(autouse=True)
def fake_get_listening_ports(monkeypatch):
    monkeypatch.setattr("vllmctl.core.vllm_probe.get_listening_ports", lambda: set())

@pytest.fixture(autouse=True)
def fake_subprocess_run(monkeypatch):
    class FakeCompleted:
        def __init__(self, stdout="", stderr="", returncode=0):
            self.stdout = stdout
            self.stderr = stderr
            self.returncode = returncode
    def fake_run(cmd, *a, **kw):
        if "tmux" in cmd:
            if "ls" in cmd:
                return FakeCompleted(stdout="vllmctl_vllm_server1_8000: 1 windows\n")
            if "list-panes" in cmd:
                return FakeCompleted(stdout="1234\n")
            if "kill-session" in cmd:
                return FakeCompleted(stdout="", returncode=0)
        if "ssh" in cmd[0]:
            return FakeCompleted(stdout="", returncode=0)
        return FakeCompleted()
    monkeypatch.setattr(subprocess, "run", fake_run)


def test_serve_success():
    result = runner.invoke(app, [
        "serve", "--server", "server1", "--model", "Qwen/Qwen2.5-Coder-32B-Instruct", "--timeout", "2"
    ])
    assert result.exit_code == 0
    assert "✓ VLLM is ready!" in result.output
    assert "tmux attach -t vllmctl_server_8000" in result.output


def test_serve_timeout(monkeypatch):
    monkeypatch.setattr("requests.get", lambda *a, **kw: (_ for _ in ()).throw(Exception("fail")))
    result = runner.invoke(app, [
        "serve", "--server", "server1", "--model", "Qwen/Qwen2.5-Coder-32B-Instruct", "--timeout", "1"])
    assert result.exit_code != 0
    assert "VLLM API did not start" in result.output


def test_tmux_forwards():
    result = runner.invoke(app, ["tmux-forwards"])
    assert result.exit_code == 0
    assert "Tmux-forwards status" in result.output


def test_kill_tmux():
    result = runner.invoke(app, ["kill-tmux", "vllmctl_vllm_server1_8000"])
    assert result.exit_code == 0
    assert "Session vllmctl_vllm_server1_8000 killed" in result.output


def test_clean_tmux_forwards():
    result = runner.invoke(app, ["clean-tmux-forwards"])
    assert result.exit_code == 0 or "Нет tmux-сессий для удаления." in result.output 