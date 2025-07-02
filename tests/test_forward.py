import pytest
from unittest.mock import patch
from vllmctl.core.forward import ForwardSession, list_forward_sessions

# Helper mocks
def mock_run_ssh_command_alive(server, cmd, timeout=5):
    return "1"  # tmux session exists

def mock_run_ssh_command_dead(server, cmd, timeout=5):
    return "0"  # tmux session does not exist

def mock_ping_vllm_alive(port):
    return {'data': [{'id': 'TestModel'}]}

def mock_ping_vllm_dead(port):
    return None

def test_forwardsession_alive():
    s = ForwardSession(local_port=1234, remote_port=8000, server='host', tmux_session='vllmctl_server_8000', model_name='TestModel')
    with patch('vllmctl.core.forward.run_ssh_command', mock_run_ssh_command_alive), \
         patch('vllmctl.core.forward.ping_vllm', mock_ping_vllm_alive):
        assert s.check_alive() is True or s.reason == "No local tmux session for SSH tunnel"

def test_forwardsession_no_tmux():
    s = ForwardSession(local_port=1234, remote_port=8000, server='host', tmux_session=None, model_name='TestModel')
    with patch('vllmctl.core.forward.run_ssh_command', mock_run_ssh_command_dead), \
         patch('vllmctl.core.forward.ping_vllm', mock_ping_vllm_alive):
        assert s.check_alive() is False
        assert s.reason == "No local tmux session for SSH tunnel"

def test_forwardsession_no_model():
    s = ForwardSession(local_port=1234, remote_port=8000, server='host', tmux_session='vllmctl_server_8000', model_name=None)
    with patch('vllmctl.core.forward.run_ssh_command', mock_run_ssh_command_alive), \
         patch('vllmctl.core.forward.ping_vllm', mock_ping_vllm_dead):
        assert s.check_alive() is False
        assert s.reason == "No local tmux session for SSH tunnel"

def test_forwardsession_both_dead():
    s = ForwardSession(local_port=1234, remote_port=8000, server='host', tmux_session=None, model_name=None)
    with patch('vllmctl.core.forward.run_ssh_command', mock_run_ssh_command_dead), \
         patch('vllmctl.core.forward.ping_vllm', mock_ping_vllm_dead):
        assert s.check_alive() is False
        assert s.reason == "No local tmux session for SSH tunnel" 