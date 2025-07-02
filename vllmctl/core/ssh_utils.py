import os
import subprocess
import re
from typing import List, Dict, Optional

SSH_CONFIG_PATH = os.path.expanduser("~/.ssh/config")

def parse_ssh_config():
    """Returns a list of hosts from the ssh-config."""
    hosts = []
    if not os.path.exists(SSH_CONFIG_PATH):
        return hosts
    with open(SSH_CONFIG_PATH) as f:
        for line in f:
            line = line.strip()
            if line.lower().startswith('host '):
                for h in line[5:].split():
                    if h != '*' and not h.startswith('?'):
                        hosts.append(h)
    return hosts

def run_ssh_command(host: str, command: str, timeout=5) -> str:
    try:
        result = subprocess.run([
            "ssh", "-o", "BatchMode=yes", host, command
        ], capture_output=True, text=True, timeout=timeout)
        return result.stdout
    except Exception as e:
        return f"[ssh error: {e}]"

def ping_remote_vllm(host: str, port: int) -> Optional[dict]:
    cmd = f"curl -s --max-time 0.2 http://127.0.0.1:{port}/v1/models"
    out = run_ssh_command(host, cmd)
    if out and out.strip().startswith('{'):
        try:
            import json
            return json.loads(out)
        except Exception:
            return None
    return None

def list_remote_models(host: str, port: int = 8000) -> Dict[int, dict]:
    info = ping_remote_vllm(host, port)
    if info:
        return {port: info}
    return {} 