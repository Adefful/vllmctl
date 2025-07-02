import subprocess
import re
import requests
import psutil
import sys
from vllmctl.core.parallel_utils import parallel_map_with_progress

TMUX_PREFIX = "vllmctl_"

def get_listening_ports():
    try:
        result = subprocess.run([
            "ss", "-tulpen"
        ], capture_output=True, text=True)
        if result.returncode != 0:
            raise FileNotFoundError
        ports = set()
        for line in result.stdout.splitlines():
            m = re.search(r"127.0.0.1:(\d+)", line)
            if m:
                ports.add(int(m.group(1)))
        return sorted(ports)
    except FileNotFoundError:
        print("[vllmctl] Error: 'ss' command not found. Please install 'iproute2' (Linux) or use 'lsof' on Mac. Example: sudo apt install iproute2")
        print("[vllmctl] On Mac, you can use: brew install lsof. Support for Mac will be added soon.")
        return []
    except Exception as e:
        print(f"[vllmctl] Error running 'ss': {e}")
        return []

def ping_vllm(port):
    try:
        r = requests.get(f"http://127.0.0.1:{port}/v1/models", timeout=0.2)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

def get_ssh_forwardings():
    try:
        result = subprocess.run(["ps", "aux"], capture_output=True, text=True)
        if result.returncode != 0:
            raise FileNotFoundError
        forwards = {}
        for line in result.stdout.splitlines():
            if "ssh -N -L" in line:
                m = re.search(r"ssh -N -L (\d+):localhost:(\d+) ([^ ]+)", line)
                if m:
                    local_port = int(m.group(1))
                    remote_port = int(m.group(2))
                    host = m.group(3)
                    pid = int(line.split()[1])
                    forwards[local_port] = (host, remote_port, pid)
        return forwards
    except FileNotFoundError:
        print("[vllmctl] Error: 'ps' command not found. Please install 'procps' (Linux) or ensure 'ps' is available in your PATH.")
        return {}
    except Exception as e:
        print(f"[vllmctl] Error running 'ps': {e}")
        return {}

def get_tmux_sessions():
    try:
        result = subprocess.run(["tmux", "ls"], capture_output=True, text=True)
        if result.returncode != 0:
            raise FileNotFoundError
        sessions = []
        for line in result.stdout.splitlines():
            name = line.split(':')[0]
            sessions.append(name)
        return sessions
    except FileNotFoundError:
        print("[vllmctl] Error: 'tmux' command not found. Please install 'tmux'. Example: sudo apt install tmux or brew install tmux")
        return []
    except Exception as e:
        print(f"[vllmctl] Error running 'tmux ls': {e}")
        return []

def list_local_models():
    ports = get_listening_ports()
    ssh_forwards = get_ssh_forwardings()
    tmux_sessions = get_tmux_sessions()
    models = {}

    def get_port_info(port):
        info = ping_vllm(port)
        if not info:
            return None
        entry = {'model': info, 'port': port}
        model_name = info['data'][0]['id'] if info.get('data') and info['data'] else 'unknown'
        if port in ssh_forwards:
            host, rport, pid = ssh_forwards[port]
            entry['forwarded'] = True
            entry['server'] = host
            entry['remote_port'] = rport
            entry['ssh_pid'] = pid
            tmux_name = f"{TMUX_PREFIX}{host}_{rport}"
            entry['tmux'] = tmux_name if tmux_name in tmux_sessions else None
        else:
            entry['forwarded'] = False
            entry['server'] = None
            entry['remote_port'] = None
            entry['ssh_pid'] = None
            entry['tmux'] = None
        entry['model_name'] = model_name
        return (port, entry)

    results = parallel_map_with_progress(get_port_info, ports, description="Scanning local models...")
    for res in results:
        if res:
            port, entry = res
            models[port] = entry
    return models