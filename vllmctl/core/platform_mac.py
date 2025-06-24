import subprocess
import re

def get_listening_ports():
    try:
        result = subprocess.run([
            "lsof", "-nP", "-iTCP", "-sTCP:LISTEN"
        ], capture_output=True, text=True)
        if result.returncode != 0:
            raise FileNotFoundError
        ports = set()
        for line in result.stdout.splitlines():
            m = re.search(r"TCP \*:(\d+) \(LISTEN\)", line)
            if m:
                ports.add(int(m.group(1)))
        return sorted(ports)
    except FileNotFoundError:
        print("[vllmctl] Error: 'lsof' command not found. Please install it with 'brew install lsof'.")
        return []
    except Exception as e:
        print(f"[vllmctl] Error running 'lsof': {e}")
        return [] 