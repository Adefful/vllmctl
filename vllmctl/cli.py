import typer
import re as regexlib
from vllmctl.core.vllm_probe import list_local_models, get_listening_ports, ping_vllm, get_tmux_sessions
from vllmctl.core.ssh_utils import parse_ssh_config, list_remote_models, run_ssh_command
from vllmctl.core.forward import auto_forward_ports
from vllmctl.core.launcher import launch_vllm
from vllmctl.core.async_utils import (
    parallel_ping_vllm_ports, 
    parallel_check_remote_servers,
    parallel_gpu_scan,
    parallel_vllm_metrics,
    parallel_auto_forward
)
from rich.progress import track
from rich.table import Table
from rich.console import Console
import subprocess
import psutil
import requests
import time
import threading
import time as time_mod
from rich.live import Live
import re
import numpy as np
from rich.text import Text

app = typer.Typer()

@app.command()
def list_local():
    """Show local vllm-models (by ports, including forwarded)."""
    console = Console()
    ports = get_listening_ports()
    tmux_sessions = get_tmux_sessions()
    
    # Используем параллельную проверку портов
    models = parallel_ping_vllm_ports(ports)
    
    table = Table(title="Local vllm models")
    table.add_column("Server")
    table.add_column("Remote\nport")
    table.add_column("Local\nport")
    table.add_column("Status")
    table.add_column("Model")
    if not models:
        typer.echo("No available vllm models on local ports.")
    else:
        local_models = list_local_models()
        for port, info in models.items():
            entry = local_models.get(port, {})
            model_name = entry.get('model_name', '-')
            local_port = str(port)
            # Default values for non-forwarded models
            server = "-"
            remote_port = "-"
            status = "Local launch"
            if entry.get('forwarded'):
                # If forwarded, fill in server and remote_port
                server = entry.get('server', '-')
                remote_port = str(entry.get('remote_port', '-'))
                status = "Forwarded"
                if entry.get('tmux') and not entry.get('ssh_pid'):
                    status = f"{entry['tmux']}"
            else:
                # If not forwarded, optionally show tmux session in status
                for tmux_name in tmux_sessions:
                    if model_name in tmux_name:
                        status = f"tmux: {tmux_name}"
                        break
            table.add_row(server, remote_port, local_port, status, model_name)
    console.print(table)

@app.command()
def list_remote(
    host_regex: str = typer.Option(None, help="Regex for filtering servers by name"),
    debug: bool = typer.Option(False, help="Show detailed information and empty servers"),
    remote_port: int = typer.Option(8000, help="Port for checking on remote servers (default 8000)")
):
    """Show vllm-models on all servers from ssh-config."""
    console = Console()
    hosts = parse_ssh_config()
    if host_regex:
        hosts = [h for h in hosts if regexlib.search(host_regex, h)]
    if not hosts:
        typer.echo("No suitable hosts in ~/.ssh/config")
        return
        
    # Используем параллельную проверку серверов
    results = parallel_check_remote_servers(hosts, remote_port, debug)
    
    table = Table(title="Remote vllm models")
    table.add_column("Server")
    table.add_column("Remote\nport")
    table.add_column("Model")
    
    for host, port, model_name, error in results:
        table.add_row(host, str(port), model_name)
    
    console.print(table)

@app.command()
def auto_forward(
    host_regex: str = typer.Option(None, help="Regex for filtering servers by name"),
    remote_port: int = typer.Option(8000, help="Port for checking on remote servers (default 8000)"),
    local_range: str = typer.Option("16100-16199", help="Range of local ports for forwarding (e.g., 16100-16199)"),
    no_kill: bool = typer.Option(False, help="Do not kill forwarding if model not found"),
    debug: bool = typer.Option(False, help="Detailed output")
):
    """Automatically forward ports with models to local machine."""
    console = Console()
    hosts = parse_ssh_config()
    if host_regex:
        hosts = [h for h in hosts if regexlib.search(host_regex, h)]
    if not hosts:
        typer.echo("No suitable hosts in ~/.ssh/config")
        return
    try:
        l1, l2 = map(int, local_range.split('-'))
        local_range_tuple = (l1, l2)
    except Exception:
        typer.echo("Error in local_range format. Example: 16100-16199")
        return
        
    # Используем параллельную проверку серверов для определения наличия моделей
    host_model_data = parallel_auto_forward(hosts, remote_port, local_range_tuple, no_kill, debug)
    
    # Теперь обрабатываем результаты и выполняем форвардинг
    results = auto_forward_ports(
        hosts,
        remote_port=remote_port,
        local_range=local_range_tuple,
        no_kill=no_kill,
        debug=debug
    )
    table = Table(title="Auto-forward results")
    table.add_column("Server")
    table.add_column("Remote\nport")
    table.add_column("Local\nport")
    table.add_column("Status")
    table.add_column("Model")
    for host, rport, lport, status, model in results:
        show_model = False
        if status.startswith("Forwarded") or status.startswith("Already forwarded") or status.startswith("duplicate session"):
            show_model = True
        elif debug and model:
            show_model = True
        table.add_row(
            str(host),
            str(rport),
            str(lport) if lport else "-",
            status,
            model if (show_model and model) else "-"
        )
    console.print(table)

@app.command()
def launch(
    server: str = typer.Option(..., help="Server name (from ssh-config)"),
    model: str = typer.Option("Qwen/Qwen2.5-Coder-32B-Instruct", help="Model name for vllm serve"),
    tensor_parallel_size: int = typer.Option(8, help="tensor-parallel-size for vllm serve"),
    remote_port: int = typer.Option(8000, help="Port on server for vllm serve"),
    local_range: str = typer.Option("16100-16199", help="Range of local ports for forwarding (e.g., 16100-16199)"),
    conda_env: str = typer.Option("vllm_env", help="Conda-environment for running vllm on server"),
    timeout: int = typer.Option(600, help="Maximum waiting time for vllm start (sec)"),
    lifetime: str = typer.Option(None, help="Maximum lifetime for vllm process (e.g., 10m, 2h, 1d, 30s)")
):
    """Run vllm on server and forward port to local machine through tmux. Can attach to any session."""
    console = Console()
    l1, l2 = map(int, local_range.split('-'))
    local_port = launch_vllm(
        server=server,
        model=model,
        tensor_parallel_size=tensor_parallel_size,
        remote_port=remote_port,
        local_range=(l1, l2),
        conda_env=conda_env,
        timeout=timeout,
        lifetime=lifetime,
        console=console
    )
    if local_port is None:
        raise typer.Exit(1)

@app.command()
def tmux_forwards(
    tmux_prefix: str = typer.Option("vllmctl_", help="Prefix for tmux sessions to search for forwards")
):
    """Show all tmux-forwards (vllmctl_*) and status: is there a model on the port. Only parses session names."""
    console = Console()
    result = subprocess.run(["tmux", "ls"], capture_output=True, text=True)
    sessions = []
    local_ports = []
    session_data = {}
    
    for line in result.stdout.splitlines():
        if line.startswith(tmux_prefix):
            name = line.split(':')[0]
            sessions.append(name)
            # Parse session name: vllmctl_{host}_{remote_port}_{local_port}
            m = regexlib.match(r"vllmctl_(.+)_(\d+)_(\d+)", name)
            if m:
                server = m.group(1)
                remote_port = m.group(2)
                local_port = int(m.group(3))
                local_ports.append(local_port)
                session_data[local_port] = {
                    'session': name,
                    'server': server,
                    'remote_port': remote_port
                }
    
    # Параллельно проверяем все локальные порты
    if local_ports:
        models_dict = parallel_ping_vllm_ports(local_ports)
    else:
        models_dict = {}
    
    table = Table(title="Tmux-forwards status")
    table.add_column("Tmux session")
    table.add_column("Server")
    table.add_column("Remote port")
    table.add_column("Local port")
    table.add_column("Model on port?")
    
    for session in sessions:
        # Parse session name: vllmctl_{host}_{remote_port}_{local_port}
        m = regexlib.match(r"vllmctl_(.+)_(\d+)_(\d+)", session)
        if m:
            server = m.group(1)
            remote_port = m.group(2)
            local_port = int(m.group(3))
            
            # Используем результаты параллельной проверки
            model_status = "-"
            if local_port in models_dict:
                model_info = models_dict[local_port]
                if model_info and 'data' in model_info and model_info['data']:
                    model_status = model_info['data'][0].get('id', 'model exists')
                elif model_info:
                    model_status = 'model exists'
                else:
                    model_status = "no model"
            else:
                model_status = "no model"
                
            table.add_row(session, server, remote_port, str(local_port), model_status)
        else:
            table.add_row(session, "-", "-", "-", "invalid session name")
    
    console.print(table)

@app.command()
def clean_tmux_forwards(
    tmux_prefix: str = typer.Option("vllmctl_", help="Prefix for tmux sessions to search for forwards")
):
    """Delete all tmux-sessions vllmctl_*, where there is no ssh-forward or model does not ping."""
    result = subprocess.run(["tmux", "ls"], capture_output=True, text=True)
    sessions = []
    for line in result.stdout.splitlines():
        if line.startswith(tmux_prefix):
            name = line.split(':')[0]
            sessions.append(name)
    killed = []
    for session in sessions:
        pid_out = subprocess.run(
            ["tmux", "list-panes", "-t", session, "-F", "#{pane_pid}"],
            capture_output=True, text=True
        )
        found = False
        for pid_str in pid_out.stdout.splitlines():
            try:
                pid = int(pid_str)
                proc = psutil.Process(pid)
                for child in proc.children(recursive=True):
                    if child.name() == "ssh":
                        cmdline = " ".join(child.cmdline())
                        m = regexlib.search(r"-L\s*(\d+):localhost:(\d+)", cmdline)
                        if m:
                            local_port = int(m.group(1))
                            model_info = ping_vllm(local_port)
                            if not model_info:
                                subprocess.run(["tmux", "kill-session", "-t", session])
                                killed.append((session, local_port, "no model"))
                                found = True
                            else:
                                found = True
            except Exception:
                continue
        if not found:
            subprocess.run(["tmux", "kill-session", "-t", session])
            killed.append((session, "-", "no ssh-forward"))
    if killed:
        table = Table(title="Deleted tmux-sessions")
        table.add_column("Tmux session")
        table.add_column("Local port")
        table.add_column("Reason")
        for session, port, reason in killed:
            table.add_row(session, str(port), reason)
        console = Console()
        console.print(table)
    else:
        print("No tmux-sessions to delete.")

@app.command()
def kill_tmux(
    session: str = typer.Argument(..., help="Name of tmux-session to kill (e.g., vllmctl_server_port)")
):
    """Kill tmux-session by name."""
    result = subprocess.run(["tmux", "kill-session", "-t", session], capture_output=True, text=True)
    console = Console()
    if result.returncode == 0:
        console.print(f"[green]Session {session} killed[/green]")
    else:
        console.print(f"[red]Error killing {session}:[/red] {result.stderr}")

@app.command()
def vllm_queue_top(
    refresh: float = typer.Option(1.0, help="Refresh interval in seconds"),
    history: int = typer.Option(30, help="Number of points for mini-graph (history)")
):
    """Show real-time vLLM queue status for all local ports (like nvtop)."""
    console = Console()
    ports = get_listening_ports()
    
    # Используем параллельное сканирование для первоначального поиска vLLM портов
    models_dict = parallel_ping_vllm_ports(ports)
    vllm_ports = list(models_dict.keys())
    port_models = {}
    
    for port in vllm_ports:
        info = models_dict[port]
        if info and 'data' in info and info['data']:
            port_models[port] = info['data'][0].get('id', '-')
        else:
            port_models[port] = '-'
    
    if not vllm_ports:
        console.print("No running vLLM instances found on local ports.")
        return

    spinner_frames = ['|', '/', '-', '\\']
    spinner_idx = [0]
    # History buffer for each port and metric
    metric_history = {port: {'waiting': [], 'running': [], 'swapped': [], 'prompt_throughput': [], 'generation_throughput': []} for port in vllm_ports}

    def get_metrics(port):
        try:
            r = requests.get(f"http://127.0.0.1:{port}/metrics", timeout=0.5)
            lines = r.text.splitlines()
            waiting = running = swapped = None
            prompt_throughput = None
            generation_throughput = None
            for line in lines:
                if 'vllm:num_requests_waiting' in line:
                    try:
                        waiting = float(line.strip().split()[-1])
                    except Exception:
                        pass
                if 'vllm:num_requests_running' in line:
                    try:
                        running = float(line.strip().split()[-1])
                    except Exception:
                        pass
                if 'vllm:num_requests_swapped' in line:
                    try:
                        swapped = float(line.strip().split()[-1])
                    except Exception:
                        pass
                if 'vllm:avg_prompt_throughput_toks_per_s' in line:
                    try:
                        prompt_throughput = float(line.strip().split()[-1])
                    except Exception:
                        pass
                if 'vllm:avg_generation_throughput_toks_per_s' in line:
                    try:
                        generation_throughput = float(line.strip().split()[-1])
                    except Exception:
                        pass
            return waiting, running, swapped, prompt_throughput, generation_throughput
        except Exception:
            return None, None, None, None, None

    def sparkline(data, width=history):
        # Simple ASCII sparkline for small numbers
        if not data:
            return ' ' * width
        minv = min(data)
        maxv = max(data)
        if maxv == minv:
            return '▁' * len(data)
        chars = '▁▂▃▄▅▆▇█'
        res = ''
        for v in data[-width:]:
            idx = int((v - minv) / (maxv - minv) * (len(chars) - 1))
            res += chars[idx]
        return res.rjust(width)

    def make_table():
        frame = spinner_frames[spinner_idx[0] % len(spinner_frames)]
        spinner_idx[0] += 1
        table = Table(title=f"{frame} vLLM Queue Status (refreshes every {refresh:.1f}s)")
        table.add_column("Local Port")
        table.add_column("Model")
        table.add_column("Waiting")
        table.add_column("Running")
        table.add_column("Wait graph")
        table.add_column("Run graph")
        table.add_column("Prompt TPT")
        table.add_column("Gen TPT")
        for port in vllm_ports:
            model = port_models.get(port, '-')
            waiting, running, swapped, prompt_throughput, generation_throughput = get_metrics(port)
            # Update history
            for key, val in zip(['waiting', 'running', 'prompt_throughput', 'generation_throughput'], [waiting, running, prompt_throughput, generation_throughput]):
                if val is not None:
                    metric_history[port][key].append(val)
                    if len(metric_history[port][key]) > history:
                        metric_history[port][key] = metric_history[port][key][-history:]
            table.add_row(
                str(port),
                model,
                str(int(waiting) if waiting is not None else '-'),
                str(int(running) if running is not None else '-'),
                sparkline(metric_history[port]['waiting']),
                sparkline(metric_history[port]['running']),
                str(f"{prompt_throughput:.1f}" if prompt_throughput is not None else '-'),
                str(f"{generation_throughput:.1f}" if generation_throughput is not None else '-')
            )
        return table

    with Live(make_table(), refresh_per_second=1/refresh if refresh > 0 else 1, console=console) as live:
        try:
            while True:
                live.update(make_table())
                time_mod.sleep(refresh)
        except KeyboardInterrupt:
            pass

@app.command()
def gpu_idle_top(
    refresh: float = typer.Option(0.5, help="Refresh interval in seconds"),
    history: int = typer.Option(30, help="Number of points for mini-graph (history)"),
    host_regex: str = typer.Option(None, help="Regex to filter hosts from ssh config")
):
    """Show real-time GPU utilization and memory for all servers in ssh config, sorted by idle (lowest utilization first)."""
    console = Console()
    hosts = parse_ssh_config()
    if host_regex:
        hosts = [h for h in hosts if re.search(host_regex, h)]
    if not hosts:
        console.print("No hosts found in ssh config.")
        return
    
    spinner_frames = ['|', '/', '-', '\\']
    spinner_idx = [0]
    util_history = {host: [] for host in hosts}
    mem_history = {host: [] for host in hosts}

    # Начальное параллельное сканирование с прогресс-баром
    gpu_stats = parallel_gpu_scan(hosts)
    
    reachable_hosts = []
    for host, (util, mem) in gpu_stats.items():
        if util is not None:
            util_history[host].append(util)
        if mem is not None:
            mem_history[host].append(mem)
        if util is not None or mem is not None:
            reachable_hosts.append(host)
        else:
            util_history[host] = []
            mem_history[host] = []

    # Only keep reachable hosts for live updates
    hosts = reachable_hosts
    util_history = {h: util_history[h] for h in hosts}
    mem_history = {h: mem_history[h] for h in hosts}

    def get_gpu_stats(host):
        try:
            out_util = run_ssh_command(host, "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits", timeout=3)
            utils = [int(x) for x in out_util.strip().splitlines() if x.strip().isdigit()]
            out_mem = run_ssh_command(host, "nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits", timeout=3)
            mems = []
            for line in out_mem.strip().splitlines():
                parts = [int(x) for x in line.strip().split(',') if x.strip().isdigit()]
                if len(parts) == 2 and parts[1] > 0:
                    mems.append(parts[0] / parts[1] * 100)
            avg_util = float(np.mean(utils)) if utils else None
            avg_mem = float(np.mean(mems)) if mems else None
            return avg_util, avg_mem
        except Exception:
            return None, None

    def sparkline(data, width=history):
        if not data:
            return ' ' * width
        minv = min(data)
        maxv = max(data)
        if maxv == minv:
            return '▁' * len(data)
        chars = '▁▂▃▄▅▆▇█'
        res = ''
        for v in data[-width:]:
            idx = int((v - minv) / (maxv - minv) * (len(chars) - 1))
            res += chars[idx]
        return res.rjust(width)

    def color_value(val):
        if val is None:
            return Text("-", style="dim")
        style = ""
        if val > 90:
            style = "bold red"
        elif val > 50:
            style = "bold orange3"
        elif val > 0:
            style = "bold yellow"
        else:
            style = "dim"
        return Text(f"{val:.1f}", style=style)

    def make_table():
        frame = spinner_frames[spinner_idx[0] % len(spinner_frames)]
        spinner_idx[0] += 1
        table = Table(title=f"{frame} GPU Idle Top (refreshes every {refresh:.1f}s)")
        table.add_column("Host")
        table.add_column("Util (%)")
        table.add_column("Util Graph")
        table.add_column("Mem (%)")
        table.add_column("Mem Graph")
        # Sort hosts by last utilization (lowest first, None last)
        sorted_hosts = sorted(hosts, key=lambda h: (util_history[h][-1] if util_history[h] else float('inf')))
        for host in sorted_hosts:
            util, mem = get_gpu_stats(host)
            if util is not None:
                util_history[host].append(util)
                if len(util_history[host]) > history:
                    util_history[host] = util_history[host][-history:]
            if mem is not None:
                mem_history[host].append(mem)
                if len(mem_history[host]) > history:
                    mem_history[host] = mem_history[host][-history:]
            table.add_row(
                host,
                color_value(util),
                sparkline(util_history[host]),
                color_value(mem),
                sparkline(mem_history[host])
            )
        return table

    with Live(make_table(), refresh_per_second=1/refresh if refresh > 0 else 1, console=console) as live:
        try:
            while True:
                live.update(make_table())
                time_mod.sleep(refresh)
        except KeyboardInterrupt:
            pass
