import concurrent.futures
from typing import List, Dict, Optional, Callable, Any
from rich.progress import Progress, TaskID, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
import time
import requests
from .ssh_utils import run_ssh_command, ping_remote_vllm
from .vllm_probe import ping_vllm


class ParallelExecutor:
    """Выполняет задачи параллельно с отображением прогресса"""
    
    def __init__(self, max_workers: int = 10):
        self.max_workers = max_workers
        
    def execute_with_progress(
        self, 
        tasks: List[tuple], 
        worker_func: Callable,
        description: str = "Processing...",
        console=None
    ) -> List[Any]:
        """
        Выполняет задачи параллельно с прогресс-баром
        
        Args:
            tasks: Список кортежей с аргументами для worker_func
            worker_func: Функция для выполнения каждой задачи
            description: Описание для прогресс-бара
            console: Rich console для вывода
            
        Returns:
            Список результатов в том же порядке, что и входящие задачи
        """
        results = [None] * len(tasks)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            
            task_id = progress.add_task(description, total=len(tasks))
            
            def update_progress(future_idx):
                progress.update(task_id, advance=1)
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Запускаем все задачи
                future_to_idx = {}
                for idx, task_args in enumerate(tasks):
                    future = executor.submit(worker_func, *task_args)
                    future_to_idx[future] = idx
                
                # Собираем результаты по мере завершения
                for future in concurrent.futures.as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        results[idx] = future.result()
                    except Exception as e:
                        results[idx] = None
                    update_progress(idx)
                    
        return results


def parallel_ping_vllm_ports(ports: List[int]) -> Dict[int, dict]:
    """Параллельно проверяет vLLM API на портах"""
    def ping_port(port):
        return port, ping_vllm(port)
    
    executor = ParallelExecutor(max_workers=20)
    port_tasks = [(port,) for port in ports]
    results = executor.execute_with_progress(
        port_tasks, 
        ping_port, 
        "Checking ports..."
    )
    
    models = {}
    for port, info in results:
        if info:
            models[port] = info
    
    return models


def parallel_check_remote_servers(hosts: List[str], remote_port: int = 8000, debug: bool = False) -> List[tuple]:
    """Параллельно проверяет удаленные серверы"""
    def check_server(host):
        try:
            models = ping_remote_vllm(host, remote_port)
            if models:
                model_name = models['data'][0]['id'] if models.get('data') and models['data'] else 'unknown'
                return (host, remote_port, model_name, None)
            elif debug:
                return (host, remote_port, "-", None)
            else:
                return None
        except Exception as e:
            if debug:
                return (host, remote_port, f"Error: {e}", e)
            return None
    
    executor = ParallelExecutor(max_workers=15)
    host_tasks = [(host,) for host in hosts]
    results = executor.execute_with_progress(
        host_tasks,
        check_server,
        "Checking servers..."
    )
    
    return [r for r in results if r is not None]


def parallel_gpu_scan(hosts: List[str]) -> Dict[str, tuple]:
    """Параллельно сканирует GPU на серверах"""
    def get_gpu_stats(host):
        try:
            # Получаем утилизацию GPU
            out_util = run_ssh_command(
                host, 
                "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits", 
                timeout=3
            )
            utils = [int(x) for x in out_util.strip().splitlines() if x.strip().isdigit()]
            
            # Получаем использование памяти
            out_mem = run_ssh_command(
                host, 
                "nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits", 
                timeout=3
            )
            mems = []
            for line in out_mem.strip().splitlines():
                parts = [int(x) for x in line.strip().split(',') if x.strip().isdigit()]
                if len(parts) == 2 and parts[1] > 0:
                    mems.append(parts[0] / parts[1] * 100)
            
            import numpy as np
            avg_util = float(np.mean(utils)) if utils else None
            avg_mem = float(np.mean(mems)) if mems else None
            return host, (avg_util, avg_mem)
        except Exception:
            return host, (None, None)
    
    executor = ParallelExecutor(max_workers=10)
    host_tasks = [(host,) for host in hosts]
    results = executor.execute_with_progress(
        host_tasks,
        get_gpu_stats,
        "Scanning GPU utilization on hosts..."
    )
    
    return dict(results)


def parallel_vllm_metrics(ports: List[int]) -> Dict[int, tuple]:
    """Параллельно получает метрики vLLM"""
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
                        
            return port, (waiting, running, swapped, prompt_throughput, generation_throughput)
        except Exception:
            return port, (None, None, None, None, None)
    
    executor = ParallelExecutor(max_workers=20)
    port_tasks = [(port,) for port in ports]
    results = executor.execute_with_progress(
        port_tasks,
        get_metrics,
        "Scanning ports for vLLM models..."
    )
    
    return dict(results)


def parallel_auto_forward(hosts: List[str], remote_port: int, local_range: tuple, no_kill: bool, debug: bool) -> List[tuple]:
    """Параллельно проверяет серверы для auto-forward"""
    from .ssh_utils import list_remote_models
    
    def check_host_for_forward(host):
        try:
            models = list_remote_models(host, port=remote_port)
            has_model = bool(models)
            model_name = None
            if has_model:
                info = list(models.values())[0]
                model_name = info['data'][0]['id'] if info.get('data') and info['data'] else 'unknown'
            return host, has_model, model_name
        except Exception as e:
            return host, False, None
    
    executor = ParallelExecutor(max_workers=15)
    host_tasks = [(host,) for host in hosts]
    results = executor.execute_with_progress(
        host_tasks,
        check_host_for_forward,
        "Checking hosts for models..."
    )
    
    return results