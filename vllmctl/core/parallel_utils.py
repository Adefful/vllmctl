from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.progress import track

def parallel_map_with_progress(func, items, description="Processing...", max_workers=32, show_progress=True):
    """
    Выполняет func(item) для каждого item из items параллельно с прогресс-баром (если show_progress=True).
    Возвращает список результатов в том же порядке, что и items.
    Если func выбрасывает исключение, результатом будет объект исключения.
    """
    results = [None] * len(items)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {executor.submit(func, item): idx for idx, item in enumerate(items)}
        iterator = as_completed(future_to_idx)
        if show_progress and description:
            iterator = track(iterator, total=len(items), description=description)
        for future in iterator:
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                results[idx] = e
    return results 