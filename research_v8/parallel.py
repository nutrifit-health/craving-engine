"""Изолированные процессы для fit и replay без зависимости от монорепозитория."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

def execute_jobs(worker, jobs: list, workers: int) -> list[dict]:
    """Ограниченный spawn-пул; ошибка сохраняется, а не исключает модель из серии."""
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 4:
        raise ValueError("Допустимо от1 до4 независимых процессов")
    if workers == 1:
        return [worker(job) for job in jobs]
    results = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(worker, job): job for job in jobs}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"Завершено {result['key']}: {result['seconds']:.1f}с", flush=True)
    return sorted(results, key=lambda result: result["key"])


