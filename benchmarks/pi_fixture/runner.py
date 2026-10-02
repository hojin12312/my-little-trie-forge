"""Synchronous bounded runner; retain job input ordering."""
from concurrent.futures import ThreadPoolExecutor
class Runner:
    def __init__(self, workers=2):
        self.workers = workers
    def run(self, functions):
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = [executor.submit(fn) for fn in functions]
            return [future.result() for future in futures]
