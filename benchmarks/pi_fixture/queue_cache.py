"""Queue/cache example: cancellation is advisory; old work may finish."""
from dataclasses import dataclass
from pathlib import Path
import json
@dataclass
class Job:
    key: str
    generation: int
    cancelled: bool = False
class Queue:
    def __init__(self, cache):
        self.cache = cache
        self.generations = {}
        self.active = {}
    def submit(self, key):
        generation = self.generations.get(key, 0) + 1
        self.generations[key] = generation
        job = Job(key, generation)
        self.active[key] = job
        return job
    def cancel(self, job):
        job.cancelled = True
        if self.active.get(job.key) is job:
            del self.active[job.key]
    def complete(self, job, value):
        self.cache.put(job.key, value)
        if self.active.get(job.key) is job:
            del self.active[job.key]
class DiskCache:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
    def put(self, key, value):
        path = self.root / (key + '.json')
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(value), encoding='utf-8')
        temp.replace(path)
    def get(self, key):
        return json.loads((self.root / (key + '.json')).read_text())
