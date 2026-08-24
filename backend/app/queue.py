from __future__ import annotations

from typing import Protocol


class AnalysisQueue(Protocol):
    def enqueue(self, job_id: str) -> None: ...


class RQAnalysisQueue:
    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self._queue = None

    @property
    def queue(self):
        if self._queue is None:
            from redis import Redis
            from rq import Queue
            self._queue = Queue("sem-analysis", connection=Redis.from_url(self.redis_url))
        return self._queue

    def enqueue(self, job_id: str) -> None:
        self.queue.enqueue("app.worker_entry.run_job", job_id, job_timeout="2h")
