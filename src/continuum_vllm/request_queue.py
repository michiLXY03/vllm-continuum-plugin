"""Continuum waiting queue implemented outside the vLLM package."""

from collections import deque
from collections.abc import Callable, Iterable, Iterator

from vllm.v1.core.sched.request_queue import FCFSRequestQueue, RequestQueue
from vllm.v1.request import Request

ContinuumRanker = Callable[[Request], tuple[int, int, float, float, str]]


class ContinuumRequestQueue(FCFSRequestQueue):
    """Deque storage whose next request is selected by a live ranker."""

    def __init__(self, requests: Iterable[Request] = ()) -> None:
        super().__init__(requests)
        self._ranker: ContinuumRanker | None = None

    def set_ranker(self, ranker: ContinuumRanker) -> None:
        self._ranker = ranker

    def rank(self, request: Request) -> tuple[int, int, float, float, str]:
        if self._ranker is None:
            return (
                1,
                1,
                request.arrival_time,
                request.arrival_time,
                request.request_id,
            )
        return self._ranker(request)

    def peek_request(self) -> Request:
        if not self:
            raise IndexError("peek from an empty queue")
        if self._ranker is None:
            return self[0]
        return min(deque.__iter__(self), key=self._ranker)

    def pop_request(self) -> Request:
        request = self.peek_request()
        self.remove(request)
        return request

    def prepend_requests(self, requests: RequestQueue) -> None:
        self.extend(requests)

    def __iter__(self) -> Iterator[Request]:
        if self._ranker is None:
            return deque.__iter__(self)
        return iter(sorted(deque.__iter__(self), key=self._ranker))
