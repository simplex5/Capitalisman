"""Bounded, expiring reference series cache with short failure backoff."""

from collections import OrderedDict
from threading import RLock
import time


class SeriesCache:
    def __init__(self, ttl_seconds, failure_ttl_seconds=30, max_entries=128, clock=None):
        self.ttl = ttl_seconds
        self.failure_ttl = failure_ttl_seconds
        self.max_entries = max_entries
        self.clock = clock or time.monotonic
        self._entries = OrderedDict()
        self._lock = RLock()

    def get_or_load(self, key, loader):
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None and self.clock() < entry[0]:
                self._entries.move_to_end(key)
                return None if entry[1] is None else entry[1].copy(deep=True)
            self._entries.pop(key, None)
        # Do not hold the cache lock across network I/O. Concurrent misses may
        # fetch twice, but a slow provider never blocks unrelated cached reads.
        value = loader()
        stored = None if value is None else value.copy(deep=True)
        with self._lock:
            self._entries[key] = (self.clock() + (self.failure_ttl if stored is None else self.ttl), stored)
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
        return value

    def clear(self):
        with self._lock:
            self._entries.clear()
