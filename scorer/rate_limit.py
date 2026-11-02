import contextlib
import threading
import time


class RPMLimiter:
    """Thread-safe request-rate limiter for one process."""

    def __init__(self, rpm: int):
        self.interval = 60.0 / rpm if rpm > 0 else 0.0
        self.lock = threading.Lock()
        self.last_request = 0.0

    @contextlib.contextmanager
    def slot(self):
        if self.interval:
            with self.lock:
                delay = self.interval - (time.monotonic() - self.last_request)
                if delay > 0:
                    time.sleep(delay)
                self.last_request = time.monotonic()
        yield
