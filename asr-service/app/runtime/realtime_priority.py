"""实时推理优先门控。

离线任务在每个可抢占批次前调用 wait_realtime_clear；实时推理进入
realtime_section 后，离线会在下一个批次边界让路。单次模型调用内部不可抢占。
"""
from contextlib import contextmanager
import threading
import time


_DEFAULT_POLL_INTERVAL = 0.1


class RealtimePriorityGate:
    def __init__(self, *, enabled: bool = True, wait_timeout: float | None = None):
        self.enabled = enabled
        self.wait_timeout = wait_timeout
        self._active = 0
        self._cond = threading.Condition()

    @property
    def realtime_active(self) -> int:
        with self._cond:
            return self._active

    @contextmanager
    def realtime_section(self):
        if not self.enabled:
            yield
            return
        with self._cond:
            self._active += 1
        try:
            yield
        finally:
            with self._cond:
                self._active = max(0, self._active - 1)
                if self._active == 0:
                    self._cond.notify_all()

    def wait_realtime_clear(
        self,
        timeout: float | None = None,
        *,
        cancelled=None,
        poll_interval: float = _DEFAULT_POLL_INTERVAL,
    ) -> bool:
        if not self.enabled:
            return True
        max_wait = self.wait_timeout if timeout is None else timeout
        with self._cond:
            if cancelled is None:
                if self._active > 0:
                    self._cond.wait_for(lambda: self._active == 0, timeout=max_wait)
                return self._active == 0

            deadline = None if max_wait is None else time.monotonic() + max_wait
            while self._active > 0:
                if cancelled():
                    return False
                wait_time = poll_interval
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return True
                    wait_time = min(wait_time, remaining)
                self._cond.wait(timeout=wait_time)
            return True
