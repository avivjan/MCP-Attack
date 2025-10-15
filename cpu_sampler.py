import os
import threading
from collections import deque

import psutil


class CPUSampler:
    def __init__(self, sample_interval_sec: float, rolling_samples: int) -> None:
        self._process = psutil.Process(os.getpid())
        self._interval = sample_interval_sec
        self._samples = deque(maxlen=rolling_samples)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="cpu-sampler", daemon=True)

    def start(self) -> None:
        try:
            self._process.cpu_percent(interval=None)
        except Exception:
            pass
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=self._interval * 3)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                value = self._process.cpu_percent(interval=self._interval)
            except Exception:
                value = float("nan")
            with self._lock:
                self._samples.append(value)

    def rolling_avg(self) -> float:
        with self._lock:
            if not self._samples:
                return float("nan")
            vals = [v for v in self._samples if v == v]
            if not vals:
                return float("nan")
            return float(sum(vals) / len(vals))


