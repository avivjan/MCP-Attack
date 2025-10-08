import os
import threading
from collections import deque
from typing import Deque

import psutil


class CPUSampler:
    """
    Samples the current process's CPU utilization percentage at fixed intervals
    and maintains a rolling average.
    """
    def __init__(self, sample_interval_sec: float, rolling_samples: int) -> None:
        self._process = psutil.Process(os.getpid())
        self._interval = sample_interval_sec
        self._samples: Deque[float] = deque(maxlen=rolling_samples)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="cpu-sampler", daemon=True)

    def start(self) -> None:
        """Starts the sampler thread."""
        # Initial call to cpu_percent to establish baseline
        try:
            self._process.cpu_percent(interval=None)
        except Exception:
            pass
        self._thread.start()

    def stop(self) -> None:
        """Stops the sampler thread."""
        self._stop.set()
        if self._thread.is_alive():
            # Wait for the thread to stop gracefully
            self._thread.join(timeout=self._interval * 3)

    def _run(self) -> None:
        """The main loop for the sampling thread."""
        while not self._stop.is_set():
            try:
                # cpu_percent reads the time since the last call or self._interval
                # Note: psutil returns a percentage per CPU, so for N cores, it can exceed 100%
                value = self._process.cpu_percent(interval=self._interval)
            except Exception:
                value = float("nan")
                
            with self._lock:
                self._samples.append(value)

    def rolling_avg(self) -> float:
        """Returns the average CPU percentage of the last N samples."""
        with self._lock:
            if not self._samples:
                return float("nan")
            
            # Filter out NaN values before calculating the mean
            # `v == v` is a common Python trick to check if a float is NOT NaN
            valid_samples = [v for v in self._samples if v == v]
            if not valid_samples:
                 return float("nan")

            return sum(valid_samples) / len(valid_samples)