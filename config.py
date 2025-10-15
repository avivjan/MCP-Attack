import os
from typing import List


# -------------------------------
# Configuration (simple constants)
# -------------------------------
STATE_KB: int = int(os.environ.get("STATE_KB", 16))
HEARTBEAT_SEC: float = float(os.environ.get("HEARTBEAT_SEC", 1.0))
DURATION_SEC: float = float(os.environ.get("DURATION_SEC", 20))
CPU_SAMPLE_SEC: float = 0.5
CPU_ROLLING_SAMPLES: int = 20  # ~10 seconds window at 0.5s/sample
METRIC_POLL_SEC: float = 2.0
N_SWEEP: List[int] = [10, 30, 60, 100, 500, 1000]


