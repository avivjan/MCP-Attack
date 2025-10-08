import os
import time
import math
import asyncio
import psutil
from typing import Optional

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from cpu_sampler import CPUSampler # Needs the existing cpu_sampler.py

# -------------------------------
# Configuration
# -------------------------------
CPU_SAMPLE_SEC: float = 0.5
CPU_ROLLING_SAMPLES: int = 20

# -------------------------------
# Global state
# -------------------------------
app = FastAPI()
CPU_SAMPLER = CPUSampler(CPU_SAMPLE_SEC, CPU_ROLLING_SAMPLES)

def _get_num_fds(process: psutil.Process) -> Optional[int]:
    try:
        return process.num_fds()
    except Exception:
        return None

# The Sampler starts when the app starts
@app.on_event("startup")
def startup_event():
    print("Metrics Sampler starting...")
    CPU_SAMPLER.start()

# The Sampler stops when the app shuts down
@app.on_event("shutdown")
def shutdown_event():
    print("Metrics Sampler stopping...")
    CPU_SAMPLER.stop()


@app.get("/internal-metrics")
def internal_metrics() -> JSONResponse:
    # Measures the metrics server process itself (os.getpid() here is the Uvicorn worker for metrics)
    proc = psutil.Process(os.getpid())
    rss_mb = proc.memory_info().rss / 1e6
    fds_open = _get_num_fds(proc)
    cpu_percent = CPU_SAMPLER.rolling_avg()
    data = {
        # Note: num_streams is NOT here; it will be added by mcp-server
        "rss_mb": rss_mb,
        "fds_open": fds_open,
        "cpu_percent": cpu_percent,
        "ts": time.time(),
    }

    # Sanitize for JSON
    for key, value in list(data.items()):
        if isinstance(value, float) and not math.isfinite(value):
            data[key] = None
    return JSONResponse(data)


if __name__ == "__main__":
    # The metrics server runs on port 8001 internally
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")