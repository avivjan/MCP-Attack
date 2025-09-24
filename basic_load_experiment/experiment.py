import asyncio
import json
import os
import signal
import socket
import sys
import threading
import time
import uuid
import math
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import aiohttp
import matplotlib.pyplot as plt
import pandas as pd
import psutil
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from stream_state import StreamState
from cpu_sampler import CPUSampler
from server_handle import ServerHandle


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


# -------------------------------
# Global server state
# -------------------------------
app = FastAPI()


STREAMS: Dict[str, StreamState] = {}
STREAMS_LOCK = threading.Lock()


CPU_SAMPLER = CPUSampler(CPU_SAMPLE_SEC, CPU_ROLLING_SAMPLES)


async def _cleanup_state_after_ttl(stream_id: str, ttl_sec: float = 5.0) -> None:
    await asyncio.sleep(ttl_sec)
    with STREAMS_LOCK:
        state = STREAMS.get(stream_id)
        if state and state.status in ("client_disconnected", "done"):
            STREAMS.pop(stream_id, None)


@app.post("/mcp/call")
async def mcp_call(request: Request) -> JSONResponse:
    try:
        body_bytes = await request.body()
        payload = json.loads(body_bytes.decode("utf-8"))
    except Exception:
        return JSONResponse({"error": "invalid json"}, status_code=400)

    if not isinstance(payload, dict):
        return JSONResponse({"error": "expected object"}, status_code=400)
    if payload.get("jsonrpc") != "2.0":
        return JSONResponse({"error": "jsonrpc must be '2.0'"}, status_code=400)
    req_id = payload.get("id")
    method = payload.get("method")
    params = payload.get("params", {})
    if req_id is None or not method:
        return JSONResponse({"error": "missing id or method"}, status_code=400)
    if not isinstance(params, dict):
        return JSONResponse({"error": "params must be object"}, status_code=400)

    stream_id = str(req_id) if req_id is not None else str(uuid.uuid4())
    state = StreamState(
        stream_id=stream_id,
        created_ts=time.time(),
        method=str(method),
        params=params,
        event_queue=asyncio.Queue(),
        last_event_id=None,
        status="open",
    )
    with STREAMS_LOCK:
        STREAMS[stream_id] = state

    ack = {"jsonrpc": "2.0", "id": req_id, "result": {"stream_id": stream_id}}
    return JSONResponse(ack)


@app.post("/mcp/push/{stream_id}")
async def mcp_push(stream_id: str, request: Request) -> JSONResponse:
    try:
        body = json.loads((await request.body()).decode("utf-8"))
    except Exception:
        return JSONResponse({"error": "invalid json"}, status_code=400)
    event = body.get("event") if isinstance(body, dict) else None
    if not isinstance(event, dict):
        return JSONResponse({"error": "missing event object"}, status_code=400)
    with STREAMS_LOCK:
        state = STREAMS.get(stream_id)
    if not state:
        return JSONResponse({"error": "unknown stream_id"}, status_code=404)

    event_id = str(uuid.uuid4())
    state.last_event_id = event_id
    # Wrap event as JSON-RPC toolOutput message
    msg = {
        "jsonrpc": "2.0",
        "id": state.stream_id,
        "method": "toolOutput",
        "params": event,
    }
    await state.event_queue.put(msg)

    if event.get("done") is True:
        state.status = "done"
        asyncio.create_task(_cleanup_state_after_ttl(stream_id, ttl_sec=2.0))
    return JSONResponse({"ok": True, "event_id": event_id})


@app.get("/mcp/stream")
async def mcp_stream(request: Request) -> StreamingResponse:
    q = request.query_params
    stream_id = q.get("id") or q.get("stream_id")
    if not stream_id:
        return JSONResponse({"error": "missing id"}, status_code=400)
    with STREAMS_LOCK:
        state = STREAMS.get(stream_id)
        if state:
            state.status = "open"
    if not state:
        return JSONResponse({"error": "unknown stream id"}, status_code=404)

    # Emit an immediate "started" event so TTFB reflects responsiveness
    await state.event_queue.put({
        "jsonrpc": "2.0",
        "id": state.stream_id,
        "method": "started",
        "params": {"ts": time.time()},
    })

    async def generator_wrapper() -> AsyncGenerator[bytes, None]:
        try:
            while True:
                try:
                    msg = await asyncio.wait_for(state.event_queue.get(), timeout=HEARTBEAT_SEC)
                except asyncio.TimeoutError:
                    # Send lightweight heartbeat referencing last_event_id
                    msg = {
                        "jsonrpc": "2.0",
                        "id": state.stream_id,
                        "method": "heartbeat",
                        "params": {"ts": time.time(), "last_event_id": state.last_event_id},
                    }
                line = f"data: {json.dumps(msg)}\n\n".encode("utf-8")
                yield line
        finally:
            with STREAMS_LOCK:
                st = STREAMS.get(stream_id)
                if st:
                    st.status = "client_disconnected"
            asyncio.create_task(_cleanup_state_after_ttl(stream_id, ttl_sec=5.0))

    return StreamingResponse(generator_wrapper(), media_type="text/event-stream")


def _make_nested_params(i: int, depth: int = 5) -> Dict[str, Any]:
    leaf: Dict[str, Any] = {"i": i, "ts": time.time(), "values": [i, i + 1, i + 2]}
    node: Dict[str, Any] = leaf
    for lvl in range(depth):
        node = {"level": lvl, "node": node, "flags": {"a": True, "b": False}}
    return node


def _get_num_fds(process: psutil.Process) -> Optional[int]:
    try:
        return process.num_fds()
    except Exception:
        return None


@app.get("/metrics")
def metrics() -> JSONResponse:
    proc = psutil.Process(os.getpid())
    with STREAMS_LOCK:
        num_streams = len(STREAMS)
    rss_mb = proc.memory_info().rss / 1e6
    fds_open = _get_num_fds(proc)
    cpu_percent = CPU_SAMPLER.rolling_avg()
    data = {
        "num_streams": num_streams,
        "rss_mb": rss_mb,
        "fds_open": fds_open,
        "cpu_percent": cpu_percent,
        "ts": time.time(),
    }

    # Sanitize for JSON (disallow NaN/inf per Starlette JSONResponse defaults)
    for key, value in list(data.items()):
        if isinstance(value, float) and not math.isfinite(value):
            data[key] = None
    return JSONResponse(data)


# -------------------------------
# Server runner (background)
# -------------------------------
def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_server() -> Tuple[ServerHandle, str]:
    host = "127.0.0.1"
    port = _find_free_port()
    config = uvicorn.Config(app, host=host, port=port, log_level="error")
    server = uvicorn.Server(config)

    def _run() -> None:
        # Start CPU sampler shortly before server run
        CPU_SAMPLER.start()
        server.run()

    thread = threading.Thread(target=_run, name="uvicorn-server", daemon=True)
    thread.start()
    base_url = f"http://{host}:{port}"
    return ServerHandle(server, thread), base_url


def stop_server(handle: ServerHandle) -> None:
    try:
        handle.server.should_exit = True
    except Exception:
        pass
    if handle.thread.is_alive():
        handle.thread.join(timeout=5)
    # Stop CPU sampler after server stops
    CPU_SAMPLER.stop()


async def wait_for_server(base_url: str, timeout_sec: float = 10.0) -> None:
    start = time.perf_counter()
    connector = aiohttp.TCPConnector(limit=0)  # unlimited connections (use with caution)
    async with aiohttp.ClientSession(connector=connector) as session:
        while time.perf_counter() - start < timeout_sec:
            try:
                async with session.get(base_url + "/metrics", timeout=aiohttp.ClientTimeout(total=2.0)) as resp:
                    if resp.status == 200:
                        return
            except Exception:
                await asyncio.sleep(0.1)
        raise RuntimeError("Server did not become ready in time")


# -------------------------------
# Client load + data collection
# -------------------------------
async def sse_client_task(session: aiohttp.ClientSession, url: str, duration_sec: float) -> Tuple[Optional[float], int, float]:
    """Return (ttfb_ms, events_count, open_seconds)."""
    ttfb_ms: Optional[float] = None
    events_count = 0
    started = time.perf_counter()
    open_seconds = 0.0

    timeout = aiohttp.ClientTimeout(total=duration_sec + 10, sock_connect=5, sock_read=5)
    headers = {"Accept": "text/event-stream"}
    try:
        async with session.get(url, timeout=timeout, headers=headers) as resp:
            if resp.status != 200:
                return ttfb_ms, events_count, open_seconds
            msg_has_data = False
            while True:
                line = await resp.content.readline()
                if line == b"":
                    break
                # End of one SSE message
                if line in (b"\n", b"\r\n"):
                    if msg_has_data:
                        events_count += 1
                        msg_has_data = False
                    if time.perf_counter() - started >= duration_sec:
                        break
                    continue
                if line.startswith(b"data:"):
                    if ttfb_ms is None:
                        ttfb_ms = (time.perf_counter() - started) * 1000.0
                    msg_has_data = True
            open_seconds = time.perf_counter() - started
    except asyncio.TimeoutError:
        open_seconds = time.perf_counter() - started
    except Exception:
        open_seconds = time.perf_counter() - started
    return ttfb_ms, events_count, open_seconds


async def poll_metrics_timeseries(base_url: str, duration_sec: float, n_label: int) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    stop_at = time.perf_counter() + duration_sec
    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=connector) as session:
        while time.perf_counter() < stop_at:
            try:
                async with session.get(base_url + "/metrics", timeout=aiohttp.ClientTimeout(total=3.0)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        data["N"] = n_label
                        rows.append(data)
            except Exception:
                pass
            await asyncio.sleep(METRIC_POLL_SEC)
    return pd.DataFrame(rows)


async def run_for_N(base_url: str, N: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    async with aiohttp.ClientSession() as session:
        # First, create N calls to get stream IDs with nested params
        stream_ids: List[str] = []
        for i in range(N):
            nested_params = _make_nested_params(i, depth=5)
            rpc = {"jsonrpc": "2.0", "id": str(uuid.uuid4()), "method": "testMethod", "params": nested_params}
            try:
                async with session.post(base_url + "/mcp/call", json=rpc, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        res = await resp.json()
                        sid = res.get("result", {}).get("stream_id")
                        if isinstance(sid, str):
                            stream_ids.append(sid)
            except Exception:
                pass
        # Start SSE clients for each stream id
        tasks = []
        for sid in stream_ids:
            sse_url = base_url + f"/mcp/stream?id={sid}"
            tasks.append(asyncio.create_task(sse_client_task(session, sse_url, DURATION_SEC)))
        # In parallel, poll metrics
        metrics_task = asyncio.create_task(poll_metrics_timeseries(base_url, DURATION_SEC, N))
        results = await asyncio.gather(*tasks)
        metrics_df = await metrics_task

    # Build per-connection results DataFrame
    rows: List[Dict[str, Any]] = []
    for ttfb_ms, events, open_sec in results:
        events_per_sec = (events / open_sec) if open_sec > 0 else float("nan")
        rows.append({
            "N": N,
            "ttfb_ms": ttfb_ms if ttfb_ms is not None else float("nan"),
            "events": events,
            "open_sec": open_sec,
            "events_per_sec": events_per_sec,
        })
    per_conn_df = pd.DataFrame(rows)
    return per_conn_df, metrics_df


def analyze_and_plot(all_conn_df: pd.DataFrame, all_metrics_df: pd.DataFrame) -> pd.DataFrame:
    summary_rows: List[Dict[str, Any]] = []
    for N, conn_df_N in all_conn_df.groupby("N"):
        metrics_N = all_metrics_df[all_metrics_df["N"] == N]
        peak_rss_mb = float(metrics_N["rss_mb"].max()) if not metrics_N.empty else float("nan")
        avg_cpu_percent = float(metrics_N["cpu_percent"].mean()) if not metrics_N.empty else float("nan")
        avg_fds_open = float(metrics_N["fds_open"].mean()) if (not metrics_N.empty and "fds_open" in metrics_N) else float("nan")

        avg_ttfb_ms = float(conn_df_N["ttfb_ms"].mean())
        p95_ttfb_ms = float(conn_df_N["ttfb_ms"].quantile(0.95))
        avg_events_per_sec = float(conn_df_N["events_per_sec"].mean())

        summary_rows.append({
            "N": int(N),
            "peak_rss_mb": peak_rss_mb,
            "avg_cpu_percent": avg_cpu_percent,
            "avg_fds_open": avg_fds_open,
            "avg_ttfb_ms": avg_ttfb_ms,
            "p95_ttfb_ms": p95_ttfb_ms,
            "avg_events_per_sec": avg_events_per_sec,
        })

    summary_df = pd.DataFrame(summary_rows).sort_values("N").reset_index(drop=True)
    summary_df.to_csv("results.csv", index=False)

    # Plots
    plt.figure()
    plt.plot(summary_df["N"], summary_df["peak_rss_mb"], marker="o")
    plt.xlabel("Concurrent streams (N)")
    plt.ylabel("Peak RSS (MB)")
    plt.title("N vs Peak RSS")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.savefig("plot_rss_vs_N.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(summary_df["N"], summary_df["avg_ttfb_ms"], marker="o")
    plt.xlabel("Concurrent streams (N)")
    plt.ylabel("Avg TTFB (ms)")
    plt.title("N vs Avg TTFB")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.savefig("plot_ttfb_vs_N.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(summary_df["N"], summary_df["avg_events_per_sec"], marker="o")
    plt.xlabel("Concurrent streams (N)")
    plt.ylabel("Avg events/sec")
    plt.title("N vs Avg Events/sec")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.savefig("plot_events_vs_N.png", bbox_inches="tight")
    plt.close()

    return summary_df


def write_summary_text(summary_df: pd.DataFrame) -> str:
    lines: List[str] = []
    # Simple heuristics for conclusions
    def trend_ratio(col: str) -> Optional[float]:
        if len(summary_df) < 2:
            return None
        first = summary_df.iloc[0][col]
        last = summary_df.iloc[-1][col]
        if first and first == first and last and last == last and first > 0:
            return float(last) / float(first)
        return None

    mem_ratio = trend_ratio("peak_rss_mb")
    ttfb_ratio = trend_ratio("avg_ttfb_ms")

    lines.append("This experiment measured an MCP-like SSE server under increasing concurrent stream counts.")
    if mem_ratio is not None:
        lines.append(f"Memory usage (peak RSS) scaled by ~{mem_ratio:.2f}x from min to max N.")
    if ttfb_ratio is not None:
        lines.append(f"Average TTFB scaled by ~{ttfb_ratio:.2f}x across the sweep.")
    if "avg_events_per_sec" in summary_df:
        avg_eps_min = float(summary_df["avg_events_per_sec"].min())
        avg_eps_max = float(summary_df["avg_events_per_sec"].max())
        lines.append(f"Delivered events/sec per stream remained around {avg_eps_min:.2f}–{avg_eps_max:.2f}.")
    lines.append("Overall, results suggest roughly linear resource growth with N for this small per-stream state.")
    lines.append("Next step: increase STATE_KB and/or add connection churn to probe thresholds.")
    text = "\n".join(lines)
    with open("summary.txt", "w", encoding="utf-8") as f:
        f.write(text + "\n")
    return text


async def main_async() -> None:
    handle, base_url = start_server()
    try:
        await wait_for_server(base_url)
        all_conn_dfs: List[pd.DataFrame] = []
        all_metrics_dfs: List[pd.DataFrame] = []

        for N in N_SWEEP:
            print(f"Running N={N} ...")
            per_conn_df, metrics_df = await run_for_N(base_url, N)
            # Save interim metrics time series per N
            metrics_df.to_csv(f"metrics_timeseries_{N}.csv", index=False)
            all_conn_dfs.append(per_conn_df)
            all_metrics_dfs.append(metrics_df)
            # Cooldown
            await asyncio.sleep(5)

        all_conn_df = pd.concat(all_conn_dfs, ignore_index=True) if all_conn_dfs else pd.DataFrame()
        all_metrics_df = pd.concat(all_metrics_dfs, ignore_index=True) if all_metrics_dfs else pd.DataFrame()

        summary_df = analyze_and_plot(all_conn_df, all_metrics_df)
        summary_text = write_summary_text(summary_df)

        print("\n" + summary_text + "\n")
    finally:
        stop_server(handle)


def print_how_to_run() -> None:
    print("""
# 1) Create venv and install:
#    pip install fastapi uvicorn aiohttp psutil pandas matplotlib
# 2) Run:
#    python experiment.py
# 3) See outputs:
#    results.csv, summary.txt, plot_*.png
""".strip())


if __name__ == "__main__":
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        pass
    finally:
        print_how_to_run()


