import asyncio
import math
import os
import json
import socket
import threading
import time
import uuid
from typing import Any, AsyncGenerator, Dict, Optional, Tuple

import psutil
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from config import HEARTBEAT_SEC, CPU_SAMPLE_SEC, CPU_ROLLING_SAMPLES, STATE_KB
from cpu_sampler import CPUSampler
from server_handle import ServerHandle
from stream_state import StreamState


app = FastAPI()


STREAMS: Dict[str, StreamState] = {}
STREAMS_LOCK = threading.Lock()


CPU_SAMPLER = CPUSampler(CPU_SAMPLE_SEC, CPU_ROLLING_SAMPLES)
@app.on_event("startup")
def _start_sampler() -> None:
    try:
        CPU_SAMPLER.start()
    except Exception:
        pass


@app.on_event("shutdown")
def _stop_sampler() -> None:
    try:
        CPU_SAMPLER.stop()
    except Exception:
        pass


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
    # Retain a small per-stream memory pad so RSS reflects N when desired
    if STATE_KB > 0:
        try:
            # Attach as attribute; dataclass allows extra attrs
            state.memory_pad = bytearray(STATE_KB * 1024)
        except Exception:
            pass
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


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_server() -> Tuple[ServerHandle, str]:
    host = os.getenv("SERVER_HOST", "127.0.0.1")
    port_env = os.getenv("SERVER_PORT")
    try:
        port_parsed = int(port_env) if port_env is not None else 0
    except Exception:
        port_parsed = 0
    port = port_parsed if port_parsed > 0 else _find_free_port()
    config = uvicorn.Config(app, host=host, port=port, log_level="error")
    server = uvicorn.Server(config)

    def _run() -> None:
        # Start CPU sampler shortly before server run
        CPU_SAMPLER.start()
        server.run()

    thread = threading.Thread(target=_run, name="uvicorn-server", daemon=True)
    thread.start()
    # If binding to 0.0.0.0 (inside container), expose localhost for the orchestrator
    visible_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    base_url = f"http://{visible_host}:{port}"
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


