import os
import time
import asyncio
import threading
from typing import Dict, Any, List
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import psutil

app = FastAPI()

# Configuration
STATE_KB: int = int(os.environ.get("STATE_KB", 16))
HEARTBEAT_SEC: float = float(os.environ.get("HEARTBEAT_SEC", 1.0))

# Simple Stream class for demonstration
class Stream:
    def __init__(self, stream_id: str, data: str, heartbeat_interval: float):
        self.id = stream_id
        self.data = data
        self.heartbeat_interval = heartbeat_interval
        self.queue = asyncio.Queue()
    
    async def queue_consumer(self):
        while True:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                yield f"data: {{'heartbeat': {time.time()}}}\n\n"
            except GeneratorExit:
                break

class StreamManager:
    def __init__(self):
        self.streams: Dict[str, Stream] = {}
    
    def add_stream(self, stream: Stream):
        self.streams[stream.id] = stream
    
    async def remove_stream(self, stream_id: str):
        if stream_id in self.streams:
            del self.streams[stream_id]

# Global state
STREAM_MANAGER = StreamManager()
STATE_DATA = "A" * (STATE_KB * 1024)

# תיקון מס' 1: טיפול בשגיאת 'num_fds' ב-Windows
def get_system_metrics() -> Dict[str, Any]:
    process = psutil.Process(os.getpid())
    
    try:
        fds_open = process.num_fds()
    except AttributeError:
        # אם אין תמיכה (למשל Windows), מגדירים None
        fds_open = None 
    
    return {
        "timestamp": time.time(),
        "cpu_percent": psutil.cpu_percent(interval=None),
        "rss_mb": process.memory_info().rss / (1024 * 1024),
        "fds_open": fds_open,
        "threads_active": threading.active_count(),
        "total_streams": len(STREAM_MANAGER.streams),
    }

@app.get("/metrics")
async def metrics_endpoint():
    return get_system_metrics()

@app.post("/mcp/call")
async def call_stream(request: Request):
    stream_id = str(time.time()).replace('.', '')
    # תיקון מס' 2: שינוי 'id' ל-'stream_id' כדי להתאים ל-Stream.__init__
    new_stream = Stream(
        stream_id=stream_id,
        data=STATE_DATA,
        heartbeat_interval=HEARTBEAT_SEC
    )
    STREAM_MANAGER.add_stream(new_stream)
    return {"status": "created", "stream_id": stream_id}

@app.get("/mcp/stream/{stream_id}")
async def stream_data(stream_id: str):
    if stream_id not in STREAM_MANAGER.streams:
        # כדאי להחזיר תגובה FastAPI חוקית עם קוד סטטוס
        return {"error": "Stream not found"}, 404
    
    async def event_generator():
        while stream_id in STREAM_MANAGER.streams:
            await asyncio.sleep(HEARTBEAT_SEC)
            yield f"data: {{'id': '{stream_id}', 'time': {time.time()}}}\n\n"
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/mcp/push/{stream_id}")
async def push_to_stream(stream_id: str, message: Dict[str, Any]):
    if stream_id not in STREAM_MANAGER.streams:
        return {"error": "Stream not found"}, 404
    return {"status": "pushed"}

@app.post("/mcp/close/{stream_id}")
async def close_stream(stream_id: str):
    if stream_id in STREAM_MANAGER.streams:
        await STREAM_MANAGER.remove_stream(stream_id)
        return {"status": "closed"}
    return {"error": "Stream not found"}, 404