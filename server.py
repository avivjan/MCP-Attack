import os
import time
import asyncio
import threading
from typing import Dict, Any, List

# ייבוא נדרש ל-FastAPI
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from stream_state import StreamManager, Stream, stream_generator # ודא שהאובייקטים האלה מיובאים כראוי

# ייבוא נדרש למדדים
import psutil 

app = FastAPI()

# פרמטרים גלובליים
STATE_KB: int = int(os.environ.get("STATE_KB", 16))
HEARTBEAT_SEC: float = float(os.environ.get("HEARTBEAT_SEC", 1.0))

# אתחול
STREAM_MANAGER = StreamManager()
STATE_DATA = "A" * (STATE_KB * 1024)

# ----------------------------------------------------
# A. נקודת קצה למדדים (מאוחדת מ-metrics-server הישן)
# ----------------------------------------------------

def get_system_metrics() -> Dict[str, Any]:
    """אוסף מדדי מערכת ואפליקציה"""
    process = psutil.Process(os.getpid())
    return {
        "timestamp": time.time(),
        "cpu_percent": psutil.cpu_percent(interval=None),
        "rss_mb": process.memory_info().rss / (1024 * 1024),
        "fds_open": process.num_fds(),
        "threads_active": threading.active_count(),
        "total_streams": len(STREAM_MANAGER.streams),
    }

@app.get("/metrics")
async def metrics_endpoint():
    """Endpoint לקליינט לאסוף מדדים של המערכת"""
    return get_system_metrics()

# ----------------------------------------------------
# B. נקודת קצה ליצירת זרם (Call)
# ----------------------------------------------------

@app.post("/mcp/call")
async def call_stream(request: Request):
    """יוצר זרם חדש ושומר את הנתונים שלו"""
    stream_id = str(time.time()).replace('.', '')
    
    # יצירת הזרם ושמירת הנתונים
    new_stream = Stream(
        id=stream_id, 
        data=STATE_DATA, 
        heartbeat_interval=HEARTBEAT_SEC
    )
    STREAM_MANAGER.add_stream(new_stream)
    
    # מתחיל את לולאת שליחת הלב פנימית
    asyncio.create_task(stream_generator(stream_id, STREAM_MANAGER))
    
    return {"status": "created", "stream_id": stream_id}

# ----------------------------------------------------
# C. נקודת קצה ל-Streaming (החזרת הזרם הפתוח)
# ----------------------------------------------------

@app.get("/mcp/stream/{stream_id}")
async def stream_data(stream_id: str):
    """מחזיר StreamingResponse ל-SSE"""
    if stream_id not in STREAM_MANAGER.streams:
        return {"error": "Stream not found"}, 404
    
    return StreamingResponse(
        STREAM_MANAGER.streams[stream_id].queue_consumer(),
        media_type="text/event-stream"
    )

# ----------------------------------------------------
# D. נקודת קצה לשליחת הודעות לזרם קיים (Push)
# ----------------------------------------------------

@app.post("/mcp/push/{stream_id}")
async def push_to_stream(stream_id: str, message: Dict[str, Any]):
    """שולח הודעה ספציפית לזרם קיים"""
    if stream_id not in STREAM_MANAGER.streams:
        return {"error": "Stream not found"}, 404
        
    await STREAM_MANAGER.streams[stream_id].send_message(message)
    return {"status": "pushed"}

# ----------------------------------------------------
# E. ניתוק (Close)
# ----------------------------------------------------

@app.post("/mcp/close/{stream_id}")
async def close_stream(stream_id: str):
    """סוגר חיבור באופן יזום ומנקה את המשאבים"""
    if stream_id in STREAM_MANAGER.streams:
        await STREAM_MANAGER.remove_stream(stream_id)
        return {"status": "closed"}
    return {"error": "Stream not found"}, 404