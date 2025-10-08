# stream_state.py
import asyncio
import time
from typing import Dict, Any, List

# ----------------------------------------------------
# 1. המחלקה Stream (מייצגת חיבור SSE בודד)
# ----------------------------------------------------
class Stream:
    def __init__(self, id: str, data: str, heartbeat_interval: float):
        self.id = id
        self.data = data # הנתונים שנשמרים לכל זרם (משפיע על זיכרון)
        self.heartbeat_interval = heartbeat_interval
        self.queue = asyncio.Queue()
        self.is_active = True
        
    async def send_message(self, message: Dict[str, Any]):
        """הוספת הודעה ספציפית לתור ה-SSE."""
        event_data = f"data: {message}\n\n"
        await self.queue.put(event_data)

    async def queue_consumer(self):
        """פונקציית צרכן המחזירה את ה-StreamingResponse."""
        while self.is_active:
            try:
                # מחכה להודעות חדשות בתור
                event_data = await asyncio.wait_for(self.queue.get(), timeout=15)
                yield event_data
                self.queue.task_done()
            except asyncio.TimeoutError:
                # אם אין הודעה, שולח Heartbeat שקט למניעת ניתוק
                yield f"data: heartbeat\n\n"
            except asyncio.CancelledError:
                # אם הלקוח מתנתק
                self.is_active = False
                break

# ----------------------------------------------------
# 2. המחלקה StreamManager (מנהלת את כל הזרמים הפתוחים)
# ----------------------------------------------------
class StreamManager:
    def __init__(self):
        self.streams: Dict[str, Stream] = {}
        
    def add_stream(self, stream: Stream):
        self.streams[stream.id] = stream
        
    async def remove_stream(self, stream_id: str):
        if stream_id in self.streams:
            self.streams[stream_id].is_active = False
            del self.streams[stream_id]

    async def broadcast(self, message: Dict[str, Any]):
        """שולח הודעה לכל הזרמים הפתוחים"""
        tasks = [stream.send_message(message) for stream in self.streams.values()]
        await asyncio.gather(*tasks)

# ----------------------------------------------------
# 3. פונקציית מחולל ה-Heartbeat (רצה ברקע)
# ----------------------------------------------------
async def stream_generator(stream_id: str, manager: StreamManager):
    """מטפלת בשליחת הודעות Heartbeat קבועות לזרם ספציפי."""
    while stream_id in manager.streams and manager.streams[stream_id].is_active:
        try:
            stream = manager.streams[stream_id]
            
            # ההודעה מכילה את גודל הנתונים השמורים
            heartbeat_message = {
                "type": "heartbeat",
                "id": stream.id,
                "size_kb": len(stream.data) / 1024
            }
            await stream.send_message(heartbeat_message)
            
            # ממתין לפי מרווח ה-Heartbeat המוגדר
            await asyncio.sleep(stream.heartbeat_interval)
            
        except KeyError:
            # הזרם נסגר או הוסר
            break
        except Exception:
            # טפל בשגיאות אחרות ונקה
            await manager.remove_stream(stream_id)
            break