from dataclasses import dataclass
from typing import Optional, Dict, Any
import asyncio


@dataclass
class StreamState:
    stream_id: str
    created_ts: float
    method: str
    params: Dict[str, Any]
    event_queue: asyncio.Queue
    last_event_id: Optional[str]
    status: str  # "open", "done", "client_disconnected"


