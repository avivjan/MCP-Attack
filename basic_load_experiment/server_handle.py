import threading
import uvicorn


class ServerHandle:
    def __init__(self, server: uvicorn.Server, thread: threading.Thread) -> None:
        self.server = server
        self.thread = thread


