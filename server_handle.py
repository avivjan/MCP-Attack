import threading
import uvicorn


class ServerHandle:
    """
    A simple container to hold the running Uvicorn server instance 
    and its thread, used for managing shutdown.
    """
    def __init__(self, server: uvicorn.Server, thread: threading.Thread) -> None:
        self.server = server
        self.thread = thread