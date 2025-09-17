import os
import time
from flask import Flask


def get_env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


app = Flask(__name__)

PORT = get_env_int("PORT", 5002)
TOUCH_MS = get_env_int("TOUCH_MS", get_env_int("INTERNAL_TOUCH_MS", 10))


@app.route("/health")
def health() -> tuple:
    return "ok", 200


@app.route("/touch")
def touch():
    if TOUCH_MS > 0:
        time.sleep(TOUCH_MS / 1000.0)
    return "ok", 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, threaded=True)
