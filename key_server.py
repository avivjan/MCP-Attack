import os
import time
from flask import Flask, request, jsonify


def get_env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


app = Flask(__name__)

PORT = get_env_int("PORT", 5001)
DELAY_MS = get_env_int("DELAY_MS", 0)
KEY_SIZE_BYTES = get_env_int("KEY_SIZE_BYTES", 1024)


@app.route("/health")
def health() -> tuple:
    return "ok", 200


@app.route("/getkey")
def getkey():
    kid = request.args.get("kid", "default")
    if DELAY_MS > 0:
        time.sleep(DELAY_MS / 1000.0)
    key = ("K" * max(0, KEY_SIZE_BYTES))[:KEY_SIZE_BYTES]
    return jsonify({"kid": kid, "key": key})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, threaded=True)
