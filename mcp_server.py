import os
import time
import json
import logging
import hashlib
import threading
from datetime import datetime
from typing import Optional, Tuple

import psutil
import requests
from cachetools import TTLCache
from flask import Flask, request, jsonify
from jsonschema import validate as jsonschema_validate, ValidationError


def get_env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def load_schema() -> dict:
    schema_path = os.path.join(os.path.dirname(__file__), "schema.json")
    if not os.path.exists(schema_path):
        # Fallback minimal schema
        return {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                "user": {"type": "string"},
                "action": {"type": "string"},
            },
            "required": ["user", "action"],
        }
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)


app = Flask(__name__)

# Configuration via environment
KEY_SERVER_URL = os.getenv("KEY_SERVER_URL", "http://localhost:5001")
INTERNAL_SVC_URL = os.getenv("INTERNAL_SVC_URL", "http://localhost:5002")
CACHE_TTL = get_env_int("CACHE_TTL", 60)
PORT = get_env_int("PORT", 5000)

METRICS_ENABLED = os.getenv("METRICS_ENABLED", "1") == "1"
METRICS_INTERVAL_MS = get_env_int("METRICS_INTERVAL_MS", 1000)
METRICS_OUT = os.getenv("METRICS_OUT", "/results/cpu_mem.csv")
MCP_LOG_CSV = os.getenv("MCP_LOG_CSV", "/results/mcp_metrics.csv")


# Cache for keys; if TTL<=0, disable cache by using None
key_cache: Optional[TTLCache] = None
if CACHE_TTL > 0:
    key_cache = TTLCache(maxsize=10_000, ttl=CACHE_TTL)


schema = load_schema()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s mcp_server %(message)s",
)


def fetch_key(kid: str) -> Tuple[str, bool]:
    """Fetch key from key_server with optional caching.

    Returns (key, cache_hit).
    """
    cache_hit = False
    if key_cache is not None:
        try:
            if kid in key_cache:
                cache_hit = True
                return key_cache[kid], True
        except Exception:
            # ignore cache errors
            pass

    resp = requests.get(
        f"{KEY_SERVER_URL}/getkey",
        params={"kid": kid},
        timeout=5,
    )
    resp.raise_for_status()
    data = resp.json()
    key = data.get("key", "")

    if key_cache is not None:
        try:
            key_cache[kid] = key
        except Exception:
            pass

    return key, cache_hit


def verify_token(kid: str, token_body: str) -> Tuple[bool, bool]:
    """Simulate token verification using fetched key.

    Returns (is_valid, cache_hit).
    """
    key, cache_hit = fetch_key(kid)
    # Simulate cryptographic verification cost with a hash
    digest = hashlib.sha256()
    digest.update(token_body.encode("utf-8", errors="ignore"))
    digest.update(key.encode("utf-8", errors="ignore"))
    _ = digest.hexdigest()
    # Always return True for simulation purposes
    return True, cache_hit


def write_csv_line(path: str, header: str, line: str) -> None:
    # Ensure directory exists
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    except Exception:
        pass
    file_exists = os.path.exists(path)
    with open(path, "a", encoding="utf-8") as f:
        if not file_exists and header:
            f.write(header + "\n")
        f.write(line + "\n")


def metrics_collector(proc: psutil.Process, interval_ms: int, out_path: str) -> None:
    # Prime cpu_percent measurement
    try:
        proc.cpu_percent(interval=None)
    except Exception:
        pass
    header = "timestamp,cpu_percent,rss_bytes"
    while True:
        try:
            ts = datetime.utcnow().isoformat()
            cpu = proc.cpu_percent(interval=None)
            rss = proc.memory_info().rss
            write_csv_line(out_path, header, f"{ts},{cpu:.2f},{rss}")
        except Exception:
            # Continue even on failure
            pass
        time.sleep(max(0.1, interval_ms / 1000.0))


@app.route("/health")
def health() -> tuple:
    return "ok", 200


@app.route("/handshake", methods=["GET"])
def handshake():
    start = time.perf_counter()
    auth = request.headers.get("Authorization", "")
    kid = request.headers.get("X-KID") or request.args.get("kid", "default")
    token_body = ""
    if auth.startswith("Bearer "):
        token_body = auth.split(" ", 1)[1]
    else:
        token_body = auth
    try:
        is_valid, cache_hit = verify_token(kid, token_body)
        latency_ms = (time.perf_counter() - start) * 1000.0
        logging.info("/handshake kid=%s cache_hit=%s latency_ms=%.2f", kid, cache_hit, latency_ms)
        write_csv_line(
            MCP_LOG_CSV,
            "timestamp,endpoint,latency_ms,cache_hit",
            f"{datetime.utcnow().isoformat()},handshake,{latency_ms:.2f},{int(cache_hit)}",
        )
        return jsonify({"ok": is_valid, "cache_hit": cache_hit, "latency_ms": latency_ms}), 200
    except Exception as e:
        latency_ms = (time.perf_counter() - start) * 1000.0
        logging.exception("/handshake error: %s", e)
        write_csv_line(
            MCP_LOG_CSV,
            "timestamp,endpoint,latency_ms,cache_hit",
            f"{datetime.utcnow().isoformat()},handshake,{latency_ms:.2f},0",
        )
        return jsonify({"error": str(e)}), 500


@app.route("/validate", methods=["POST"])
def validate_json():
    start = time.perf_counter()
    raw = request.get_data(cache=False, as_text=True)
    try:
        data = json.loads(raw)
        jsonschema_validate(instance=data, schema=schema)
        latency_ms = (time.perf_counter() - start) * 1000.0
        logging.info("/validate ok latency_ms=%.2f size_bytes=%d", latency_ms, len(raw.encode("utf-8")))
        return jsonify({"ok": True, "latency_ms": latency_ms}), 200
    except ValidationError as ve:
        latency_ms = (time.perf_counter() - start) * 1000.0
        logging.warning("/validate schema_error latency_ms=%.2f msg=%s", latency_ms, str(ve))
        return jsonify({"ok": False, "error": str(ve), "latency_ms": latency_ms}), 400
    except Exception as e:
        latency_ms = (time.perf_counter() - start) * 1000.0
        logging.exception("/validate error: %s", e)
        return jsonify({"error": str(e), "latency_ms": latency_ms}), 500


@app.route("/fanout", methods=["GET"])
def fanout():
    start = time.perf_counter()
    n_raw = request.args.get("n", "1")
    try:
        n = int(n_raw)
    except Exception:
        n = 1
    successes = 0
    for _ in range(max(0, n)):
        try:
            r = requests.get(f"{INTERNAL_SVC_URL}/touch", timeout=5)
            if r.status_code == 200:
                successes += 1
        except Exception:
            pass
    latency_ms = (time.perf_counter() - start) * 1000.0
    logging.info("/fanout n=%d successes=%d latency_ms=%.2f", n, successes, latency_ms)
    return jsonify({"n": n, "successes": successes, "latency_ms": latency_ms}), 200


def main() -> None:
    if METRICS_ENABLED:
        try:
            proc = psutil.Process(os.getpid())
            t = threading.Thread(
                target=metrics_collector,
                args=(proc, METRICS_INTERVAL_MS, METRICS_OUT),
                daemon=True,
            )
            t.start()
            logging.info("metrics_collector started interval_ms=%d out=%s", METRICS_INTERVAL_MS, METRICS_OUT)
        except Exception as e:
            logging.warning("failed to start metrics_collector: %s", e)

    app.run(host="0.0.0.0", port=PORT, threaded=True)


if __name__ == "__main__":
    main()
