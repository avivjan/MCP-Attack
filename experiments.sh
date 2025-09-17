#!/usr/bin/env bash
set -euo pipefail

# Usage: ./experiments.sh [A|B|C|D|E|all]
EXPERIMENT="${1:-all}"
RUN_TIME="${RUN_TIME:-60s}"
SPAWN_RATE="${SPAWN_RATE:-50}"
RESULTS_DIR="$(pwd)/results"
HOST="http://mcp_server:5000"

mkdir -p "$RESULTS_DIR"

wait_for_health() {
  local url="${1:-http://localhost:5000/health}"
  echo "Waiting for $url ..."
  for i in {1..60}; do
    if curl -fsS "$url" >/dev/null 2>&1; then
      echo "Service healthy."
      return 0
    fi
    sleep 1
  done
  echo "Timed out waiting for $url" >&2
  return 1
}

compose_up() {
  echo "Restarting services with DELAY_MS=$DELAY_MS CACHE_TTL=$CACHE_TTL INTERNAL_TOUCH_MS=$INTERNAL_TOUCH_MS KEY_SIZE_BYTES=$KEY_SIZE_BYTES"
  DELAY_MS=${DELAY_MS:-0} CACHE_TTL=${CACHE_TTL:-60} INTERNAL_TOUCH_MS=${INTERNAL_TOUCH_MS:-10} KEY_SIZE_BYTES=${KEY_SIZE_BYTES:-1024} \
  docker compose up -d --force-recreate --no-deps key_server internal_service mcp_server
  wait_for_health "http://localhost:5000/health"
}

run_locust() {
  local out_dir="$1"; shift
  local scenario="$1"; shift
  mkdir -p "$RESULTS_DIR/$out_dir"
  # Remove rolling metrics so we get per-experiment copies later
  rm -f "$RESULTS_DIR/cpu_mem.csv" "$RESULTS_DIR/mcp_metrics.csv" || true
  echo "Running locust scenario=$scenario output=$out_dir users=$USERS time=$RUN_TIME"
  docker compose run --rm -T locust \
    locust -f locustfile.py --headless \
      -u "$USERS" -r "$SPAWN_RATE" -t "$RUN_TIME" \
      --host "$HOST" \
      --csv "/results/$out_dir/locust" --csv-full-history \
      --scenario "$scenario" "$@"
  # Copy per-experiment process metrics
  if [[ -f "$RESULTS_DIR/cpu_mem.csv" ]]; then
    mv "$RESULTS_DIR/cpu_mem.csv" "$RESULTS_DIR/$out_dir/cpu_mem.csv"
  fi
  if [[ -f "$RESULTS_DIR/mcp_metrics.csv" ]]; then
    mv "$RESULTS_DIR/mcp_metrics.csv" "$RESULTS_DIR/$out_dir/mcp_metrics.csv"
  fi
}

run_A() {
  # Baseline: DELAY_MS=0, CACHE_TTL=60, fixed kid, JSON small; concurrency [50,100,200]
  export DELAY_MS=0 CACHE_TTL=60 INTERNAL_TOUCH_MS=10 KEY_SIZE_BYTES=1024
  compose_up
  for USERS in 50 100 200; do
    run_locust "A_baseline/handshake_u${USERS}" handshake --kid-mode fixed --jwt-bytes 512
    run_locust "A_baseline/validate_u${USERS}" validate --json-bytes 2048 --json-depth 5
  done
}

run_B() {
  # Expensive-auth (cache OFF): DELAY_MS=200, CACHE_TTL=0; fixed vs random kid; concurrency [50,100,200,400]
  export DELAY_MS=200 CACHE_TTL=0 INTERNAL_TOUCH_MS=10 KEY_SIZE_BYTES=1024
  compose_up
  for USERS in 50 100 200 400; do
    run_locust "B_expensive_auth_fixed_u${USERS}" handshake --kid-mode fixed --jwt-bytes 512
    run_locust "B_expensive_auth_random_u${USERS}" handshake --kid-mode random --jwt-bytes 512
  done
}

run_C() {
  # Cache-bust comparison: DELAY_MS=200, CACHE_TTL=60; fixed then random kid; same concurrencies as B
  export DELAY_MS=200 CACHE_TTL=60 INTERNAL_TOUCH_MS=10 KEY_SIZE_BYTES=1024
  compose_up
  for USERS in 50 100 200 400; do
    run_locust "C_cache_bust_fixed_u${USERS}" handshake --kid-mode fixed --jwt-bytes 512
    run_locust "C_cache_bust_random_u${USERS}" handshake --kid-mode random --jwt-bytes 512
  done
}

run_D() {
  # JSON parsing: DELAY_MS=0, CACHE_TTL=60; sizes [2KB,10KB,50KB], depths [5,20,50]; concurrency [50,100,200]
  export DELAY_MS=0 CACHE_TTL=60 INTERNAL_TOUCH_MS=10 KEY_SIZE_BYTES=1024
  compose_up
  for USERS in 50 100 200; do
    for SIZE in 2048 10240 51200; do
      for DEPTH in 5 20 50; do
        run_locust "D_json_users${USERS}_${SIZE}B_d${DEPTH}" validate --json-bytes "$SIZE" --json-depth "$DEPTH"
      done
    done
  done
}

run_E() {
  # Fan-out amplification: INTERNAL_TOUCH_MS=10; n in [1,5,20,50]; moderate concurrency 100
  export DELAY_MS=0 CACHE_TTL=60 INTERNAL_TOUCH_MS=10 KEY_SIZE_BYTES=1024
  compose_up
  USERS=100
  for N in 1 5 20 50; do
    run_locust "E_fanout_n${N}_u${USERS}" fanout --fanout-n "$N"
  done
}

case "$EXPERIMENT" in
  A) run_A ;;
  B) run_B ;;
  C) run_C ;;
  D) run_D ;;
  E) run_E ;;
  all)
    run_A
    run_B
    run_C
    run_D
    run_E
    ;;
  *)
    echo "Unknown experiment: $EXPERIMENT" >&2
    exit 1
    ;;
 esac

echo "All experiments complete. Results in $RESULTS_DIR"
