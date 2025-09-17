# MCP-Attack: Practical MCP Server Experiments

Warning: Run load tests only on your own environment. Do not target external servers. Ensure you have permission.

## Requirements

- Docker & Docker Compose
- Python 3.10+ (optional for local runs)

## Quick Start (Docker)

1. Build and start services + Locust UI:
   - `docker compose up --build -d`
   - UI at http://localhost:8089 (Host: `http://mcp_server:5000`)
2. Run predefined experiments headless (A-E):
   - `chmod +x experiments.sh`
   - `./experiments.sh all` (env overrides: `DELAY_MS=... CACHE_TTL=... INTERNAL_TOUCH_MS=...`)
3. Results: CSVs in `results/<experiment>/`. Plots will be exported by the notebook to `notebooks/outputs/`.

## Services

- `key_server.py` (port 5001): `DELAY_MS`, `KEY_SIZE_BYTES`
- `internal_service.py` (port 5002): `TOUCH_MS` (or `INTERNAL_TOUCH_MS`)
- `mcp_server.py` (port 5000): `CACHE_TTL`, `KEY_SERVER_URL`, `INTERNAL_SVC_URL`

## Locust Scenarios

- `--scenario`: handshake | validate | fanout | mix
- `--kid-mode`: fixed | random, plus `--fixed-kid`
- JSON params: `--json-bytes`, `--json-depth`; JWT size: `--jwt-bytes`; fanout: `--fanout-n`

## Notebook

- `docker compose exec mcp_server bash -lc "jupyter nbconvert --to notebook --execute notebooks/analysis.ipynb --output executed.ipynb"`
- Or open locally and run all cells. Ensure `results/` contains experiment CSVs.

## Reproducibility

- To change env vars: `DELAY_MS=200 CACHE_TTL=0 docker compose up -d --force-recreate key_server mcp_server`
- Experiments script sets env per scenario and saves outputs in `results/`.
