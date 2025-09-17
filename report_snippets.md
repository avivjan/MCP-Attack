## Background

We evaluate two practical stress patterns against a simulated MCP server: (1) expensive authentication when cache is cold/off, and (2) fan-out amplification to an internal service. A JSON validation workload is included to isolate parsing cost.

## Method

Three Flask services compose the system: `mcp_server` (auth + validate + fanout), `key_server` (key fetch with delay and size), and `internal_service` (touch sleep). Locust drives traffic at fixed user counts and durations. Each run exports CSV with RPS, p50/p95/p99, failures, and server CPU/RSS. Cache TTL and key fetch delay control auth expense.

## Key Results (to summarize after running)

- With cache off and key fetch delay 200 ms, p95 increased by X ms and RPS dropped by Y% vs cache on.
- Random `kid` (cache bust) degraded handshake latency significantly vs fixed kid.
- JSON payload size/depth increased p95 non-linearly; 50 KB deep structures show clear parsing penalties.
- Fan-out `n` scales latency roughly linearly; failures rise when CPU saturation begins.

## Short Theory Snippet (LaTeX)

Throughput approximation under M/M/1:

```tex
\[ \rho = \frac{\lambda}{\mu}, \quad \text{stable if } \rho < 1 \]
```

Where \( \lambda \) is arrival rate (RPS) and \( \mu \) is service rate. As \(\rho\) approaches 1, queueing inflates tail latencies (p95/p99).

## Table Template

| Experiment                 | Users | RPS | p50 (ms) | p95 (ms) | p99 (ms) | Failures |
| -------------------------- | ----: | --: | -------: | -------: | -------: | -------: |
| A (baseline)               |   100 |     |          |          |          |
| B (expensive-auth, fixed)  |   200 |     |          |          |          |
| B (expensive-auth, random) |   200 |     |          |          |          |
| C (cache-bust, fixed)      |   200 |     |          |          |          |
| C (cache-bust, random)     |   200 |     |          |          |          |
| D (json 50KB d=50)         |   100 |     |          |          |          |
| E (fanout n=50)            |   100 |     |          |          |          |
