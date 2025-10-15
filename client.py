import asyncio
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from config import DURATION_SEC, METRIC_POLL_SEC

BASE_DIR = Path(__file__).resolve().parent


async def sse_client_task(session: aiohttp.ClientSession, url: str, duration_sec: float) -> Tuple[Optional[float], int, float]:
    """Return (ttfb_ms, events_count, open_seconds)."""
    ttfb_ms: Optional[float] = None
    events_count = 0
    started = time.perf_counter()
    open_seconds = 0.0

    timeout = aiohttp.ClientTimeout(total=duration_sec + 10, sock_connect=5, sock_read=5)
    headers = {"Accept": "text/event-stream"}
    try:
        async with session.get(url, timeout=timeout, headers=headers) as resp:
            if resp.status != 200:
                return ttfb_ms, events_count, open_seconds
            msg_has_data = False
            while True:
                line = await resp.content.readline()
                if line == b"":
                    break
                # End of one SSE message
                if line in (b"\n", b"\r\n"):
                    if msg_has_data:
                        events_count += 1
                        msg_has_data = False
                    if time.perf_counter() - started >= duration_sec:
                        break
                    continue
                if line.startswith(b"data:"):
                    if ttfb_ms is None:
                        ttfb_ms = (time.perf_counter() - started) * 1000.0
                    msg_has_data = True
            open_seconds = time.perf_counter() - started
    except asyncio.TimeoutError:
        open_seconds = time.perf_counter() - started
    except Exception:
        open_seconds = time.perf_counter() - started
    return ttfb_ms, events_count, open_seconds


async def poll_metrics_timeseries(base_url: str, duration_sec: float, n_label: int) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    stop_at = time.perf_counter() + duration_sec
    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=connector) as session:
        while time.perf_counter() < stop_at:
            try:
                async with session.get(base_url + "/metrics", timeout=aiohttp.ClientTimeout(total=3.0)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        data["N"] = n_label
                        rows.append(data)
            except Exception:
                pass
            await asyncio.sleep(METRIC_POLL_SEC)
    return pd.DataFrame(rows)


async def run_for_N(base_url: str, N: int, *, call_base_url: Optional[str] = None, stream_base_url: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    async with aiohttp.ClientSession() as session:
        # First, create N calls to get stream IDs with nested params
        stream_ids: List[str] = []
        for i in range(N):
            nested_params: Dict[str, Any] = {"i": i, "ts": time.time(), "values": [i, i + 1, i + 2]}
            # build nested params similar to original helper
            node: Dict[str, Any] = nested_params
            for lvl in range(5):
                node = {"level": lvl, "node": node, "flags": {"a": True, "b": False}}
            rpc = {"jsonrpc": "2.0", "id": str(uuid.uuid4()), "method": "testMethod", "params": node}
            try:
                async with session.post((call_base_url or base_url) + "/mcp/call", json=rpc, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        res = await resp.json()
                        sid = res.get("result", {}).get("stream_id")
                        if isinstance(sid, str):
                            stream_ids.append(sid)
            except Exception:
                pass
        # Start SSE clients for each stream id
        tasks = []
        for sid in stream_ids:
            sse_url = (stream_base_url or base_url) + f"/mcp/stream?id={sid}"
            tasks.append(asyncio.create_task(sse_client_task(session, sse_url, DURATION_SEC)))
        # In parallel, poll metrics
        metrics_task = asyncio.create_task(poll_metrics_timeseries((call_base_url or base_url), DURATION_SEC, N))
        results = await asyncio.gather(*tasks)
        metrics_df = await metrics_task

    # Build per-connection results DataFrame
    rows: List[Dict[str, Any]] = []
    for ttfb_ms, events, open_sec in results:
        events_per_sec = (events / open_sec) if open_sec > 0 else float("nan")
        rows.append({
            "N": N,
            "ttfb_ms": ttfb_ms if ttfb_ms is not None else float("nan"),
            "events": events,
            "open_sec": open_sec,
            "events_per_sec": events_per_sec,
        })
    per_conn_df = pd.DataFrame(rows)
    return per_conn_df, metrics_df


async def wait_for_server(base_url: str, timeout_sec: float = 10.0) -> None:
    start = time.perf_counter()
    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=connector) as session:
        while time.perf_counter() - start < timeout_sec:
            try:
                async with session.get(base_url + "/metrics", timeout=aiohttp.ClientTimeout(total=2.0)) as resp:
                    if resp.status == 200:
                        return
            except Exception:
                await asyncio.sleep(0.1)
        raise RuntimeError("Server did not become ready in time")


def analyze_and_plot(all_conn_df: pd.DataFrame, all_metrics_df: pd.DataFrame) -> pd.DataFrame:
    summary_rows: List[Dict[str, Any]] = []
    for N, conn_df_N in all_conn_df.groupby("N"):
        metrics_N = all_metrics_df[all_metrics_df["N"] == N]
        peak_rss_mb = float(metrics_N["rss_mb"].max()) if not metrics_N.empty else float("nan")
        avg_cpu_percent = float(metrics_N["cpu_percent"].mean()) if not metrics_N.empty else float("nan")
        avg_fds_open = float(metrics_N["fds_open"].mean()) if (not metrics_N.empty and "fds_open" in metrics_N) else float("nan")

        avg_ttfb_ms = float(conn_df_N["ttfb_ms"].mean())
        p95_ttfb_ms = float(conn_df_N["ttfb_ms"].quantile(0.95))
        avg_events_per_sec = float(conn_df_N["events_per_sec"].mean())

        summary_rows.append({
            "N": int(N),
            "peak_rss_mb": peak_rss_mb,
            "avg_cpu_percent": avg_cpu_percent,
            "avg_fds_open": avg_fds_open,
            "avg_ttfb_ms": avg_ttfb_ms,
            "p95_ttfb_ms": p95_ttfb_ms,
            "avg_events_per_sec": avg_events_per_sec,
        })

    summary_df = pd.DataFrame(summary_rows).sort_values("N").reset_index(drop=True)
    summary_df.to_csv(BASE_DIR / "results.csv", index=False)

    # Plots
    plt.figure()
    plt.plot(summary_df["N"], summary_df["peak_rss_mb"], marker="o")
    plt.xlabel("Concurrent streams (N)")
    plt.ylabel("Peak RSS (MB)")
    plt.title("N vs Peak RSS")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.savefig(str(BASE_DIR / "plot_rss_vs_N.png"), bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(summary_df["N"], summary_df["avg_ttfb_ms"], marker="o")
    plt.xlabel("Concurrent streams (N)")
    plt.ylabel("Avg TTFB (ms)")
    plt.title("N vs Avg TTFB")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.savefig(str(BASE_DIR / "plot_ttfb_vs_N.png"), bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(summary_df["N"], summary_df["avg_events_per_sec"], marker="o")
    plt.xlabel("Concurrent streams (N)")
    plt.ylabel("Avg events/sec")
    plt.title("N vs Avg Events/sec")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.savefig(str(BASE_DIR / "plot_events_vs_N.png"), bbox_inches="tight")
    plt.close()

    return summary_df


def write_summary_text(summary_df: pd.DataFrame) -> str:
    lines: List[str] = []
    # Simple heuristics for conclusions
    def trend_ratio(col: str) -> Optional[float]:
        if len(summary_df) < 2:
            return None
        first = summary_df.iloc[0][col]
        last = summary_df.iloc[-1][col]
        if first and first == first and last and last == last and first > 0:
            return float(last) / float(first)
        return None

    mem_ratio = trend_ratio("peak_rss_mb")
    ttfb_ratio = trend_ratio("avg_ttfb_ms")

    lines.append("This experiment measured an MCP-like SSE server under increasing concurrent stream counts.")
    if mem_ratio is not None:
        lines.append(f"Memory usage (peak RSS) scaled by ~{mem_ratio:.2f}x from min to max N.")
    if ttfb_ratio is not None:
        lines.append(f"Average TTFB scaled by ~{ttfb_ratio:.2f}x across the sweep.")
    if "avg_events_per_sec" in summary_df:
        avg_eps_min = float(summary_df["avg_events_per_sec"].min())
        avg_eps_max = float(summary_df["avg_events_per_sec"].max())
        lines.append(f"Delivered events/sec per stream remained around {avg_eps_min:.2f}–{avg_eps_max:.2f}.")
    lines.append("Overall, results suggest roughly linear resource growth with N for this small per-stream state.")
    lines.append("Next step: increase STATE_KB and/or add connection churn to probe thresholds.")
    text = "\n".join(lines)
    with open(BASE_DIR / "summary.txt", "w", encoding="utf-8") as f:
        f.write(text + "\n")
    return text


