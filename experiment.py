import asyncio
from typing import List

import pandas as pd
from pathlib import Path

from client import analyze_and_plot, run_for_N, wait_for_server, write_summary_text
from config import N_SWEEP
from server import start_server, stop_server
BASE_DIR = Path(__file__).resolve().parent


async def main_async() -> None:
    handle, base_url = start_server()
    try:
        await wait_for_server(base_url)
        all_conn_dfs: List[pd.DataFrame] = []
        all_metrics_dfs: List[pd.DataFrame] = []

        for N in N_SWEEP:
            print(f"Running N={N} ...")
            per_conn_df, metrics_df = await run_for_N(base_url, N)
            metrics_df.to_csv(BASE_DIR / f"metrics_timeseries_{N}.csv", index=False)
            all_conn_dfs.append(per_conn_df)
            all_metrics_dfs.append(metrics_df)
            await asyncio.sleep(5)

        all_conn_df = pd.concat(all_conn_dfs, ignore_index=True) if all_conn_dfs else pd.DataFrame()
        all_metrics_df = pd.concat(all_metrics_dfs, ignore_index=True) if all_metrics_dfs else pd.DataFrame()

        summary_df = analyze_and_plot(all_conn_df, all_metrics_df)
        summary_text = write_summary_text(summary_df)

        print("\n" + summary_text + "\n")
    finally:
        stop_server(handle)


def print_how_to_run() -> None:
    print("""
# 1) Create venv and install:
#    pip install fastapi uvicorn aiohttp psutil pandas matplotlib
# 2) Run:
#    python firstExperiment/experiment.py
# 3) See outputs:
#    results.csv, summary.txt, plot_*.png
""".strip())


if __name__ == "__main__":
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        pass
    finally:
        print_how_to_run()


