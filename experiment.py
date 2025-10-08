import os
import asyncio
import time
import aiohttp
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional

# --- הגדרות הניסוי המותאמות ---
SERVER_URL: str = os.environ.get("SERVER_URL", "http://host.docker.internal") 

# פרמטרים הניתנים לשינוי לבדיקת עומס (Stress Testing)
STATE_KB: int = int(os.environ.get("STATE_KB", 16)) 
HEARTBEAT_SEC: float = float(os.environ.get("HEARTBEAT_SEC", 1.0)) 
DURATION_SEC: float = float(os.environ.get("DURATION_SEC", 15)) # 15 שניות לייצוב הריצה

# מספר הזרמים המקבילים לבדיקה: טווח רחב לגרף מלא
N_SWEEP: List[int] = [10, 100, 300, 500, 700, 900, 1100] 
# ----------------------

# נתיבים לנקודות הקצה
METRICS_URL: str = f"{SERVER_URL}/metrics"
CALL_URL: str = f"{SERVER_URL}/mcp/call"
STREAM_BASE_URL: str = f"{SERVER_URL}/mcp/stream"
CLOSE_BASE_URL: str = f"{SERVER_URL}/mcp/close"

# ----------------------------------------------------------------------
# פונקציות עזר לבדיקות ולוגיקה
# ----------------------------------------------------------------------

async def health_check(session: aiohttp.ClientSession, url: str) -> bool:
    """בודק אם השרת מוכן ומחזיר סטטוס תקין."""
    try:
        # שימוש ב-http://nginx במקום host.docker.internal ל-Healthcheck
        async with session.get(f"http://nginx/metrics", timeout=5) as response: 
            return response.status == 200
    except Exception:
        return False

async def wait_for_server_ready(url: str, timeout: int = 20):
    """ממתין שהשרת יעלה ויחזיר קוד 200."""
    print(f"Waiting for server at {url}...")
    async with aiohttp.ClientSession(trust_env=True) as session:
        start_time = time.time()
        # הפונקציה הזו תעבור רק אם ה-Healthcheck של Docker נכשל
        # כיוון שהיא נועדה להיות מבוטלת על ידי depends_on: service_healthy
        # נשנה את בדיקת ה-Healthcheck הפנימית ל-NGINX
        while time.time() - start_time < timeout:
            if await health_check(session, url):
                print("Server is ready.")
                return
            await asyncio.sleep(1)
        raise TimeoutError("Server did not become ready in time")

async def create_stream(session: aiohttp.ClientSession) -> Optional[str]:
    """שולח בקשת CALL ליצירת זרם חדש."""
    try:
        async with session.post(CALL_URL) as response: 
            if response.status == 200:
                data = await response.json()
                return data.get("stream_id")
            print(f"Warning: Failed to create stream, status {response.status}")
            return None
    except Exception as e:
        print(f"Warning: Exception during stream creation: {e}")
        return None

async def close_stream(session: aiohttp.ClientSession, stream_id: str):
    """סוגר חיבור באופן יזום."""
    try:
        await session.post(f"{CLOSE_BASE_URL}/{stream_id}")
    except Exception:
        pass

async def sse_client_task(session: aiohttp.ClientSession, stream_id: str, results_list: List[Dict[str, Any]], n_concurrent: int):
    """משימת לקוח המאזינה לזרם SSE. כולל תיקון readline ושמירת N."""
    events_received = 0
    start_time = time.time()
    
    try:
        async with session.get(f"{STREAM_BASE_URL}/{stream_id}", timeout=DURATION_SEC * 2) as response:
            
            # שימוש ב-readline (תיקון יציבות)
            content = response.content
            while time.time() - start_time < DURATION_SEC + 5:
                
                line = await asyncio.wait_for(content.readline(), timeout=HEARTBEAT_SEC * 2) 
                
                if not line:
                    break
                    
                line = line.decode('utf-8').strip()

                if line.startswith("data:"):
                    events_received += 1
                
                if time.time() - start_time > DURATION_SEC * 1.5:
                    break 

    except asyncio.TimeoutError:
        pass
    except asyncio.CancelledError:
        pass 
    except Exception as e:
        print(f"SSE client task exception: Server disconnected or error: {e}")
    finally:
        end_time = time.time()
        # שמירת N יציב (תיקון יציבות ניתוח)
        results_list.append({
            'N': n_concurrent,
            'stream_id': stream_id,
            'events_received': events_received,
            'duration': end_time - start_time,
            'avg_events_per_sec': events_received / (end_time - start_time) if end_time > start_time else 0
        })
        await close_stream(session, stream_id)


async def poll_metrics(session: aiohttp.ClientSession, n_concurrent: int, metrics_list: List[Dict[str, Any]]):
    """משימת Poll למדדי ביצועים מהשרת. יוצרת קובץ timeseries לכל N."""
    start_time = time.time()
    
    # שמירת מדדים ל-N ספציפי ליצירת קובץ Timeseries
    n_metrics_data = [] 

    while time.time() - start_time < DURATION_SEC + 5:
        try:
            async with session.get(METRICS_URL, timeout=5) as response:
                if response.status == 200:
                    metrics = await response.json()
                    current_time = time.time()
                    metrics['time'] = current_time
                    metrics['N'] = n_concurrent
                    
                    metrics_list.append(metrics)
                    
                    # הוספת נתונים לרשימת ה-Timeseries המקומית
                    n_metrics_data.append(metrics) 
                    
        except Exception:
            pass
        await asyncio.sleep(1)
        
    # יצירת קובץ Timeseries ל-N ספציפי לאחר סיום הניסוי
    if n_metrics_data:
        df_timeseries = pd.DataFrame(n_metrics_data)
        df_timeseries.to_csv(f'metrics_timeseries_{n_concurrent}.csv', index=False)


async def run_single_sweep(n_concurrent: int, all_metrics: List[Dict[str, Any]], all_results: List[Dict[str, Any]]):
    """מריץ שלב יחיד ב-N נתון."""
    print(f"\n--- Running N={n_concurrent} ---")
    start_total_time = time.time()
    
    async with aiohttp.ClientSession(trust_env=True) as session:
        # 1. יצירת זרמים
        print(f"Attempting to create {n_concurrent} streams...")
        stream_ids: List[str] = [
            sid for sid in await asyncio.gather(
                *[create_stream(session) for _ in range(n_concurrent)]
            ) if sid is not None
        ]
        
        streams_created = len(stream_ids)
        if streams_created < n_concurrent:
            print(f"Warning: Only created {streams_created} out of {n_concurrent} streams.")
        
        # 2. הפעלת משימות קליינט ו-Poll
        client_tasks = [
            asyncio.create_task(sse_client_task(session, sid, all_results, n_concurrent)) 
            for sid in stream_ids
        ]
        
        metrics_task = asyncio.create_task(poll_metrics(session, streams_created, all_metrics))
        
        # 3. המתנה למשך הניסוי
        await asyncio.sleep(DURATION_SEC)
        
        # 4. ניקוי וסגירת משימות
        print("Cleaning up tasks and closing connections...")
        for task in client_tasks:
            task.cancel()
        metrics_task.cancel()

        await asyncio.gather(*client_tasks, metrics_task, return_exceptions=True)
        
        end_total_time = time.time()
        print(f"Sweep for N={streams_created} finished in {end_total_time - start_total_time:.2f} seconds.")


# ----------------------------------------------------------------------
# פונקציית ריצה ראשית וניתוח תוצאות
# ----------------------------------------------------------------------

def analyze_and_plot(df_metrics: pd.DataFrame, df_results: pd.DataFrame):
    """מחשב מדדים מסכמים ומפיק גרפים."""
    
    # 1. חישוב מדדי ביצועים ממוצעים לכל N (מתוך מדדי השרת)
    summary_metrics = df_metrics.groupby('N').agg(
        avg_cpu_percent=('cpu_percent', 'mean'),
        peak_cpu_percent=('cpu_percent', 'max'),
        avg_rss_mb=('rss_mb', 'mean'),
        peak_rss_mb=('rss_mb', 'max'),
        avg_fds_open=('fds_open', 'mean'),
        peak_fds_open=('fds_open', 'max'),
    ).reset_index()

    # 2. חישוב מדדי קליינט ממוצעים לכל N (מתוך הזרמים)
    summary_results = df_results.groupby('N').agg(
        total_streams=('stream_id', 'count'),
        avg_events_per_stream=('events_received', 'mean'),
        total_events_per_sec=('avg_events_per_sec', 'sum'),
    ).reset_index()
    
    # מיזוג התוצאות
    df_summary = pd.merge(summary_metrics, summary_results, on='N')

    # שמירת סיכום
    df_summary.to_csv('results.csv', index=False)
    
    with open('summary.txt', 'w') as f:
        f.write(f"--- MCP Load Test Summary ---\n")
        f.write(f"Parameters: STATE_KB={STATE_KB}, HEARTBEAT_SEC={HEARTBEAT_SEC}, DURATION_SEC={DURATION_SEC}\n\n")
        f.write(df_summary.to_string())

    print("\n" + "="*50)
    print("Test Complete. Summary:")
    print("="*50)
    print(df_summary.to_string())
    print("="*50 + "\n")

    # 3. יצירת גרפים (שלושה גרפים עם השמות המקוריים)
    
    # גרף 1: Throughput (אירועים) vs. N - (plot_events_vs_N.png)
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel('N Concurrent Streams')
    ax1.set_ylabel('Total Throughput (Events/sec)')
    ax1.plot(df_summary['N'], df_summary['total_events_per_sec'], label='Total Events/sec', color='purple', marker='o')
    ax1.set_title('Total System Throughput vs. Concurrent Streams (N)')
    ax1.legend()
    fig1.savefig('plot_events_vs_N.png')
    plt.close(fig1)
    
    # גרף 2: Memory (RSS) vs. N - (plot_rss_vs_N.png)
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.set_xlabel('N Concurrent Streams')
    ax2.set_ylabel('Peak Memory (RSS MB)')
    ax2.plot(df_summary['N'], df_summary['peak_rss_mb'], label='Peak RSS (MB)', color='green', marker='x')
    ax2.set_title('Peak Memory Usage (RSS) vs. Concurrent Streams (N)')
    ax2.legend()
    fig2.savefig('plot_rss_vs_N.png')
    plt.close(fig2)
    
    # גרף 3: CPU vs. N - (משתמשים בשם plot_ttfb_vs_N.png)
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.set_xlabel('N Concurrent Streams')
    ax3.set_ylabel('Average CPU Usage (%)')
    ax3.plot(df_summary['N'], df_summary['avg_cpu_percent'], label='Avg CPU (%)', color='red', marker='s')
    ax3.set_title('Average CPU Usage vs. Concurrent Streams (N)')
    ax3.legend()
    fig3.savefig('plot_ttfb_vs_N.png') 
    plt.close(fig3)


async def main_run():
    all_metrics: List[Dict[str, Any]] = []
    all_results: List[Dict[str, Any]] = []
    
    try:
        # ההמתנה בוטלה על ידי Docker depends_on: service_healthy, 
        # אבל נשאיר את הפונקציה ליציבות סביבתית
        # await wait_for_server_ready(METRICS_URL) 
        
        print(f"\n--- Starting experiment sweep with STATE_KB={STATE_KB}, HEARTBEAT_SEC={HEARTBEAT_SEC}, DURATION_SEC={DURATION_SEC} ---")
        
        for n in N_SWEEP:
            await run_single_sweep(n, all_metrics, all_results) 
            
        df_metrics = pd.DataFrame(all_metrics)
        df_results = pd.DataFrame(all_results)
        
        if df_metrics.empty or df_results.empty:
            print("\nError: No valid metrics or results were collected.")
            return

        # ניתוח והצגה
        analyze_and_plot(df_metrics, df_results)

    except TimeoutError as e:
        print(f"\nError during main run: Server not ready: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    try:
        asyncio.run(main_run())
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user.")