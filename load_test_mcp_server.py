"""
MCP Server for SSE Load Testing
Allows Claude to orchestrate and run load tests via MCP protocol
Save this as: load_test_mcp_server.py
"""

import os
import sys
import asyncio
import time
import aiohttp
import pandas as pd
import json
from typing import Dict, Any, List, Optional
from datetime import datetime

# Windows event loop fix
if sys.platform == 'win32':
    # הפעלת פוליסת לולאת אירועים מתאימה בווינדוס
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# MCP SDK imports
from mcp.server import Server
from mcp.types import Tool, TextContent

# ============================================================================
# Configuration
# ============================================================================
SERVER_URL: str = os.environ.get("SERVER_URL", "http://localhost:8000")
if not SERVER_URL.startswith("http://"):
    SERVER_URL = "http://" + SERVER_URL

STATE_KB: int = int(os.environ.get("STATE_KB", 16))
HEARTBEAT_SEC: float = float(os.environ.get("HEARTBEAT_SEC", 1.0))

# API Endpoints
METRICS_URL: str = f"{SERVER_URL}/metrics"
CALL_URL: str = f"{SERVER_URL}/mcp/call"
STREAM_BASE_URL: str = f"{SERVER_URL}/mcp/stream"
CLOSE_BASE_URL: str = f"{SERVER_URL}/mcp/close"

# Global state for tracking tests
current_test_results: Dict[str, Any] = {}
test_in_progress: bool = False

# ============================================================================
# Helper Functions
# ============================================================================

async def health_check(session: aiohttp.ClientSession) -> bool:
    """Check if server is healthy"""
    try:
        async with session.get(METRICS_URL, timeout=5) as response:
            return response.status == 200
    except Exception:
        return False


async def get_server_health() -> Dict[str, Any]:
    """Get server health status and metrics"""
    connector = aiohttp.TCPConnector(ssl=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        if await health_check(session):
            try:
                async with session.get(METRICS_URL, timeout=5) as response:
                    if response.status == 200:
                        metrics = await response.json()
                        return {
                            "status": "healthy",
                            "metrics": metrics
                        }
            except Exception as e:
                return {"status": "error", "message": str(e)}
        return {"status": "unhealthy", "message": "Server not responding"}


async def create_stream(session: aiohttp.ClientSession) -> Optional[str]:
    """Create a new stream"""
    try:
        async with session.post(CALL_URL, timeout=10) as response:
            if response.status == 200:
                data = await response.json()
                return data.get("stream_id")
    except Exception:
        pass
    return None


async def close_stream(session: aiohttp.ClientSession, stream_id: str):
    """Close a stream"""
    try:
        await session.post(f"{CLOSE_BASE_URL}/{stream_id}", timeout=5)
    except Exception:
        pass


async def sse_client_task(
    session: aiohttp.ClientSession,
    stream_id: str,
    results_list: List[Dict[str, Any]],
    n_concurrent: int,
    duration: float
):
    """Client task listening to SSE stream"""
    events_received = 0
    start_time = time.time()
    
    try:
        async with session.get(
            f"{STREAM_BASE_URL}/{stream_id}",
            timeout=duration + 10
        ) as response:
            
            if response.status != 200:
                return
            
            # לולאת הקריאה (הוגבלה מעט כדי למנוע ריצה אינסופית במקרה של תקלה)
            while time.time() - start_time < duration + 5:
                try:
                    line = await asyncio.wait_for(
                        response.content.readline(),
                        timeout=HEARTBEAT_SEC * 3
                    )
                    
                    if not line:
                        break
                    
                    line_str = line.decode('utf-8').strip()
                    if line_str.startswith("data:"):
                        events_received += 1
                    
                    # יציאה מוקדמת אם עבר יותר מדי זמן
                    if time.time() - start_time > duration * 1.5:
                        break
                
                except asyncio.TimeoutError:
                    continue
                except Exception:
                    break
    
    except Exception:
        pass
    
    finally:
        end_time = time.time()
        duration_actual = max(end_time - start_time, 0.1)
        
        results_list.append({
            'N': n_concurrent,
            'stream_id': stream_id,
            'events_received': events_received,
            'duration': duration_actual,
            'avg_events_per_sec': events_received / duration_actual if duration_actual > 0 else 0
        })
        
        await close_stream(session, stream_id)


async def poll_metrics(
    session: aiohttp.ClientSession,
    n_concurrent: int,
    metrics_list: List[Dict[str, Any]],
    duration: float
):
    """Poll server metrics"""
    start_time = time.time()
    n_metrics_data = []
    
    while time.time() - start_time < duration + 5:
        try:
            async with session.get(METRICS_URL, timeout=5) as response:
                if response.status == 200:
                    metrics = await response.json()
                    metrics['time'] = time.time()
                    metrics['elapsed_sec'] = time.time() - start_time
                    metrics['N'] = n_concurrent
                    
                    metrics_list.append(metrics)
                    n_metrics_data.append(metrics)
        except Exception:
            pass
        
        await asyncio.sleep(1)
    
    if n_metrics_data:
        # שמירת נתונים ל-CSV לניתוח מעמיק יותר
        df_ts = pd.DataFrame(n_metrics_data)
        filename = f'metrics_timeseries_N{n_concurrent}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        df_ts.to_csv(filename, index=False)


async def run_load_test(n_streams: int, duration_sec: float) -> Dict[str, Any]:
    """Run a single load test sweep"""
    global current_test_results, test_in_progress
    
    test_in_progress = True
    all_metrics: List[Dict[str, Any]] = []
    all_results: List[Dict[str, Any]] = []
    
    try:
        connector = aiohttp.TCPConnector(
            ssl=False,
            limit=max(n_streams + 10, 100),
            limit_per_host=n_streams + 5
        )
        
        async with aiohttp.ClientSession(connector=connector) as session:
            # Create streams
            stream_ids = []
            # יצירת מופע של בדיקה לכל חיבור מבוקש
            create_tasks = [create_stream(session) for _ in range(n_streams)]
            
            for sid in await asyncio.gather(*create_tasks):
                if sid is not None:
                    stream_ids.append(sid)
            
            streams_created = len(stream_ids)
            
            if streams_created == 0:
                return {"error": "Failed to create any streams"}
            
            # Start client and metrics tasks
            client_tasks = [
                asyncio.create_task(
                    sse_client_task(session, sid, all_results, streams_created, duration_sec)
                )
                for sid in stream_ids
            ]
            
            # הפעלת איסוף המדדים במקביל
            metrics_task = asyncio.create_task(
                poll_metrics(session, streams_created, all_metrics, duration_sec)
            )
            
            # Run for duration
            await asyncio.sleep(duration_sec)
            
            # Cleanup
            for task in client_tasks:
                task.cancel()
            metrics_task.cancel()
            
            # המתנה לסיום כל המשימות (או ביטולן)
            await asyncio.gather(*client_tasks, metrics_task, return_exceptions=True)
        
        # Analyze results
        if all_metrics and all_results:
            df_metrics = pd.DataFrame(all_metrics)
            df_results = pd.DataFrame(all_results)
            
            # סיכום מדדי השרת
            summary_metrics = df_metrics.groupby('N').agg(
                avg_cpu_percent=('cpu_percent', 'mean'),
                peak_cpu_percent=('cpu_percent', 'max'),
                avg_rss_mb=('rss_mb', 'mean'),
                peak_rss_mb=('rss_mb', 'max'),
            ).reset_index()
            
            # סיכום מדדי הבדיקה
            summary_results = df_results.groupby('N').agg(
                total_streams=('stream_id', 'count'),
                total_events_received=('events_received', 'sum'),
                total_events_per_sec=('avg_events_per_sec', 'sum'),
            ).reset_index()
            
            df_summary = pd.merge(summary_metrics, summary_results, on='N')
            
            result_dict = df_summary.to_dict('records')[0] if len(df_summary) > 0 else {}
            
            current_test_results = {
                "status": "completed",
                "streams": streams_created,
                "duration": duration_sec,
                "timestamp": datetime.now().isoformat(),
                "summary": result_dict
            }
            
            return current_test_results
        
        return {"error": "No data collected"}
    
    finally:
        test_in_progress = False

# ============================================================================
# MCP Server Setup (Tools for Claude)
# ============================================================================

server = Server("load-test-server")


@server.list_tools()
async def list_tools():
    """List available tools for Claude"""
    return [
        Tool(
            name="check_server_health",
            description="Check if the SSE server is healthy and get current metrics",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="run_load_test",
            description="Run a load test with specified number of concurrent streams and duration",
            inputSchema={
                "type": "object",
                "properties": {
                    "n_streams": {
                        "type": "integer",
                        "description": "Number of concurrent streams to create (e.g., 10, 50, 100, 500)"
                    },
                    "duration_seconds": {
                        "type": "integer",
                        "description": "How long to maintain the load in seconds (e.g., 15)"
                    }
                },
                "required": ["n_streams", "duration_seconds"]
            }
        ),
        Tool(
            name="get_test_results",
            description="Get the results from the last completed load test",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="get_server_config",
            description="Get the current server configuration and URL",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]):
    """Handle tool calls from Claude"""
    
    if name == "check_server_health":
        result = await get_server_health()
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
    
    elif name == "run_load_test":
        if test_in_progress:
            return [TextContent(type="text", text=json.dumps({
                "error": "Test already in progress"
            }))]
        
        n_streams = arguments.get("n_streams", 10)
        duration_seconds = arguments.get("duration_seconds", 15)
        
        result = await run_load_test(n_streams, duration_seconds)
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
    
    elif name == "get_test_results":
        if current_test_results:
            return [TextContent(type="text", text=json.dumps(current_test_results, indent=2))]
        else:
            return [TextContent(type="text", text=json.dumps({
                "message": "No test results available yet"
            }))]
    
    elif name == "get_server_config":
        return [TextContent(type="text", text=json.dumps({
            "server_url": SERVER_URL,
            "state_kb": STATE_KB,
            "heartbeat_sec": HEARTBEAT_SEC,
            "metrics_url": METRICS_URL,
            "call_url": CALL_URL
        }, indent=2))]
    
    else:
        return [TextContent(type="text", text=json.dumps({
            "error": f"Unknown tool: {name}"
        }))]


async def main():
    """Run the MCP server (Used when running with Claude CLI)"""
    print("Starting Load Test MCP Server...")
    print(f"Server URL: {SERVER_URL}")
    print(f"State KB: {STATE_KB}")
    print(f"Heartbeat: {HEARTBEAT_SEC}s")
    print("\nServer ready. Waiting for Claude to connect...")
    
    # Run server on stdio
    async with server:
        await server.wait_until_closed()


# ============================================================================
# Standalone Mode for Local Testing (THE ADDITION)
# ============================================================================

async def standalone_load_test():
    """פונקציה להרצת בדיקת עומס עצמאית לבדיקה מקומית (ללא Claude)"""
    print("Starting Standalone Load Test...")
    
    # הגדר את פרמטרי הבדיקה הרצויים: 20 חיבורים למשך 10 שניות
    N_STREAMS = 20 
    DURATION_SEC = 10
    
    health = await get_server_health()
    print("\n--- Server Health Check ---")
    print(json.dumps(health, indent=2))
    
    if health.get("status") == "healthy":
        print(f"\n--- Running Load Test: {N_STREAMS} streams for {DURATION_SEC}s ---")
        results = await run_load_test(N_STREAMS, DURATION_SEC)
        print("\n--- Load Test Results ---")
        print(json.dumps(results, indent=2))
    else:
        print("\n!!! ERROR: Target server is not healthy (Is server.py running on port 8000?). Load test aborted. !!!")


if __name__ == "__main__":

    # הרצת בדיקה עצמאית:
    asyncio.run(standalone_load_test())
    
    # asyncio.run(main())