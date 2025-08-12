# Calculator MCP Server (Python)

A minimal Model Context Protocol (MCP) stdio server exposing four tools: add, subtract, multiply, divide.

## Install

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## Run (invoked by MCP clients via stdio)

Most MCP clients will spawn the server via stdio. If you want to run it manually for debugging:

```bash
python /Users/avivjan/git/MCP-Attack/calculator_server.py
```

This will wait for JSON-RPC over stdio.

## Configure in an MCP client

Example configuration snippet for an MCP-aware client:

```json
{
  "mcpServers": {
    "calculator": {
      "command": "python",
      "args": ["/Users/avivjan/git/MCP-Attack/calculator_server.py"]
    }
  }
}
```

## Tools

- add(a: number, b: number) -> string result
- subtract(a: number, b: number) -> string result
- multiply(a: number, b: number) -> string result
- divide(a: number, b: number) -> string result (errors on division by zero)

The server responds with text content containing the numeric result.
