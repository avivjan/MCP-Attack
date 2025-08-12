import asyncio
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent


server = Server("calculator")


number_pair_args = {
    "type": "object",
    "properties": {
        "a": {"type": "number", "description": "First number"},
        "b": {"type": "number", "description": "Second number"},
    },
    "required": ["a", "b"],
}


@server.tool(
    "add",
    description="Add two numbers (a + b)",
    args=number_pair_args,
)
async def add(a: float, b: float) -> list[TextContent]:
    return [TextContent(type="text", text=str(a + b))]


@server.tool(
    "subtract",
    description="Subtract two numbers (a - b)",
    args=number_pair_args,
)
async def subtract(a: float, b: float) -> list[TextContent]:
    return [TextContent(type="text", text=str(a - b))]


@server.tool(
    "multiply",
    description="Multiply two numbers (a * b)",
    args=number_pair_args,
)
async def multiply(a: float, b: float) -> list[TextContent]:
    return [TextContent(type="text", text=str(a * b))]


@server.tool(
    "divide",
    description="Divide two numbers (a / b)",
    args=number_pair_args,
)
async def divide(a: float, b: float) -> list[TextContent]:
    if b == 0:
        raise ValueError("Division by zero is not allowed.")
    return [TextContent(type="text", text=str(a / b))]


async def main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream)


if __name__ == "__main__":
    asyncio.run(main())
