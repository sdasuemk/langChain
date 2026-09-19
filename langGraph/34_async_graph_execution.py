"""
=============================================================================
Phase 7: Advanced Production Patterns - Lesson 34: Asynchronous Graph Execution
=============================================================================

Pattern:
                     START
                       │
                       ▼
                 fetch_remote_data   (async non-blocking I/O)
                       │
                       ▼
                 process_async_data
                       │
                       ▼
                      END

Key Concepts:
1. Native Async Nodes (`async def`):
   LangGraph natively supports async node functions, allowing non-blocking
   HTTP requests, database calls, and concurrent LLM inferences.
2. Async API (`ainvoke`, `astream`):
   - `await app.ainvoke(...)`: Non-blocking invocation for FastAPI/ASGI servers.
   - `async for event in app.astream(...)`: Real-time async streaming.
=============================================================================
"""

import sys
import io
import asyncio
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

# Ensure proper utf-8 encoding for Windows terminals
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


# --- 1. Define State Schema ---
class AsyncState(TypedDict):
    endpoint_url: str
    downloaded_data: str
    processing_duration: float


# --- 2. Define Asynchronous Node Functions ---
async def fetch_remote_data_node(state: AsyncState) -> dict:
    print(f"\n[Async Step 1] Initiating async non-blocking fetch from: '{state['endpoint_url']}'...")
    # Simulate non-blocking network I/O with asyncio.sleep
    await asyncio.sleep(0.3)
    print("  --> Async fetch completed without blocking event loop.")
    return {"downloaded_data": "Status: 200 OK | Records: 1,500 payload items."}


async def process_async_data_node(state: AsyncState) -> dict:
    print("\n[Async Step 2] Transforming payload asynchronously...")
    await asyncio.sleep(0.2)
    print("  --> Payload normalized and indexed.")
    return {"processing_duration": 0.5}


async def run_async_workflow():
    print("=" * 65)
    print("  LangGraph Advanced: Lesson 34 - Async Execution (ainvoke / astream)")
    print("=" * 65)

    # --- 3. Assemble Graph ---
    graph = StateGraph(AsyncState)

    graph.add_node("fetch", fetch_remote_data_node)
    graph.add_node("process", process_async_data_node)

    graph.add_edge(START, "fetch")
    graph.add_edge("fetch", "process")
    graph.add_edge("process", END)

    app = graph.compile()

    initial_input: AsyncState = {
        "endpoint_url": "https://api.example.com/v1/telemetry",
        "downloaded_data": "",
        "processing_duration": 0.0
    }

    # --- 4. Test ainvoke (Non-blocking) ---
    print("\n>>> 1. Testing 'app.ainvoke' (Standard Async Invocations)")
    final_state = await app.ainvoke(initial_input)
    print(f"Result Payload : {final_state['downloaded_data']}")
    print(f"Total Duration : {final_state['processing_duration']}s")

    # --- 5. Test astream (Async Streaming) ---
    print("\n" + "=" * 60)
    print(">>> 2. Testing 'app.astream' (Real-time Async Streaming)")
    print("=" * 60)

    async for event in app.astream(initial_input, stream_mode="updates"):
        for node_name, updates in event.items():
            print(f"Received async update from node [{node_name}]: {updates}")


def main():
    asyncio.run(run_async_workflow())


if __name__ == "__main__":
    main()
