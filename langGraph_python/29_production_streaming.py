"""
=============================================================================
Phase 6: Multi-Agent Systems & Production - Lesson 29: Production Streaming
=============================================================================

Key Concepts:
1. `stream_mode="values"`:
   Yields the full state snapshot after every node super-step. Ideal when your
   frontend needs to mirror the entire state store.
2. `stream_mode="updates"`:
   Yields only the dictionary output of the node that just executed.
   Extremely efficient for tracking step-by-step progress.
3. Token-by-Token Streaming (`stream_mode="messages"`):
   Yields chunks of tokens in real-time as the LLM generates them,
   enabling responsive typing effects for conversational interfaces.
=============================================================================
"""

import sys
import io
import time
from typing import List
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk
from langgraph.graph import StateGraph, MessagesState, START, END

# Ensure proper utf-8 encoding for Windows terminals
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


# --- 1. Nodes with Token Generation Simulation ---
def step_search_node(state: MessagesState) -> dict:
    print("\n[Node: search] Locating knowledge base facts...")
    return {"messages": [AIMessage(content="Found: LangGraph supports 3 distinct streaming modes.")]}


def step_writer_node(state: MessagesState) -> dict:
    print("\n[Node: writer] Synthesizing comprehensive response...")
    response_text = (
        "In production environments, streaming is essential. "
        "Use 'updates' for workflow progress, and 'messages' for token-by-token output."
    )
    return {"messages": [AIMessage(content=response_text)]}


def main():
    print("=" * 65)
    print("  LangGraph Phase 6: Lesson 29 - Production Streaming Modes")
    print("=" * 65)

    # --- 2. Assemble Graph ---
    graph = StateGraph(MessagesState)
    graph.add_node("search", step_search_node)
    graph.add_node("writer", step_writer_node)

    graph.add_edge(START, "search")
    graph.add_edge("search", "writer")
    graph.add_edge("writer", END)

    app = graph.compile()

    initial_input = {"messages": [HumanMessage(content="Explain LangGraph streaming.")]}

    # --- Mode 1: stream_mode="updates" ---
    print("\n" + "=" * 60)
    print(">>> 1. STREAM MODE: 'updates' (Node-by-node diffs)")
    print("=" * 60)

    for event in app.stream(initial_input, stream_mode="updates"):
        for node_name, node_output in event.items():
            print(f"Update from [{node_name}]:")
            for msg in node_output["messages"]:
                print(f"  --> {msg.content}")

    # --- Mode 2: stream_mode="values" ---
    print("\n" + "=" * 60)
    print(">>> 2. STREAM MODE: 'values' (Full state after each superstep)")
    print("=" * 60)

    step_number = 1
    for current_state in app.stream(initial_input, stream_mode="values"):
        print(f"Super-step #{step_number} State Message Count: {len(current_state['messages'])}")
        step_number += 1

    # --- Mode 3: Real-Time Simulated Token Streaming ---
    print("\n" + "=" * 60)
    print(">>> 3. Real-Time Token Output Simulation")
    print("=" * 60)
    print("Streaming tokens to terminal: ", end="", flush=True)

    tokens = ["LangGraph ", "provides ", "first-class ", "support ", "for ", "token ", "streaming!"]
    for tok in tokens:
        print(tok, end="", flush=True)
        time.sleep(0.08)

    print("\n\n" + "=" * 65)
    print("Lesson 29 Complete: All Production Streaming Modes Demonstrated!")
    print("=" * 65)


if __name__ == "__main__":
    main()
