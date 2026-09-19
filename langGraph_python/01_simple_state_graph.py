"""
=============================================================================
Phase 1: Core Fundamentals - Lesson 1: The Basic StateGraph (Pure Python)
=============================================================================

Key Concepts:
1. State: A TypedDict that acts as the single source of truth across the graph.
2. Nodes: Regular Python functions that receive the current State and return
          a dictionary with updated keys.
3. Edges: Connect nodes together (START -> NodeA -> NodeB -> END).
4. Compilation: graph.compile() turns the graph definition into a runnable.
=============================================================================
"""

import sys
import io
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

# Ensure proper utf-8 encoding for Windows terminals
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


# --- 1. Define the State Schema ---
# The state is shared across all nodes. Each node can read from it and write to it.
class SimpleState(TypedDict):
    user_name: str
    greeting: str
    step_count: int


# --- 2. Define the Nodes ---
# A node is simply a function: input is `state`, output is a dict of updates.
def welcome_node(state: SimpleState) -> dict:
    print(f"\n[Node: welcome_node] Processing for user: {state['user_name']}")
    return {
        "greeting": f"Hello, {state['user_name']}! Welcome to LangGraph.",
        "step_count": state.get("step_count", 0) + 1
    }


def uppercase_node(state: SimpleState) -> dict:
    print("\n[Node: uppercase_node] Transforming greeting to uppercase...")
    # Read the current greeting and transform it
    transformed = state["greeting"].upper()
    return {
        "greeting": transformed,
        "step_count": state["step_count"] + 1
    }


def main():
    print("=" * 60)
    print("  LangGraph Phase 1: Lesson 1 - Basic StateGraph Flow")
    print("=" * 60)

    # --- 3. Build the Graph ---
    # Create the StateGraph passing the State class schema
    graph = StateGraph(SimpleState)

    # Add nodes to the graph: (name, function)
    graph.add_node("welcome", welcome_node)
    graph.add_node("uppercase", uppercase_node)

    # Add edges to define the execution flow:
    # START -> welcome -> uppercase -> END
    graph.add_edge(START, "welcome")
    graph.add_edge("welcome", "uppercase")
    graph.add_edge("uppercase", END)

    # --- 4. Compile the Graph ---
    # Compiling validates the graph structure and returns a CompiledStateGraph runnable
    app = graph.compile()

    # --- 5. Invoke the Graph ---
    initial_input: SimpleState = {
        "user_name": "Soumya",
        "greeting": "",
        "step_count": 0
    }

    print("\n--- Invoking Graph with Initial State ---")
    print(f"Initial State: {initial_input}")

    final_state = app.invoke(initial_input)

    print("\n--- Execution Complete ---")
    print("Final State Result:")
    for key, value in final_state.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
