"""
=============================================================================
Phase 1: Core Graph Patterns - Lesson 4: Serial (Sequential) Graph
=============================================================================

Pattern:
START -> Node A -> Node B -> Node C -> END

Characteristics:
- A linear pipeline where each node executes in strict sequential order.
- Each node receives the accumulated state, performs its task, and returns
  updates that modify the state for subsequent nodes.
- Ideal for data extraction -> cleaning -> transformation -> reporting.
=============================================================================
"""

import sys
import io
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

# Ensure proper utf-8 encoding for Windows terminals
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


# --- 1. Define State Schema ---
class PipelineState(TypedDict):
    raw_text: str
    cleaned_text: str
    word_count: int
    summary: str


# --- 2. Define Sequential Nodes ---
def clean_text_node(state: PipelineState) -> dict:
    """Step 1: Normalizes whitespace and strips unwanted characters."""
    print("\n[Step 1: clean_text_node] Cleaning raw input text...")
    cleaned = " ".join(state["raw_text"].strip().split())
    return {"cleaned_text": cleaned}


def count_words_node(state: PipelineState) -> dict:
    """Step 2: Analyzes cleaned text and counts total words."""
    print("[Step 2: count_words_node] Counting words in cleaned text...")
    words = state["cleaned_text"].split()
    return {"word_count": len(words)}


def generate_summary_node(state: PipelineState) -> dict:
    """Step 3: Creates final summary report based on earlier calculations."""
    print("[Step 3: generate_summary_node] Creating final summary report...")
    summary = (
        f"Document Summary: {state['word_count']} words processed. "
        f"Preview: '{state['cleaned_text'][:40]}...'"
    )
    return {"summary": summary}


def main():
    print("=" * 65)
    print("  LangGraph Pattern: Serial (Sequential) Execution Flow")
    print("=" * 65)

    # --- 3. Assemble the Serial Graph ---
    graph = StateGraph(PipelineState)

    # Add all three nodes
    graph.add_node("clean_text", clean_text_node)
    graph.add_node("count_words", count_words_node)
    graph.add_node("generate_summary", generate_summary_node)

    # Connect nodes in strict linear sequence:
    # START -> clean_text -> count_words -> generate_summary -> END
    graph.add_edge(START, "clean_text")
    graph.add_edge("clean_text", "count_words")
    graph.add_edge("count_words", "generate_summary")
    graph.add_edge("generate_summary", END)

    # --- 4. Compile the Graph ---
    app = graph.compile()

    # --- 5. Execute with Sample Input ---
    initial_input: PipelineState = {
        "raw_text": "   LangGraph makes it   easy to build   stateful multi-agent workflows.   ",
        "cleaned_text": "",
        "word_count": 0,
        "summary": ""
    }

    print(f"Initial State:\n  raw_text = '{initial_input['raw_text']}'")

    final_state = app.invoke(initial_input)

    print("\n--- Final Accumulated State ---")
    for key, value in final_state.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
