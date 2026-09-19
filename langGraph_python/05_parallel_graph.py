"""
=============================================================================
Phase 1: Core Graph Patterns - Lesson 5: Parallel (Fan-Out / Fan-In) Graph
=============================================================================

Pattern:
                  START
                    |
               input_reader
                 /     \        (Fan-Out: Concurrent execution)
                /       \
       analyze_sentiment  extract_keywords
                \       /
                 \     /        (Fan-In: Aggregator waits for both)
                aggregator
                    |
                   END

Characteristics:
- Fan-Out: A single node triggers multiple outgoing edges simultaneously.
- Fan-In: The downstream node (aggregator) waits until ALL incoming parallel
  branches finish before it executes (barrier synchronization).
- State Safety:
  - If parallel nodes update DIFFERENT keys (e.g. 'sentiment' and 'keywords'),
    no conflict occurs.
  - If parallel nodes write to the SAME key, you MUST use an Annotated reducer
    (like Annotated[List[str], operator.add]) to cleanly merge their outputs.
=============================================================================
"""

import sys
import io
import operator
from typing import Annotated, List
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

# Ensure proper utf-8 encoding for Windows terminals
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


# --- 1. Define State with Reducers for Parallel Writes ---
class ParallelState(TypedDict):
    customer_review: str
    sentiment: str
    keywords: List[str]
    # Shared execution audit log: multiple parallel nodes will append to this simultaneously
    activity_log: Annotated[List[str], operator.add]
    final_report: str


# --- 2. Define Nodes ---
def input_reader_node(state: ParallelState) -> dict:
    print("\n[Node: input_reader] Validating customer review...")
    return {
        "activity_log": ["Input reader verified review."]
    }


def analyze_sentiment_node(state: ParallelState) -> dict:
    """Worker 1: Runs in parallel."""
    print("[Worker 1: analyze_sentiment] Evaluating sentiment...")
    review = state["customer_review"].lower()
    
    # Simple rule-based sentiment detection for demonstration
    if any(word in review for word in ["great", "love", "excellent", "best"]):
        detected = "Positive"
    elif any(word in review for word in ["bad", "terrible", "slow", "broken"]):
        detected = "Negative"
    else:
        detected = "Neutral"

    return {
        "sentiment": detected,
        "activity_log": [f"Sentiment analysis resolved to: {detected}"]
    }


def extract_keywords_node(state: ParallelState) -> dict:
    """Worker 2: Runs in parallel."""
    print("[Worker 2: extract_keywords] Extracting important keywords...")
    words = state["customer_review"].lower().replace(".", "").split()
    # Simple extraction: words longer than 5 chars
    found_keywords = [w for w in set(words) if len(w) > 5]

    return {
        "keywords": found_keywords,
        "activity_log": [f"Keyword extraction identified: {found_keywords}"]
    }


def aggregator_node(state: ParallelState) -> dict:
    """Aggregator: Executes only AFTER both parallel workers have finished."""
    print("\n[Aggregator: aggregator_node] Combining results from all parallel workers...")
    report = (
        f"Customer Feedback Analysis:\n"
        f"  - Sentiment: {state['sentiment']}\n"
        f"  - Keywords: {', '.join(state['keywords'])}\n"
        f"  - Audit Trail Steps: {len(state['activity_log'])}"
    )
    return {
        "final_report": report,
        "activity_log": ["Aggregator compiled final customer report."]
    }


def main():
    print("=" * 65)
    print("  LangGraph Pattern: Parallel (Fan-Out / Fan-In) Execution Flow")
    print("=" * 65)

    # --- 3. Assemble the Parallel Graph ---
    graph = StateGraph(ParallelState)

    # Register nodes
    graph.add_node("input_reader", input_reader_node)
    graph.add_node("analyze_sentiment", analyze_sentiment_node)
    graph.add_node("extract_keywords", extract_keywords_node)
    graph.add_node("aggregator", aggregator_node)

    # --- Setup Fan-Out ---
    graph.add_edge(START, "input_reader")
    # input_reader splits into two parallel branches:
    graph.add_edge("input_reader", "analyze_sentiment")
    graph.add_edge("input_reader", "extract_keywords")

    # --- Setup Fan-In ---
    # Both branches converge on aggregator:
    graph.add_edge("analyze_sentiment", "aggregator")
    graph.add_edge("extract_keywords", "aggregator")

    # Exit
    graph.add_edge("aggregator", END)

    # --- 4. Compile the Graph ---
    app = graph.compile()

    # --- 5. Run with Sample Review ---
    initial_input: ParallelState = {
        "customer_review": "The application is excellent and fast, but account setup was somewhat complicated.",
        "sentiment": "",
        "keywords": [],
        "activity_log": [],
        "final_report": ""
    }

    print(f"\nReview Input:\n  '{initial_input['customer_review']}'")

    final_state = app.invoke(initial_input)

    print("\n--- Aggregated Final Report ---")
    print(final_state["final_report"])

    print("\n--- Execution Audit Log (Merged via operator.add reducer) ---")
    for step in final_state["activity_log"]:
        print(f"  * {step}")


if __name__ == "__main__":
    main()
