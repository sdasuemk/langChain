"""
=============================================================================
Phase 6: Multi-Agent Systems & Production - Lesson 26: Supervisor Architecture
=============================================================================

Pattern:
                     START
                       │
                       ▼
                ┌──────────────┐
        ┌──────►│  supervisor  │◄──────┐
        │       └──────┬───────┘       │
        │              │               │
        │      [next_worker_router]    │
        │        /            \        │
        │   "researcher"    "coder"    │
        │       │              │       │
        │       ▼              ▼       │
        │  research_node   code_node   │
        │       │              │       │
        └───────┴──────────────┴───────┘
                       │
                   ("FINISH")
                       │
                       ▼
                      END

Key Concepts:
1. Centralized Orchestration:
   A central "Supervisor" agent acts as the project manager. It assesses the
   overall user goal, inspects worker progress, and decides which specialized
   worker to assign next.
2. Worker Specialization:
   Workers focus purely on their distinct skill domain (e.g. Research, Coding, Writing)
   and always hand their completed work back to the supervisor.
=============================================================================
"""

import sys
import io
from typing import List, Literal
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END

# Ensure proper utf-8 encoding for Windows terminals
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


# --- 1. Define Multi-Agent State ---
class TeamState(TypedDict):
    task: str
    messages: List[str]
    next_worker: str
    iteration: int


# --- 2. Supervisor Node ---
def supervisor_node(state: TeamState) -> dict:
    iter_count = state.get("iteration", 0) + 1
    messages = state.get("messages", [])

    print(f"\n[Supervisor] Reviewing project status (Cycle #{iter_count})...")

    # Has research been completed?
    has_research = any("Research Findings" in m for m in messages)
    # Has code been completed?
    has_code = any("Code Implementation" in m for m in messages)

    if not has_research:
        next_worker = "researcher"
        decision_rationale = "Assigning task to Researcher to gather specs."
    elif not has_code:
        next_worker = "coder"
        decision_rationale = "Research complete. Assigning task to Coder to build module."
    else:
        next_worker = "FINISH"
        decision_rationale = "All deliverables satisfied. Concluding project."

    print(f"  --> Decision: {decision_rationale} -> Next: [{next_worker}]")
    return {
        "next_worker": next_worker,
        "iteration": iter_count
    }


# --- 3. Specialized Worker Nodes ---
def researcher_node(state: TeamState) -> dict:
    print("\n  [Worker: Researcher] Conducting research on requirements...")
    finding = "Research Findings: High-performance caching requires Redis with LRU eviction policy."
    return {
        "messages": state["messages"] + [finding]
    }


def coder_node(state: TeamState) -> dict:
    print("\n  [Worker: Coder] Writing code based on research findings...")
    implementation = (
        "Code Implementation:\n"
        "import redis\n"
        "cache = redis.Redis(host='localhost', port=6379, maxmemory_policy='allkeys-lru')"
    )
    return {
        "messages": state["messages"] + [implementation]
    }


# --- 4. Supervisor Conditional Router ---
def route_supervisor(state: TeamState) -> str:
    return state["next_worker"]


def main():
    print("=" * 65)
    print("  LangGraph Phase 6: Lesson 26 - Supervisor Multi-Agent Team")
    print("=" * 65)

    # --- 5. Assemble Multi-Agent Graph ---
    graph = StateGraph(TeamState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("coder", coder_node)

    # Entry point is always the supervisor
    graph.add_edge(START, "supervisor")

    # Supervisor delegates to workers or finishes
    graph.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {
            "researcher": "researcher",
            "coder": "coder",
            "FINISH": END
        }
    )

    # Workers always report back to the supervisor
    graph.add_edge("researcher", "supervisor")
    graph.add_edge("coder", "supervisor")

    app = graph.compile()

    # --- 6. Run Project Workflow ---
    initial_input: TeamState = {
        "task": "Build a scalable in-memory caching service for user sessions",
        "messages": [],
        "next_worker": "",
        "iteration": 0
    }

    print(f"Initial Goal: '{initial_input['task']}'")

    final_state = app.invoke(initial_input)

    print("\n" + "=" * 65)
    print("Team Project Completed! Final Accumulated Deliverables:")
    print("=" * 65)
    for idx, msg in enumerate(final_state["messages"], start=1):
        print(f"\n--- Deliverable {idx} ---\n{msg}")


if __name__ == "__main__":
    main()
