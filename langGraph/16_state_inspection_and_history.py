"""
=============================================================================
Phase 3: Persistence & Memory - Lesson 16: State Inspection & History
=============================================================================

Key Concepts:
1. `app.get_state(config)`:
   Retrieves the current `StateSnapshot` for a given thread:
   - `snapshot.values`: The active state dictionary.
   - `snapshot.next`: Tuple of node names scheduled to run next.
   - `snapshot.config`: Contains checkpoint_id and thread_id.
   - `snapshot.metadata`: Information regarding which node created this checkpoint.
2. `app.get_state_history(config)`:
   An iterator yielding every snapshot in the thread's chronological timeline.
   LangGraph stores checkpoints as an immutable, event-sourced audit log.
=============================================================================
"""

import sys
import io
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver

# Ensure proper utf-8 encoding for Windows terminals
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


# --- 1. Nodes with Intermediate Milestones ---
def step_analyzer_node(state: MessagesState) -> dict:
    print("  [Step 1: Analyzer] Extracting intent...")
    return {"messages": [AIMessage(content="[Analysis]: User request validated.")]}


def step_responder_node(state: MessagesState) -> dict:
    print("  [Step 2: Responder] Generating final answer...")
    return {"messages": [AIMessage(content="[Response]: Here is the requested report.")]}


def main():
    print("=" * 65)
    print("  LangGraph Phase 3: Lesson 16 - State Inspection & History Audit")
    print("=" * 65)

    # --- 2. Assemble Graph ---
    graph = StateGraph(MessagesState)
    graph.add_node("analyzer", step_analyzer_node)
    graph.add_node("responder", step_responder_node)

    graph.add_edge(START, "analyzer")
    graph.add_edge("analyzer", "responder")
    graph.add_edge("responder", END)

    checkpointer = MemorySaver()
    app = graph.compile(checkpointer=checkpointer)

    thread_config = {"configurable": {"thread_id": "audit_session_404"}}

    # Execute Turn 1
    print("\n--- Invoking Step Sequence ---")
    app.invoke(
        {"messages": [HumanMessage(content="Generate monthly analytics report.")]},
        config=thread_config
    )

    # --- 3. Inspect Current State Snapshot ---
    print("\n" + "=" * 60)
    print(">>> 1. Inspecting Current State with app.get_state()")
    print("=" * 60)

    current_snapshot = app.get_state(thread_config)

    print(f"Next Nodes to Run : {current_snapshot.next}")  # Empty tuple () when graph has ended
    print(f"Checkpoint ID     : {current_snapshot.config['configurable'].get('checkpoint_id')}")
    print(f"Created By Step   : {current_snapshot.metadata.get('step')}")
    print(f"Total Messages    : {len(current_snapshot.values['messages'])}")

    print("\nCurrent Message Stack:")
    for idx, msg in enumerate(current_snapshot.values["messages"], start=1):
        print(f"  {idx}. [{msg.__class__.__name__}]: {msg.content}")

    # --- 4. Inspect State History Timeline ---
    print("\n" + "=" * 60)
    print(">>> 2. Traversing Complete State History Timeline with app.get_state_history()")
    print("=" * 60)

    # get_state_history yields snapshots from newest to oldest
    timeline = list(app.get_state_history(thread_config))
    print(f"Total Checkpoints Recorded in Thread History: {len(timeline)}\n")

    for i, snapshot in enumerate(timeline, start=1):
        step_source = snapshot.metadata.get("source", "unknown")
        step_num = snapshot.metadata.get("step", "N/A")
        ckpt_id = snapshot.config["configurable"].get("checkpoint_id")
        msg_count = len(snapshot.values.get("messages", []))

        print(f"Checkpoint #{i}:")
        print(f"  - Checkpoint ID : {ckpt_id}")
        print(f"  - Step # / Source: {step_num} / {step_source}")
        print(f"  - Messages at this instant: {msg_count}")
        print(f"  - Next Nodes Scheduled   : {snapshot.next}")
        print()


if __name__ == "__main__":
    main()
