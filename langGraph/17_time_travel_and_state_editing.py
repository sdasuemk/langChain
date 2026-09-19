"""
=============================================================================
Phase 3: Persistence & Memory - Lesson 17: Time Travel & State Editing
=============================================================================

Key Concepts:
1. State Editing (`app.update_state`):
   Allows you to manually inject, modify, or correct state variables in an active
   thread without rerunning the entire conversation.
2. The `as_node` Parameter:
   When calling `update_state(config, updates, as_node="node_name")`, LangGraph
   treats the update as if it were produced by that node, which determines
   which node executes next according to the graph's edges.
3. Time Travel (Forking from Past Checkpoints):
   By passing a specific historical checkpoint ID into `app.invoke()` or
   `app.update_state()`, you fork execution from that exact historical moment,
   exploring an alternate timeline while preserving the original record!
=============================================================================
"""

import sys
import io
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver

# Ensure proper utf-8 encoding for Windows terminals
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


# --- 1. Define Nodes ---
def generate_draft_node(state: MessagesState) -> dict:
    last_msg = state["messages"][-1].content
    print(f"\n  [Draft Node] Drafting based on: '{last_msg}'")
    return {"messages": [AIMessage(content=f"Initial Draft: Plan for '{last_msg}'.")]}


def review_draft_node(state: MessagesState) -> dict:
    last_ai_msg = state["messages"][-1].content
    print(f"  [Review Node] Reviewing: '{last_ai_msg}'")
    return {"messages": [AIMessage(content=f"Final Polish: {last_ai_msg} - Approved.")]}


def main():
    print("=" * 65)
    print("  LangGraph Phase 3: Lesson 17 - Time Travel & State Editing")
    print("=" * 65)

    # --- 2. Assemble Graph ---
    graph = StateGraph(MessagesState)
    graph.add_node("draft", generate_draft_node)
    graph.add_node("review", review_draft_node)

    graph.add_edge(START, "draft")
    graph.add_edge("draft", "review")
    graph.add_edge("review", END)

    checkpointer = MemorySaver()
    app = graph.compile(checkpointer=checkpointer)

    thread_config = {"configurable": {"thread_id": "time_travel_demo"}}

    # --- 3. Initial Run ---
    print("\n>>> 1. Running Initial Workflow")
    initial_res = app.invoke(
        {"messages": [HumanMessage(content="Launch Product Alpha in Q3")]},
        config=thread_config
    )

    print("\nInitial Run Completed. Messages:")
    for m in initial_res["messages"]:
        print(f"  [{m.__class__.__name__}]: {m.content}")

    # --- 4. State Editing: Overriding State with update_state ---
    print("\n" + "=" * 60)
    print(">>> 2. State Editing with app.update_state()")
    print("=" * 60)

    # Simulate a human editor modifying the state before continuing
    corrected_message = AIMessage(
        content="Human Correction: Product Alpha will launch in Q4 with expanded budget."
    )

    # Update state treating it as if 'draft' produced it
    app.update_state(
        thread_config,
        {"messages": [corrected_message]},
        as_node="draft"
    )

    print("Updated state manually. Now re-running review node on the updated state:")
    # Re-invoke with None to continue from current state
    res_after_edit = app.invoke(None, config=thread_config)

    print("\nMessages after State Editing:")
    for m in res_after_edit["messages"]:
        print(f"  [{m.__class__.__name__}]: {m.content}")

    # --- 5. Time Travel: Forking from an Earlier Historical Checkpoint ---
    print("\n" + "=" * 60)
    print(">>> 3. Time Travel: Forking from an Early Checkpoint")
    print("=" * 60)

    # Retrieve all historical checkpoints
    history = list(app.get_state_history(thread_config))
    # Pick the earliest snapshot (when the user just submitted the first query)
    earliest_checkpoint = history[-1]
    earliest_config = earliest_checkpoint.config

    print(f"Targeting Earliest Checkpoint ID: {earliest_config['configurable']['checkpoint_id']}")
    print(f"Messages at that moment: {len(earliest_checkpoint.values.get('messages', []))}")

    # Fork an alternate reality from that point by submitting a new message to that config
    print("\nForking alternative path from the beginning...")
    fork_config = earliest_config.copy()

    forked_res = app.invoke(
        {"messages": [HumanMessage(content="Pivot: Cancel Alpha, Launch Product Beta instead!")]},
        config=fork_config
    )

    print("\nAlternative Timeline Result:")
    for m in forked_res["messages"]:
        print(f"  [{m.__class__.__name__}]: {m.content}")


if __name__ == "__main__":
    main()
