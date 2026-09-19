"""
=============================================================================
Phase 3: Persistence & Memory - Lesson 15: SQLite Disk Persistence
=============================================================================

Key Concepts:
1. Persistent Checkpointers:
   `MemorySaver` is lost when the Python process exits. In production,
   `SqliteSaver` persists state across restarts, crashes, and redeployments.
2. The Database Setup Gotcha:
   `SqliteSaver` creates tables (`checkpoints`, `checkpoint_blobs`, `checkpoint_writes`).
   The recommended pattern is either:
   - Using `with SqliteSaver.from_conn_string(DB_PATH) as checkpointer:` (context manager)
   - Or calling `checkpointer.setup()` if passing a raw `sqlite3.Connection`.
3. Standalone Fallback:
   This lesson includes both the official `SqliteSaver` implementation and a
   zero-dependency fallback so the script runs reliably in any environment.
=============================================================================
"""

import os
import sys
import io
import sqlite3
import json
from typing import Optional
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, MessagesState, START, END

# Ensure proper utf-8 encoding for Windows terminals
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

DB_PATH = os.path.join(os.path.dirname(__file__), "state_checkpoints.db")


# --- 1. Define Agent Node ---
def support_agent_node(state: MessagesState) -> dict:
    history = state["messages"]
    last_msg = history[-1].content
    print(f"\n  [Agent Node] Current Turn: '{last_msg}'")
    print(f"  [Agent Node] Total conversation history length: {len(history)} messages")

    # Inspect prior context
    past_text = " ".join([m.content for m in history[:-1]])

    if "ticket" in last_msg.lower():
        return {"messages": [AIMessage(content="Ticket #8849 has been created for your issue.")]}
    elif "status" in last_msg.lower():
        if "#8849" in past_text or "ticket" in past_text:
            return {"messages": [AIMessage(content="Your Ticket #8849 is currently marked as 'In Progress' by Tier 2 support.")]}
        return {"messages": [AIMessage(content="I could not find an active ticket in this conversation thread.")]}
    else:
        return {"messages": [AIMessage(content=f"Received: '{last_msg}'. How may I assist your account?")]}


def assemble_graph(checkpointer):
    """Builds and compiles the graph with the given checkpointer."""
    graph = StateGraph(MessagesState)
    graph.add_node("agent", support_agent_node)
    graph.add_edge(START, "agent")
    graph.add_edge("agent", END)
    return graph.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Zero-dependency SQLite Fallback (Runs if langgraph-checkpoint-sqlite is absent)
# ---------------------------------------------------------------------------
class SimpleSqliteCheckpointer:
    """
    A lightweight, pure-Python SQLite checkpointer demonstrating how
    disk persistence stores and restores serialized state across runs.
    """
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS session_checkpoints (
                    thread_id TEXT PRIMARY KEY,
                    messages_json TEXT
                )
            """)
            conn.commit()

    def save(self, thread_id: str, messages: list):
        serialized = []
        for m in messages:
            serialized.append({"type": m.__class__.__name__, "content": m.content})
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO session_checkpoints (thread_id, messages_json) VALUES (?, ?)",
                (thread_id, json.dumps(serialized))
            )
            conn.commit()

    def load(self, thread_id: str) -> list:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("SELECT messages_json FROM session_checkpoints WHERE thread_id = ?", (thread_id,))
            row = cur.fetchone()
            if not row:
                return []
            raw = json.loads(row[0])
            restored = []
            for item in raw:
                if item["type"] == "HumanMessage":
                    restored.append(HumanMessage(content=item["content"]))
                else:
                    restored.append(AIMessage(content=item["content"]))
            return restored


def run_with_official_sqlite():
    """Runs persistence using official SqliteSaver."""
    from langgraph.checkpoint.sqlite import SqliteSaver

    thread_id = "customer_session_77"
    config = {"configurable": {"thread_id": thread_id}}

    print("\n" + "=" * 60)
    print(">>> PROCESS RUN #1: Customer opens support ticket (using SqliteSaver)")
    print("=" * 60)

    # Use the official context manager pattern: handles setup & tables automatically
    with SqliteSaver.from_conn_string(DB_PATH) as checkpointer:
        app1 = assemble_graph(checkpointer)

        print("\nCustomer: 'Please open a ticket for my payment failure.'")
        res1 = app1.invoke(
            {"messages": [HumanMessage(content="Please open a ticket for my payment failure.")]},
            config=config
        )
        print(f"Agent: {res1['messages'][-1].content}")

    print("\n[Simulated System Event]: Process Run #1 exited. DB connection closed.")

    print("\n" + "=" * 60)
    print(">>> PROCESS RUN #2: Fresh process reconnects to the SQLite database")
    print("=" * 60)

    # Reconnect in a brand new context
    with SqliteSaver.from_conn_string(DB_PATH) as checkpointer:
        app2 = assemble_graph(checkpointer)

        print("\nCustomer: 'What is the status of my ticket?'")
        res2 = app2.invoke(
            {"messages": [HumanMessage(content="What is the status of my ticket?")]},
            config=config
        )
        print(f"Agent: {res2['messages'][-1].content}")

        print("\nPersisted Full History restored from SQLite:")
        for msg in res2["messages"]:
            print(f"  [{msg.__class__.__name__}]: {msg.content}")


def run_with_fallback_sqlite():
    """Runs persistence using pure Python sqlite3 checkpointer."""
    thread_id = "customer_session_77"
    store = SimpleSqliteCheckpointer(DB_PATH)

    print("\n" + "=" * 60)
    print(">>> PROCESS RUN #1: Customer opens ticket (SQLite fallback)")
    print("=" * 60)

    from langgraph.checkpoint.memory import MemorySaver
    mem_checkpointer1 = MemorySaver()
    app1 = assemble_graph(mem_checkpointer1)
    config = {"configurable": {"thread_id": thread_id}}

    print("\nCustomer: 'Please open a ticket for my payment failure.'")
    res1 = app1.invoke(
        {"messages": [HumanMessage(content="Please open a ticket for my payment failure.")]},
        config=config
    )
    print(f"Agent: {res1['messages'][-1].content}")

    # Persist snapshot to SQLite disk table
    store.save(thread_id, res1["messages"])
    print("\n[Saved]: Checkpoint committed to SQLite database on disk.")
    del app1

    print("\n" + "=" * 60)
    print(">>> PROCESS RUN #2: Resuming from disk in new process")
    print("=" * 60)

    # Load persisted history from disk
    restored_history = store.load(thread_id)
    print(f"[Loaded]: Retrieved {len(restored_history)} historical messages from SQLite file.")

    mem_checkpointer2 = MemorySaver()
    app2 = assemble_graph(mem_checkpointer2)

    # Prime new session with restored messages
    print("\nCustomer: 'What is the status of my ticket?'")
    res2 = app2.invoke(
        {"messages": restored_history + [HumanMessage(content="What is the status of my ticket?")]},
        config=config
    )
    print(f"Agent: {res2['messages'][-1].content}")

    print("\nRestored Conversation History:")
    for msg in res2["messages"]:
        print(f"  [{msg.__class__.__name__}]: {msg.content}")


def main():
    print("=" * 65)
    print("  LangGraph Phase 3: Lesson 15 - SQLite Disk Persistence")
    print("=" * 65)

    has_official_sqlite = False
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
        has_official_sqlite = True
    except ImportError:
        pass

    if has_official_sqlite:
        print("Found official 'langgraph-checkpoint-sqlite'. Using SqliteSaver.")
        run_with_official_sqlite()
    else:
        print("Note: 'langgraph-checkpoint-sqlite' not found in environment.")
        print("Running with zero-dependency built-in SQLite persistence.\n")
        run_with_fallback_sqlite()

    # Clean up demo database file
    if os.path.exists(DB_PATH):
        try:
            os.remove(DB_PATH)
            print(f"\n[Cleanup]: Removed temporary demo database '{DB_PATH}'")
        except Exception:
            pass


if __name__ == "__main__":
    main()
