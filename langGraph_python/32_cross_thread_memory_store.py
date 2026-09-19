"""
=============================================================================
Phase 7: Advanced Production Patterns - Lesson 32: Cross-Thread Memory Store
=============================================================================

Key Concepts:
1. Thread Memory vs. Long-Term Cross-Thread Memory:
   - Checkpointers (`MemorySaver`, `SqliteSaver`) isolate memory within ONE `thread_id`.
   - The LangGraph `Store` (`InMemoryStore`) stores global, user-level or
     organization-level memories that persist ACROSS different threads!
2. Namespaced Key-Value Storage:
   Memories are organized into hierarchical tuples, e.g.:
   `store.put(namespace=("users", user_id), key="profile", value={...})`
3. Accessing Store in Nodes:
   When a node declares a `store` argument:
   `def my_node(state: MyState, store: BaseStore) -> dict:`
   LangGraph injects the compiled store automatically.
=============================================================================
"""

import sys
import io
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver

# Ensure proper utf-8 encoding for Windows terminals
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


# --- 1. Custom State with User ID ---
class UserProfileState(MessagesState):
    user_id: str


# --- 2. Define Node with Injected Store ---
def memory_agent_node(state: UserProfileState, store) -> dict:
    user_id = state.get("user_id", "default_user")
    namespace = ("users", user_id)
    last_query = state["messages"][-1].content
    print(f"\n[Memory Agent] Processing query for user '{user_id}': '{last_query}'")

    # Step A: Check if we have cross-thread memories for this user
    existing_mem = store.get(namespace, "preferences")
    if existing_mem:
        prefs = existing_mem.value
        print(f"  [Store Hit]: Found cross-thread profile: {prefs}")
    else:
        prefs = {}
        print("  [Store Miss]: No prior cross-thread profile found.")

    # Step B: Check if user is sharing a new preference
    if "prefer" in last_query.lower() or "like" in last_query.lower():
        if "python" in last_query.lower():
            prefs["fav_language"] = "Python"
        if "concise" in last_query.lower():
            prefs["style"] = "Concise & Direct"

        # Save to cross-thread long term store
        store.put(namespace, "preferences", prefs)
        print(f"  --> [Store Saved]: Committed preferences to cross-thread namespace {namespace}")
        reply = f"Noted! I have saved your preferences {prefs} across all your future conversation threads."
    else:
        # Utilize cross-thread memory in answer
        fav = prefs.get("fav_language", "unspecified")
        style = prefs.get("style", "standard")
        reply = f"Hello {user_id}! (Recalled Profile: Language={fav}, Style={style}). How can I help you in this new session?"

    return {"messages": [AIMessage(content=reply)]}


def main():
    print("=" * 65)
    print("  LangGraph Advanced: Lesson 32 - Cross-Thread Memory (Store)")
    print("=" * 65)

    # Check for LangGraph store support
    try:
        from langgraph.store.memory import InMemoryStore
    except ImportError:
        print("Note: LangGraph Store is available in langgraph >= 0.2.0.")
        return

    # --- 3. Assemble Graph with Both Checkpointer and Store ---
    graph = StateGraph(UserProfileState)
    graph.add_node("agent", memory_agent_node)
    graph.add_edge(START, "agent")
    graph.add_edge("agent", END)

    checkpointer = MemorySaver()
    global_store = InMemoryStore()

    # Compile with both checkpointer (per-thread) and store (cross-thread)
    app = graph.compile(checkpointer=checkpointer, store=global_store)

    user_alice = "alice_007"

    # --- Thread 1: Alice teaches the bot her preferences ---
    print("\n" + "=" * 60)
    print(">>> SESSION 1 (thread_id: 'thread_january'): Storing Preferences")
    print("=" * 60)

    config_thread_1 = {
        "configurable": {
            "thread_id": "thread_january"
        }
    }

    t1_res = app.invoke(
        {
            "user_id": user_alice,
            "messages": [HumanMessage(content="Please remember: I prefer Python code and concise explanations.")]
        },
        config=config_thread_1
    )
    print(f"Agent: {t1_res['messages'][-1].content}")

    # --- Thread 2: Alice starts a completely NEW thread months later ---
    print("\n" + "=" * 60)
    print(">>> SESSION 2 (thread_id: 'thread_march'): Brand New Thread")
    print("=" * 60)
    print("Notice: Thread history is empty, but cross-thread Store remembers Alice!")

    config_thread_2 = {
        "configurable": {
            "thread_id": "thread_march"
        }
    }

    t2_res = app.invoke(
        {
            "user_id": user_alice,
            "messages": [HumanMessage(content="Hello assistant! Can you help me today?")]
        },
        config=config_thread_2
    )
    print(f"Agent: {t2_res['messages'][-1].content}")


if __name__ == "__main__":
    main()
