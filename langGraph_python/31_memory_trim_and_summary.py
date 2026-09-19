"""
=============================================================================
Phase 7: Advanced Production Patterns - Lesson 31: Memory Trimming & Summarization
=============================================================================

Pattern:
                     START
                       │
                       ▼
                 check_context_length
                       │
             [needs_summarization?]
              /                  \
          (yes: > 6 msgs)     (no: <= 6 msgs)
             │                     │
             ▼                     │
       summarize_history           │
             │                     │
             └──────────┬──────────┘
                        │
                        ▼
                     chatbot
                        │
                       END

Key Concepts:
1. The Unbounded Memory Problem:
   In long-running threads, checkpointers accumulate hundreds of messages.
   Eventually, you exceed the LLM's context window and inference latency explodes.
2. Context Compaction via Summarization:
   When message history exceeds a threshold, an intermediate node condenses
   older messages into a running `summary` string and purges older raw messages.
3. Combining Summary with Recent Messages:
   The chatbot node receives: System Prompt + Running Summary + Last N Messages.
=============================================================================
"""

import sys
import io
from typing import List, Optional
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, RemoveMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver

# Ensure proper utf-8 encoding for Windows terminals
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


# --- 1. Define State with Summary Field ---
class MemoryManagedState(MessagesState):
    summary: str


# --- 2. Define Nodes ---
def summarize_history_node(state: MemoryManagedState) -> dict:
    messages = state["messages"]
    print(f"\n[Summarizer Node] History has {len(messages)} messages -> Compacting memory...")

    # Older messages to condense (everything except the last 2 messages)
    older_messages = messages[:-2]
    condensed_text = " ".join([f"{m.__class__.__name__}: {m.content}" for m in older_messages])

    new_summary = f"Summary of past dialogue: User and AI discussed [{condensed_text[:120]}...]"
    print(f"  --> Updated Summary: '{new_summary}'")

    # In LangGraph, returning RemoveMessage(id=...) cleans up past messages from state
    delete_actions = [RemoveMessage(id=m.id) for m in older_messages if hasattr(m, "id") and m.id]

    # If messages don't have explicit IDs, simply updating summary achieves the goal
    return {
        "summary": new_summary,
        "messages": delete_actions
    }


def chatbot_node(state: MemoryManagedState) -> dict:
    summary = state.get("summary", "")
    recent_messages = state["messages"]

    print(f"\n[Chatbot Node] Invoking assistant:")
    if summary:
        print(f"  Context Summary Active: '{summary}'")
    print(f"  Recent Raw Messages in Context: {len(recent_messages)}")

    last_user_query = recent_messages[-1].content
    reply = f"Acknowledged query: '{last_user_query}'. Memory managed efficiently!"
    return {"messages": [AIMessage(content=reply)]}


# --- 3. Routing Condition ---
def should_summarize(state: MemoryManagedState) -> str:
    """If conversation has more than 6 messages, trigger summarization."""
    if len(state["messages"]) > 6:
        print(f"\n--> [Router]: Message count ({len(state['messages'])}) > 6 -> Routing to SUMMARIZE.")
        return "summarize"
    return "chatbot"


def main():
    print("=" * 65)
    print("  LangGraph Advanced: Lesson 31 - Memory Trimming & Summarization")
    print("=" * 65)

    # --- 4. Assemble Graph ---
    graph = StateGraph(MemoryManagedState)

    graph.add_node("summarize", summarize_history_node)
    graph.add_node("chatbot", chatbot_node)

    # Check length at START
    graph.add_conditional_edges(
        START,
        should_summarize,
        {
            "summarize": "summarize",
            "chatbot": "chatbot"
        }
    )

    graph.add_edge("summarize", "chatbot")
    graph.add_edge("chatbot", END)

    checkpointer = MemorySaver()
    app = graph.compile(checkpointer=checkpointer)

    thread_config = {"configurable": {"thread_id": "long_session_999"}}

    # --- 5. Simulate Multi-Turn Conversation Crossing Threshold ---
    print("\nSimulating 4 conversation turns...")

    user_queries = [
        "Hi, I want to learn cloud architectures.",
        "Can you explain microservices vs monoliths?",
        "What are the best practices for container security?",
        "Now give me a brief recap of everything we discussed."
    ]

    for turn, query in enumerate(user_queries, start=1):
        print("\n" + "-" * 55)
        print(f"TURN {turn}: User Query = '{query}'")
        res = app.invoke(
            {"messages": [HumanMessage(content=query)]},
            config=thread_config
        )
        print(f"Assistant: {res['messages'][-1].content}")


if __name__ == "__main__":
    main()
