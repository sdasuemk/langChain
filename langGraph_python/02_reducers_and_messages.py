"""
=============================================================================
Phase 1: Core Fundamentals - Lesson 2: State Reducers & Message Handling
=============================================================================

Key Concepts:
1. Overwrite vs. Append:
   By default, returning a key from a node replaces the existing value in state.
2. Reducers:
   Using `Annotated[type, reducer_function]`, you tell LangGraph HOW to merge updates.
   - `Annotated[list, operator.add]`: Standard list concatenation (appends items).
   - `Annotated[list, add_messages]`: Specialized reducer for chat messages.
     - Appends new messages.
     - Updates existing messages if they share the same ID.
3. Prebuilt `MessagesState`:
   A shortcut provided by LangGraph that already defines `messages: Annotated[list[AnyMessage], add_messages]`.
=============================================================================
"""

import sys
import io
import operator
from typing import Annotated, List
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# Ensure proper utf-8 encoding for Windows terminals
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


# ---------------------------------------------------------------------------
# Part A: Demonstrating List Append Reducer (operator.add)
# ---------------------------------------------------------------------------
class CustomReducerState(TypedDict):
    # 'operator.add' tells LangGraph: state['logs'] = state['logs'] + new_logs
    logs: Annotated[List[str], operator.add]
    # Without Annotated, 'current_status' will be completely overwritten by each node
    current_status: str


def node_step_one(state: CustomReducerState) -> dict:
    return {
        "logs": ["Step 1 completed: Initialized environment."],
        "current_status": "Step 1 Done"
    }


def node_step_two(state: CustomReducerState) -> dict:
    return {
        "logs": ["Step 2 completed: Processed incoming payload."],
        "current_status": "Step 2 Done"
    }


# ---------------------------------------------------------------------------
# Part B: Demonstrating the 'add_messages' Reducer for Chat History
# ---------------------------------------------------------------------------
class ChatState(TypedDict):
    # 'add_messages' handles HumanMessage, AIMessage, ToolMessage, etc.
    messages: Annotated[List, add_messages]


def assistant_node(state: ChatState) -> dict:
    last_message = state["messages"][-1]
    print(f"\n[Assistant Node] Received prompt from user: '{last_message.content}'")
    
    # Simulate generating an AI response
    reply = AIMessage(
        content=f"Echoing your thought: '{last_message.content}'. How can I assist further?"
    )
    # Returning a message appends it to state['messages'] via add_messages
    return {"messages": [reply]}


def main():
    print("=" * 65)
    print("  LangGraph Phase 1: Lesson 2 - Reducers & Message Handling")
    print("=" * 65)

    # --- Test Part A: Custom operator.add Reducer ---
    print("\n>>> PART A: Demonstrating operator.add reducer for log lists")
    graph_a = StateGraph(CustomReducerState)
    graph_a.add_node("step_one", node_step_one)
    graph_a.add_node("step_two", node_step_two)
    
    graph_a.add_edge(START, "step_one")
    graph_a.add_edge("step_one", "step_two")
    graph_a.add_edge("step_two", END)
    
    app_a = graph_a.compile()

    result_a = app_a.invoke({"logs": ["System Boot."], "current_status": "Started"})
    print("\nFinal Logs list (appended by reducer, NOT overwritten):")
    for log in result_a["logs"]:
        print(f"  - {log}")
    print(f"Final Status (overwritten): {result_a['current_status']}")

    # --- Test Part B: Chat Messages with add_messages ---
    print("\n" + "-" * 65)
    print(">>> PART B: Demonstrating add_messages reducer for chat conversations")
    graph_b = StateGraph(ChatState)
    graph_b.add_node("assistant", assistant_node)
    graph_b.add_edge(START, "assistant")
    graph_b.add_edge("assistant", END)
    
    app_b = graph_b.compile()

    # Turn 1
    input_turn_1 = {
        "messages": [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="Hello! I am learning LangGraph.")
        ]
    }
    result_turn_1 = app_b.invoke(input_turn_1)

    print("\nConversation State after Turn 1:")
    for msg in result_turn_1["messages"]:
        print(f"  [{msg.__class__.__name__}]: {msg.content}")

    # Turn 2: Provide new message to continue the history
    input_turn_2 = {
        "messages": result_turn_1["messages"] + [HumanMessage(content="Explain what a state graph is.")]
    }
    result_turn_2 = app_b.invoke(input_turn_2)

    print("\nConversation State after Turn 2:")
    for msg in result_turn_2["messages"]:
        print(f"  [{msg.__class__.__name__}]: {msg.content}")


if __name__ == "__main__":
    main()
