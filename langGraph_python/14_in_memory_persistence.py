"""
=============================================================================
Phase 3: Persistence & Memory - Lesson 14: In-Memory Checkpointing (MemorySaver)
=============================================================================

Key Concepts:
1. Checkpointer:
   A component that automatically saves graph state after every super-step.
   `MemorySaver` stores state in an in-memory dictionary.
2. `thread_id`:
   The unique session identifier passed in `config={"configurable": {"thread_id": "..."}}`.
   State is partitioned by `thread_id`, allowing you to support thousands of
   concurrent, isolated user conversations.
3. Automatic Context Resumption:
   You only need to pass NEW messages in subsequent turns; LangGraph retrieves
   past turns automatically from the checkpoint!
=============================================================================
"""

import sys
import io
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver

# Ensure proper utf-8 encoding for Windows terminals
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


# --- 1. Define Agent Node ---
def chatbot_node(state: MessagesState) -> dict:
    messages = state["messages"]
    last_user_msg = messages[-1].content
    print(f"\n  [Chatbot Node] Total messages in history: {len(messages)}")
    print(f"  [Chatbot Node] Current User Query: '{last_user_msg}'")

    # Simulate smart context awareness
    history_text = " ".join([m.content for m in messages[:-1]])

    if "name is" in last_user_msg.lower():
        extracted_name = last_user_msg.split("name is")[-1].strip().strip(".")
        return {"messages": [AIMessage(content=f"Pleased to meet you, {extracted_name}! I have committed your name to memory.")]}
    elif "my name" in last_user_msg.lower():
        if "Alice" in history_text:
            return {"messages": [AIMessage(content="Your name is Alice! I recall from our earlier message.")]}
        elif "Bob" in history_text:
            return {"messages": [AIMessage(content="Your name is Bob! I recall from our earlier message.")]}
        else:
            return {"messages": [AIMessage(content="You haven't told me your name in this conversation thread yet.")]}
    else:
        return {"messages": [AIMessage(content=f"I heard: '{last_user_msg}'. How can I help you today?")]}


def main():
    print("=" * 65)
    print("  LangGraph Phase 3: Lesson 14 - In-Memory Checkpointing")
    print("=" * 65)

    # --- 2. Assemble Graph with Checkpointer ---
    graph = StateGraph(MessagesState)
    graph.add_node("chatbot", chatbot_node)
    graph.add_edge(START, "chatbot")
    graph.add_edge("chatbot", END)

    # Initialize in-memory checkpointer
    checkpointer = MemorySaver()

    # Pass checkpointer to compile()
    app = graph.compile(checkpointer=checkpointer)

    # --- 3. Conversation Thread 1: Alice ---
    print("\n" + "=" * 60)
    print(">>> THREAD 1: Alice's Conversation")
    print("=" * 60)

    config_alice = {"configurable": {"thread_id": "thread_alice_101"}}

    # Turn 1: Alice introduces herself
    print("\n[Turn 1] Alice speaks:")
    turn1_input = {"messages": [HumanMessage(content="Hello! My name is Alice.")]}
    response1 = app.invoke(turn1_input, config=config_alice)
    print(f"Assistant: {response1['messages'][-1].content}")

    # Turn 2: Alice asks for her name WITHOUT resending previous messages!
    print("\n[Turn 2] Alice asks for her name (only sending the new question):")
    turn2_input = {"messages": [HumanMessage(content="Do you remember what my name is?")]}
    response2 = app.invoke(turn2_input, config=config_alice)
    print(f"Assistant: {response2['messages'][-1].content}")

    # --- 4. Conversation Thread 2: Bob (Thread Isolation) ---
    print("\n" + "=" * 60)
    print(">>> THREAD 2: Bob's Conversation (Isolated Session)")
    print("=" * 60)

    config_bob = {"configurable": {"thread_id": "thread_bob_202"}}

    # Turn 1: Bob asks if assistant knows his name
    print("\n[Turn 1] Bob asks for his name:")
    turn1_bob = {"messages": [HumanMessage(content="What is my name?")]}
    response_bob = app.invoke(turn1_bob, config=config_bob)
    print(f"Assistant: {response_bob['messages'][-1].content}")

    # Turn 2: Bob states his name
    print("\n[Turn 2] Bob introduces himself:")
    turn2_bob = {"messages": [HumanMessage(content="My name is Bob.")]}
    app.invoke(turn2_bob, config=config_bob)

    # Turn 3: Bob asks again
    print("\n[Turn 3] Bob asks again:")
    turn3_bob = {"messages": [HumanMessage(content="What is my name now?")]}
    response_bob_final = app.invoke(turn3_bob, config=config_bob)
    print(f"Assistant: {response_bob_final['messages'][-1].content}")

    # --- 5. Verify Alice's Thread is Still Intact ---
    print("\n" + "=" * 60)
    print(">>> VERIFYING THREAD 1 REMAINS UNTOUCHED")
    print("=" * 60)
    check_alice = app.invoke({"messages": [HumanMessage(content="Who am I again?")]}, config=config_alice)
    print(f"Assistant (for Alice): {check_alice['messages'][-1].content}")


if __name__ == "__main__":
    main()
