"""
=============================================================================
Phase 1: Core Fundamentals - Lesson 3: LLM Integration with MessagesState
=============================================================================

Key Concepts:
1. `MessagesState`:
   LangGraph's pre-built state containing `messages: Annotated[list[AnyMessage], add_messages]`.
   You can also extend it with extra keys!
2. Agent Node:
   A node that takes `state["messages"]`, calls `llm.invoke(...)`, and returns
   the new `AIMessage`.
3. Graph Streaming:
   `app.stream(..., stream_mode="updates")` allows you to observe what each
   node outputs step-by-step.
=============================================================================
"""

import os
import sys
import io
from dotenv import load_dotenv
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, MessagesState, START, END

# Ensure proper utf-8 encoding for Windows terminals
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Load environment variables
load_dotenv()


# --- Optional: Extending MessagesState ---
# If you want more than just chat messages (e.g. tracking topic or user info),
# you can subclass MessagesState:
class CustomChatState(MessagesState):
    topic: str
    turns_count: int


# --- Helper: Get LLM ---
def get_chat_model():
    """
    Initializes the Chat LLM using HUGGINGFACEHUB_API_TOKEN from .env.
    Falls back to a MockChatModel if no API key is available so this
    script can run anytime.
    """
    token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    if token and not token.startswith("hf_..."):
        try:
            from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
            print("Connecting to Hugging Face Inference Endpoint...")
            endpoint = HuggingFaceEndpoint(
                repo_id="deepseek-ai/DeepSeek-V4-Pro",
                task="text-generation",
                max_new_tokens=256,
                temperature=0.3,
                huggingfacehub_api_token=token
            )
            return ChatHuggingFace(llm=endpoint)
        except Exception as e:
            print(f"Warning: Could not initialize HuggingFace model ({e}). Using mock assistant.")

    # Fallback / Demo Assistant
    class MockChatModel:
        def invoke(self, messages):
            user_msg = messages[-1].content
            return AIMessage(
                content=f"[Local Assistant]: I received your question about '{user_msg}'. "
                        f"LangGraph manages state flow seamlessly!"
            )
    return MockChatModel()


def main():
    print("=" * 65)
    print("  LangGraph Phase 1: Lesson 3 - Chat LLM in a StateGraph")
    print("=" * 65)

    model = get_chat_model()

    # --- 1. Define the Agent Node ---
    def call_model(state: MessagesState) -> dict:
        print("\n--> [Node: call_model] Calling LLM with message history...")
        response = model.invoke(state["messages"])
        # Returning {"messages": [response]} appends to the conversation
        return {"messages": [response]}

    # --- 2. Build the Graph ---
    # We use LangGraph's prebuilt MessagesState directly
    workflow = StateGraph(MessagesState)

    workflow.add_node("agent", call_model)

    workflow.add_edge(START, "agent")
    workflow.add_edge("agent", END)

    # Compile the graph
    app = workflow.compile()

    # --- 3. Stream Graph Execution ---
    print("\nStarting conversation simulation...")

    user_query = "What is the difference between a chain (DAG) and a graph (cycle) in LangGraph?"
    initial_input = {
        "messages": [
            SystemMessage(content="You are a clear, concise AI tutor."),
            HumanMessage(content=user_query)
        ]
    }

    print(f"\nUser Query: '{user_query}'\n")
    print("--- Streaming Node Updates ---")

    # stream_mode="updates" yields dicts showing which node executed and what it returned
    for event in app.stream(initial_input, stream_mode="updates"):
        for node_name, node_output in event.items():
            print(f"\n[Completed Node: {node_name}]")
            for msg in node_output["messages"]:
                print(f"  {msg.__class__.__name__}: {msg.content}")

    print("\n" + "=" * 65)
    print("Lesson 3 Complete! You now understand StateGraph + MessagesState.")
    print("=" * 65)


if __name__ == "__main__":
    main()
