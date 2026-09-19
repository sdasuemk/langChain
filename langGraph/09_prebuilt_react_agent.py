"""
=============================================================================
Phase 2: Tool Calling & Dynamic Routing - Lesson 9: Prebuilt create_react_agent
=============================================================================

Key Concepts:
1. `create_react_agent`:
   The production shortcut from `langgraph.prebuilt`. It automatically:
   - Binds the tools to the chat model.
   - Sets up the agent node, ToolNode, tools_condition, and return loop.
   - Compiles and returns a ready-to-run graph application.
2. System Prompts & State Modifiers:
   Passing custom instructions to guide how tools should be used.
=============================================================================
"""

import os
import sys
import io
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.prebuilt import create_react_agent

# Ensure proper utf-8 encoding for Windows terminals
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Load environment variables
load_dotenv()


# --- 1. Define Tools ---
@tool
def calculate_area(length: float, width: float) -> float:
    """Calculates the area of a rectangle given length and width."""
    print(f"\n  [Tool: calculate_area] Computing area: {length} x {width}")
    return length * width


@tool
def search_product_inventory(item_name: str) -> str:
    """Checks the warehouse inventory stock count for a given item."""
    print(f"\n  [Tool: search_product_inventory] Checking warehouse for: '{item_name}'")
    stock = {"laptop": 15, "monitor": 42, "keyboard": 0, "mouse": 80}
    count = stock.get(item_name.lower(), "Item not recognized")
    return f"Stock for {item_name}: {count} units available."


tools = [calculate_area, search_product_inventory]


# --- 2. Model Initializer ---
def get_chat_model():
    token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    if token and not token.startswith("hf_..."):
        try:
            from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
            endpoint = HuggingFaceEndpoint(
                repo_id="deepseek-ai/DeepSeek-V4-Pro",
                task="text-generation",
                max_new_tokens=512,
                temperature=0.1,
                huggingfacehub_api_token=token
            )
            return ChatHuggingFace(llm=endpoint)
        except Exception as e:
            print(f"Warning: HuggingFace model failed ({e}). Using simulated agent.")

    class SimulatedAssistant:
        def bind_tools(self, tools_list):
            return self

        def invoke(self, messages):
            last_msg = messages[-1]
            if last_msg.__class__.__name__ == "ToolMessage":
                return AIMessage(content=f"Result confirmed: {last_msg.content}")

            query = str(last_msg.content).lower()
            if "laptop" in query or "stock" in query:
                return AIMessage(
                    content="",
                    tool_calls=[{"name": "search_product_inventory", "args": {"item_name": "laptop"}, "id": "call_inv_1"}]
                )
            return AIMessage(content="I am ready to help with inventory checks and area calculations.")

    return SimulatedAssistant()


def main():
    print("=" * 65)
    print("  LangGraph Phase 2: Lesson 9 - Prebuilt create_react_agent")
    print("=" * 65)

    model = get_chat_model()

    # --- 3. Create Agent with One Line ---
    # create_react_agent encapsulates the entire graph setup internally
    app = create_react_agent(
        model=model,
        tools=tools,
        prompt="You are an expert warehouse logistics assistant. Always query inventory tools when asked about stock."
    )

    # --- 4. Invoke the Prebuilt Agent ---
    query = "How many laptops do we currently have in stock?"
    print(f"\nUser Query: '{query}'")

    response = app.invoke({"messages": [HumanMessage(content=query)]})

    print("\n--- Final Conversation History ---")
    for msg in response["messages"]:
        print(f"[{msg.__class__.__name__}]: {msg.content}")


if __name__ == "__main__":
    main()
