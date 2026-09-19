"""
=============================================================================
Phase 2: Tool Calling & Dynamic Routing - Lesson 8: Custom ReAct Agent
=============================================================================

Pattern:
                     START
                       │
                       ▼
                 ┌───────────┐
         ┌──────►│   agent   │
         │       └─────┬─────┘
         │             │
         │     [tools_condition]
         │        /         \
         │   (has tools)   (no tools)
         │      /             \
         │     ▼               ▼
         └── ToolNode         END

Key Concepts:
1. `ToolNode`: Prebuilt node that automatically executes tool calls found in
   the last message and returns `ToolMessage` instances.
2. `tools_condition`: Built-in conditional router that checks if the LLM's
   last message has `tool_calls`. If yes -> "tools"; if no -> END.
3. Cyclic ReAct Loop: The edge `tools -> agent` sends tool execution results
   back to the LLM so it can reason and formulate the final answer.
=============================================================================
"""

import os
import sys
import io
from dotenv import load_dotenv
from typing_extensions import TypedDict
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition

# Ensure proper utf-8 encoding for Windows terminals
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Load environment variables
load_dotenv()


# --- 1. Define Custom Tools using @tool ---
@tool
def add_numbers(a: float, b: float) -> float:
    """Use this tool to add two numbers together."""
    print(f"\n  [Tool: add_numbers] Executing: {a} + {b}")
    return a + b


@tool
def multiply_numbers(a: float, b: float) -> float:
    """Use this tool to multiply two numbers together."""
    print(f"\n  [Tool: multiply_numbers] Executing: {a} * {b}")
    return a * b


@tool
def get_stock_price(ticker: str) -> str:
    """Use this tool to look up the current stock price for a company ticker."""
    print(f"\n  [Tool: get_stock_price] Looking up stock for: {ticker.upper()}")
    mock_prices = {"AAPL": "$185.50", "GOOG": "$175.20", "MSFT": "$420.10", "NVDA": "$125.80"}
    return mock_prices.get(ticker.upper(), f"Ticker {ticker} not found.")


tools = [add_numbers, multiply_numbers, get_stock_price]


# --- 2. Model Helper ---
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

    # Simulated ReAct LLM for standalone testing
    class SimulatedReActModel:
        def __init__(self, bound_tools):
            self.tools = {t.name: t for t in bound_tools}

        def invoke(self, messages):
            last_msg = messages[-1]

            # If the last message is from a tool, synthesize the final answer
            if last_msg.__class__.__name__ == "ToolMessage":
                return AIMessage(
                    content=f"Based on the tool output ({last_msg.content}), here is your final answer!"
                )

            query = str(last_msg.content).lower()

            # Simulate tool selection
            if "stock" in query or "aapl" in query or "nvda" in query:
                ticker = "NVDA" if "nvda" in query else "AAPL"
                return AIMessage(
                    content="",
                    tool_calls=[{"name": "get_stock_price", "args": {"ticker": ticker}, "id": "call_stock_1"}]
                )
            elif "multiply" in query or "*" in query or "times" in query:
                return AIMessage(
                    content="",
                    tool_calls=[{"name": "multiply_numbers", "args": {"a": 12.0, "b": 15.0}, "id": "call_mult_1"}]
                )
            else:
                return AIMessage(
                    content="I can help answer questions or calculate numbers and stock prices using tools."
                )

    return SimulatedReActModel(tools)


def main():
    print("=" * 65)
    print("  LangGraph Phase 2: Lesson 8 - Custom ReAct Agent with ToolNode")
    print("=" * 65)

    base_model = get_chat_model()

    # Bind tools to the model (if supported by standard ChatModel)
    if hasattr(base_model, "bind_tools"):
        model_with_tools = base_model.bind_tools(tools)
    else:
        model_with_tools = base_model

    # --- 3. Define the Agent Node ---
    def call_agent_node(state: MessagesState) -> dict:
        print("\n--> [Agent Node] Calling LLM with state history...")
        response = model_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    # --- 4. Assemble the Graph ---
    graph = StateGraph(MessagesState)

    # Add agent node
    graph.add_node("agent", call_agent_node)

    # Add prebuilt ToolNode
    tool_node = ToolNode(tools)
    graph.add_node("tools", tool_node)

    # Connect START to agent
    graph.add_edge(START, "agent")

    # Conditional routing from agent:
    # If the LLM returned tool_calls -> route to "tools"
    # Otherwise -> route to END
    graph.add_conditional_edges("agent", tools_condition)

    # Return loop: from tools back to agent
    graph.add_edge("tools", "agent")

    # Compile the graph
    app = graph.compile()

    # --- 5. Test the ReAct Loop ---
    query = "What is the current stock price of NVDA?"
    print(f"\nUser Query: '{query}'")

    initial_input = {
        "messages": [
            SystemMessage(content="You are a helpful assistant with access to stock and calculation tools."),
            HumanMessage(content=query)
        ]
    }

    # Stream the steps so we can inspect every node transition
    print("\n--- Streaming Execution Steps ---")
    for event in app.stream(initial_input, stream_mode="updates"):
        for node_name, node_output in event.items():
            print(f"\n[Completed Step: {node_name}]")
            for msg in node_output["messages"]:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    print(f"  Tool Calls Requested: {msg.tool_calls}")
                else:
                    print(f"  Content: {msg.content}")


if __name__ == "__main__":
    main()
