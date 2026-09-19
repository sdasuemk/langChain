"""
=============================================================================
Phase 2: Tool Calling & Dynamic Routing - Lesson 11: Parallel Tool Calling & Recursion Limits
=============================================================================

Key Concepts:
1. Multi-Tool Calling in a Single Step:
   Modern LLMs can return multiple tool calls in a single AIMessage
   (e.g., querying two cities' weather or running two math problems at once).
2. Parallel Tool Execution in ToolNode:
   ToolNode inspects all tool calls in AIMessage.tool_calls and executes them,
   returning multiple ToolMessage instances mapped to their respective tool_call_id.
3. Recursion Limit Protection:
   If an agent gets stuck in a tool loop (e.g. repeated tool calls with the same error),
   LangGraph's `config={"recursion_limit": N}` forcefully halts execution with a
   GraphRecursionError to protect your API quota.
=============================================================================
"""

import sys
import io
from typing_extensions import TypedDict
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.errors import GraphRecursionError

# Ensure proper utf-8 encoding for Windows terminals
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


# --- 1. Define Tools ---
@tool
def get_temperature(city: str) -> str:
    """Returns the current temperature for a city."""
    print(f"\n  [Tool: get_temperature] Fetching temperature for '{city}'...")
    temps = {"Tokyo": "18°C", "New York": "24°C", "London": "15°C", "Paris": "21°C"}
    return temps.get(city.title(), "20°C (Default)")


@tool
def get_population(city: str) -> str:
    """Returns the population count for a city."""
    print(f"\n  [Tool: get_population] Fetching population for '{city}'...")
    pops = {"Tokyo": "14 Million", "New York": "8.3 Million", "London": "9 Million"}
    return pops.get(city.title(), "Unknown")


tools = [get_temperature, get_population]


# --- 2. Simulated Model that Generates MULTIPLE Parallel Tool Calls ---
class MultiCallModel:
    """Simulates an LLM producing multiple tool calls in a single turn."""
    def invoke(self, messages):
        last_msg = messages[-1]

        # If the last message is a tool response, check if all tool calls were answered
        if last_msg.__class__.__name__ == "ToolMessage":
            tool_msgs = [m for m in messages if m.__class__.__name__ == "ToolMessage"]
            return AIMessage(
                content=f"Summary of all retrieved data:\n" +
                        "\n".join([f" - Result: {m.content}" for m in tool_msgs])
            )

        # First turn: User asks for both temperature and population
        return AIMessage(
            content="",
            tool_calls=[
                {"name": "get_temperature", "args": {"city": "Tokyo"}, "id": "call_tokyo_temp"},
                {"name": "get_population", "args": {"city": "Tokyo"}, "id": "call_tokyo_pop"}
            ]
        )


def main():
    print("=" * 65)
    print("  LangGraph Phase 2: Lesson 11 - Parallel Tool Calls & Recursion Limits")
    print("=" * 65)

    model = MultiCallModel()

    # --- 3. Assemble Graph ---
    graph = StateGraph(MessagesState)

    def agent_node(state: MessagesState) -> dict:
        print("\n--> [Agent Node] Evaluating state and invoking model...")
        response = model.invoke(state["messages"])
        return {"messages": [response]}

    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")

    app = graph.compile()

    # --- 4. Test Multi-Tool Execution ---
    print("\n>>> TEST 1: Multiple parallel tool calls in one step")
    user_query = "What is the temperature and population of Tokyo?"
    print(f"User Query: '{user_query}'")

    input_data = {"messages": [HumanMessage(content=user_query)]}

    for event in app.stream(input_data, stream_mode="updates"):
        for node_name, node_output in event.items():
            print(f"\n[Executed Node: {node_name}]")
            for msg in node_output["messages"]:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    print(f"  Parallel Tool Calls Generated ({len(msg.tool_calls)} total):")
                    for tc in msg.tool_calls:
                        print(f"    * {tc['name']}({tc['args']}) [ID: {tc['id']}]")
                elif msg.__class__.__name__ == "ToolMessage":
                    print(f"  ToolMessage Return: '{msg.content}' [Ref ID: {msg.tool_call_id}]")
                else:
                    print(f"  Final Synthesis: {msg.content}")

    # --- 5. Test Recursion Limit Protection ---
    print("\n" + "-" * 65)
    print(">>> TEST 2: Testing Recursion Limit Protection")

    # A broken agent that repeatedly calls tools forever
    class InfiniteLoopModel:
        def invoke(self, messages):
            return AIMessage(
                content="",
                tool_calls=[{"name": "get_temperature", "args": {"city": "Paris"}, "id": "loop_call"}]
            )

    infinite_graph = StateGraph(MessagesState)
    infinite_graph.add_node("agent", lambda s: {"messages": [InfiniteLoopModel().invoke(s["messages"])]})
    infinite_graph.add_node("tools", ToolNode(tools))
    infinite_graph.add_edge(START, "agent")
    infinite_graph.add_conditional_edges("agent", tools_condition)
    infinite_graph.add_edge("tools", "agent")
    infinite_app = infinite_graph.compile()

    # Set recursion limit to 4 steps to trigger safety stop early
    safe_config = {"recursion_limit": 4}
    print("Running with strict recursion_limit = 4...")

    try:
        infinite_app.invoke({"messages": [HumanMessage(content="Infinite loop test")]}, config=safe_config)
    except GraphRecursionError as e:
        print(f"\n[GraphRecursionError Caught Successfully!]")
        print(f"  Safety Triggered: LangGraph stopped execution after hitting recursion limit.")
        print(f"  Details: {e}")


if __name__ == "__main__":
    main()
