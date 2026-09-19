"""
=============================================================================
Phase 2: Tool Calling & Dynamic Routing - Lesson 10: Advanced Tool Routing
=============================================================================

Pattern:
                     START
                       │
                       ▼
                     agent
                       │
             [custom_router_logic]
            /          │          \
       (read_tools) (write_tools) (END)
            │          │            │
            ▼          ▼            │
        SafeTools  MutationTools    │
            │          │            │
            └──────────┴────────────┘
                       │
                       ▼
                     agent

Key Concepts:
1. Segregating Tools by Permission / Risk:
   Splitting tools into multiple ToolNodes (e.g., read-only vs. write/destructive).
2. Resilient Error Handling:
   Configuring ToolNode with `handle_tool_errors=True` so tool exceptions are
   gracefully caught and fed back to the LLM as error messages for self-recovery.
=============================================================================
"""

import sys
import io
from typing_extensions import TypedDict
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode

# Ensure proper utf-8 encoding for Windows terminals
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


# --- 1. Define Read-Only and Destructive Tools ---
@tool
def fetch_account_balance(user_id: str) -> str:
    """Safe read-only tool to inspect an account balance."""
    print(f"\n  [Safe Tool: fetch_account_balance] Checking balance for: {user_id}")
    return f"Account {user_id} balance: $1,450.00"


@tool
def transfer_funds(user_id: str, amount: float, recipient: str) -> str:
    """High-privilege tool that moves real funds between accounts."""
    print(f"\n  [Mutation Tool: transfer_funds] Transferring ${amount} to {recipient}...")
    if amount > 1000.0:
        # Simulate a business rule error
        raise ValueError("Transfers over $1,000 require secondary manager approval.")
    return f"Successfully sent ${amount} from {user_id} to {recipient}."


read_tools = [fetch_account_balance]
write_tools = [transfer_funds]
all_tools = {t.name: t for t in read_tools + write_tools}


# --- 2. Custom Router to Segregate Tool Calls by Risk Level ---
def route_tools_by_risk(state: MessagesState) -> str:
    """
    Inspects tool calls and routes to either:
    - 'read_tools_node' if only safe tools are called
    - 'write_tools_node' if any mutation tools are called
    - END if no tools are needed
    """
    last_message = state["messages"][-1]
    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return "finish"

    # Check the requested tool names
    tool_names = [call["name"] for call in last_message.tool_calls]
    if any(name == "transfer_funds" for name in tool_names):
        print(f"\n--> [Router]: Detected high-risk mutation tool: {tool_names} -> Routing to WRITE node")
        return "write_tools"
    
    print(f"\n--> [Router]: Detected safe read-only tool: {tool_names} -> Routing to READ node")
    return "read_tools"


def main():
    print("=" * 65)
    print("  LangGraph Phase 2: Lesson 10 - Advanced Tool Routing & Error Recovery")
    print("=" * 65)

    # --- 3. Assemble Segregated Tool Graph ---
    graph = StateGraph(MessagesState)

    # Simulated LLM Agent node
    def agent_node(state: MessagesState) -> dict:
        last_msg = state["messages"][-1]
        print(f"\n[Agent Node] Reviewing last message...")

        # If we just received a ToolMessage with an error, handle it gracefully
        if last_msg.__class__.__name__ == "ToolMessage":
            if "Error" in str(last_msg.content) or "ValueError" in str(last_msg.content):
                return {"messages": [AIMessage(content=f"Notice: The operation could not be completed ({last_msg.content}).")]}
            return {"messages": [AIMessage(content=f"Operation finished successfully: {last_msg.content}")]}

        query = str(last_msg.content).lower()
        if "transfer" in query:
            return {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[{
                            "name": "transfer_funds",
                            "args": {"user_id": "usr_99", "amount": 1500.0, "recipient": "Alice"},
                            "id": "call_tx_1"
                        }]
                    )
                ]
            }
        else:
            return {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[{
                            "name": "fetch_account_balance",
                            "args": {"user_id": "usr_99"},
                            "id": "call_bal_1"
                        }]
                    )
                ]
            }

    graph.add_node("agent", agent_node)

    # ToolNode with handle_tool_errors=True catches Python exceptions
    # and formats them into a ToolMessage error payload instead of crashing!
    safe_tool_node = ToolNode(read_tools, handle_tool_errors=True)
    write_tool_node = ToolNode(write_tools, handle_tool_errors=True)

    graph.add_node("safe_tools", safe_tool_node)
    graph.add_node("mutation_tools", write_tool_node)

    # Connect entry
    graph.add_edge(START, "agent")

    # Conditional routing to the appropriate tool node
    graph.add_conditional_edges(
        "agent",
        route_tools_by_risk,
        {
            "read_tools": "safe_tools",
            "write_tools": "mutation_tools",
            "finish": END
        }
    )

    # Both tool nodes loop back to agent
    graph.add_edge("safe_tools", "agent")
    graph.add_edge("mutation_tools", "agent")

    # Compile
    app = graph.compile()

    # --- 4. Test Error Handling & Routing ---
    test_queries = [
        "What is my current balance?",
        "Please transfer $1500 from my account to Alice."
    ]

    for idx, query in enumerate(test_queries, start=1):
        print("\n" + "=" * 55)
        print(f"CASE {idx}: '{query}'")
        print("=" * 55)

        res = app.invoke({"messages": [HumanMessage(content=query)]})
        print("\n--- Final Message Output ---")
        print(res["messages"][-1].content)


if __name__ == "__main__":
    main()
