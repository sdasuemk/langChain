"""
=============================================================================
Phase 4: Human-in-the-Loop (HITL) - Lesson 21: Tool Call Interception & Verification
=============================================================================

Pattern:
                     START
                       │
                       ▼
                 ┌───────────┐
         ┌──────►│   agent   │
         │       └─────┬─────┘
         │             │
         │      [safety_router]
         │        /    │    \
         │  (safe)     │   (dangerous tool called)
         │   /         │      \
         │  ▼          │     [BREAKPOINT: Human Verification]
         │ ToolNode    │      /         \
         │   ▲         │ (Approved)   (Rejected)
         │   │         │    /             \
         │   └─────────┴───┘            Inject Rejection ToolMessage
         │                                    │
         └────────────────────────────────────┘

Key Concepts:
1. Intercepting High-Stakes Tool Calls:
   Non-destructive tools (read-only search, calculations) run automatically.
   Destructive or financial tools trigger a breakpoint before `ToolNode` runs.
2. In-Flight Tool Argument Modification:
   The supervisor can use `update_state()` to modify tool parameters
   (e.g., reducing transfer amount) before `ToolNode` executes.
=============================================================================
"""

import sys
import io
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# Ensure proper utf-8 encoding for Windows terminals
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


# --- 1. Tools Definition ---
@tool
def lookup_balance(account_id: str) -> str:
    """Safe read-only tool to inspect balances."""
    print(f"\n  [Safe Tool: lookup_balance] Account {account_id} balance: $12,000.00")
    return "$12,000.00"


@tool
def send_wire_transfer(recipient: str, amount: float) -> str:
    """CRITICAL tool: Transfers funds to third party."""
    print(f"\n  [Critical Tool: send_wire_transfer] EXECUTING WIRE of ${amount} to {recipient}!")
    return f"Wire transfer of ${amount} to {recipient} confirmed."


all_tools = [lookup_balance, send_wire_transfer]


# --- 2. Safety Routing Logic ---
def route_with_safety_check(state: MessagesState) -> str:
    last_msg = state["messages"][-1]
    if not hasattr(last_msg, "tool_calls") or not last_msg.tool_calls:
        return "finish"

    tool_names = [call["name"] for call in last_msg.tool_calls]

    # If critical wire transfer is requested, route to sensitive_tools node
    if "send_wire_transfer" in tool_names:
        print(f"\n--> [Safety Router]: CRITICAL TOOL DETECTED: {tool_names}. Routing to sensitive gate.")
        return "sensitive_tools"

    return "safe_tools"


def main():
    print("=" * 65)
    print("  LangGraph Phase 4: Lesson 21 - Tool Call Interception & Verification")
    print("=" * 65)

    # --- 3. Assemble Graph ---
    graph = StateGraph(MessagesState)

    def agent_node(state: MessagesState) -> dict:
        last_msg = state["messages"][-1]
        print(f"\n[Agent Node] Processing message history...")

        if last_msg.__class__.__name__ == "ToolMessage":
            return {"messages": [AIMessage(content=f"Operation report: {last_msg.content}")]}

        # Simulate agent requesting a wire transfer
        call = {
            "name": "send_wire_transfer",
            "args": {"recipient": "Acme Corp", "amount": 7500.0},
            "id": "wire_call_101"
        }
        return {"messages": [AIMessage(content="", tool_calls=[call])]}

    graph.add_node("agent", agent_node)
    graph.add_node("safe_tools", ToolNode([lookup_balance]))
    graph.add_node("sensitive_tools", ToolNode([send_wire_transfer]))

    graph.add_edge(START, "agent")

    graph.add_conditional_edges(
        "agent",
        route_with_safety_check,
        {
            "safe_tools": "safe_tools",
            "sensitive_tools": "sensitive_tools",
            "finish": END
        }
    )

    graph.add_edge("safe_tools", "agent")
    graph.add_edge("sensitive_tools", "agent")

    checkpointer = MemorySaver()

    # Compile with breakpoint BEFORE sensitive_tools executes!
    app = graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["sensitive_tools"]
    )

    config = {"configurable": {"thread_id": "wire_session_505"}}

    # --- 4. Run Stage 1 (Agent generates call and halts) ---
    print("\n--- STAGE 1: Agent requests sensitive tool call ---")
    app.invoke(
        {"messages": [HumanMessage(content="Wire $7500 to Acme Corp for invoices.")]},
        config=config
    )

    # --- 5. Inspect Intercepted Tool Call ---
    print("\n--- STAGE 2: Intercepting Tool Call at Breakpoint ---")
    snapshot = app.get_state(config)
    print(f"Scheduled Next Node: {snapshot.next}")

    last_ai_message = snapshot.values["messages"][-1]
    intercepted_call = last_ai_message.tool_calls[0]
    print(f"Intercepted Tool Call: {intercepted_call['name']}")
    print(f"Proposed Arguments   : {intercepted_call['args']}")

    # --- 6. Supervisor Approves and Modifies Parameter ---
    print("\n>>> SUPERVISOR INTERVENTION <<<")
    print("Supervisor: 'Reducing amount from $7,500 to $5,000 for compliance.'")

    # Update tool call arguments in-flight!
    modified_message = AIMessage(
        content="",
        tool_calls=[{
            "name": "send_wire_transfer",
            "args": {"recipient": "Acme Corp", "amount": 5000.0},
            "id": intercepted_call["id"]
        }]
    )

    app.update_state(config, {"messages": [modified_message]}, as_node="agent")

    # --- 7. Resume Execution ---
    print("\n--- STAGE 3: Resuming Tool Execution ---")
    final_state = app.invoke(None, config=config)

    print("\n--- Final Agent Response Delivered ---")
    print(final_state["messages"][-1].content)


if __name__ == "__main__":
    main()
