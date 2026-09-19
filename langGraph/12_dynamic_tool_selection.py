"""
=============================================================================
Phase 2: Tool Calling & Dynamic Routing - Lesson 12: Dynamic Tool Selection
=============================================================================

Pattern:
                     START
                       │
                       ▼
                 ┌───────────┐
         ┌──────►│   agent   │ (Dynamically selects tools based on user_role)
         │       └─────┬─────┘
         │             │
         │     [dynamic_router]
         │        /         \
         │   (has tools)   (no tools)
         │      /             \
         │     ▼               ▼
         └── ToolNode         END

Key Concepts:
1. Context-Aware Tool Provisioning:
   In production, you don't expose all tools to every user. Tools should be
   filtered dynamically based on state (e.g., user permission level, tenant,
   or current sub-task).
2. Extending State with Context:
   Subclassing `MessagesState` to include `user_role: str` or `department: str`.
=============================================================================
"""

import sys
import io
from typing import List
from typing_extensions import TypedDict
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition

# Ensure proper utf-8 encoding for Windows terminals
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


# --- 1. Custom State with Context ---
class RoleBasedAgentState(MessagesState):
    user_role: str  # "guest", "employee", "admin"
    authorized_tools: List[str]


# --- 2. Tool Pool Definition ---
@tool
def public_faq_search(query: str) -> str:
    """Searches general knowledge base. Available to everyone."""
    print(f"\n  [Tool: public_faq_search] Query: '{query}'")
    return "Our office is open Monday to Friday, 9am to 6pm."


@tool
def internal_directory_lookup(name: str) -> str:
    """Finds employee contact info. Available to internal staff and admin only."""
    print(f"\n  [Tool: internal_directory_lookup] Looking up employee: '{name}'")
    return f"Employee {name}: Engineering Dept, Desk 4B, ext 5541."


@tool
def server_reboot(server_id: str) -> str:
    """Restarts a production cloud server. Restricted to ADMIN only."""
    print(f"\n  [Tool: server_reboot] WARNING: Rebooting critical server: '{server_id}'")
    return f"Server {server_id} successfully rebooted."


all_tools_list = [public_faq_search, internal_directory_lookup, server_reboot]
all_tools_map = {t.name: t for t in all_tools_list}


# --- 3. Dynamic Tool Filter Function ---
def get_allowed_tools_for_role(role: str) -> list:
    """Determines which tool instances a specific user role is permitted to call."""
    if role == "admin":
        return all_tools_list
    elif role == "employee":
        return [public_faq_search, internal_directory_lookup]
    else:  # "guest"
        return [public_faq_search]


def main():
    print("=" * 65)
    print("  LangGraph Phase 2: Lesson 12 - Dynamic Tool Selection & RBAC")
    print("=" * 65)

    # --- 4. Assemble Graph ---
    graph = StateGraph(RoleBasedAgentState)

    def agent_node(state: RoleBasedAgentState) -> dict:
        role = state.get("user_role", "guest")
        allowed_tools = get_allowed_tools_for_role(role)
        allowed_names = [t.name for t in allowed_tools]

        last_msg = state["messages"][-1]
        print(f"\n[Agent Node] User Role: '{role.upper()}' | Permitted Tools: {allowed_names}")

        # If last message was a tool return, produce the final answer
        if last_msg.__class__.__name__ == "ToolMessage":
            return {"messages": [AIMessage(content=f"Authorized result: {last_msg.content}")]}

        query = str(last_msg.content).lower()

        # Simulate dynamic tool dispatch respecting permissions
        if "reboot" in query:
            if "server_reboot" in allowed_names:
                call = {"name": "server_reboot", "args": {"server_id": "srv-prod-01"}, "id": "tc_reb"}
                return {"messages": [AIMessage(content="", tool_calls=[call])], "authorized_tools": allowed_names}
            else:
                return {
                    "messages": [AIMessage(content=f"Access Denied: Role '{role}' is not authorized to reboot servers.")],
                    "authorized_tools": allowed_names
                }
        elif "employee" in query or "directory" in query:
            if "internal_directory_lookup" in allowed_names:
                call = {"name": "internal_directory_lookup", "args": {"name": "Alex"}, "id": "tc_dir"}
                return {"messages": [AIMessage(content="", tool_calls=[call])], "authorized_tools": allowed_names}
            else:
                return {
                    "messages": [AIMessage(content=f"Access Denied: Role '{role}' cannot view internal directory.")],
                    "authorized_tools": allowed_names
                }
        else:
            call = {"name": "public_faq_search", "args": {"query": query}, "id": "tc_faq"}
            return {"messages": [AIMessage(content="", tool_calls=[call])], "authorized_tools": allowed_names}

    # Register ToolNode containing all possible tools
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(all_tools_list))

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")

    app = graph.compile()

    # --- 5. Test with Different User Roles ---
    scenarios = [
        {"role": "guest", "query": "Please reboot server srv-prod-01"},
        {"role": "admin", "query": "Please reboot server srv-prod-01"},
        {"role": "guest", "query": "Where is the internal office directory?"},
        {"role": "employee", "query": "Where is the internal office directory?"}
    ]

    for i, scen in enumerate(scenarios, start=1):
        print("\n" + "=" * 60)
        print(f"SCENARIO {i}: Role = '{scen['role']}' | Query: '{scen['query']}'")
        print("=" * 60)

        initial_state: RoleBasedAgentState = {
            "messages": [HumanMessage(content=scen["query"])],
            "user_role": scen["role"],
            "authorized_tools": []
        }

        result = app.invoke(initial_state)
        print(f"\nFinal AI Output: {result['messages'][-1].content}")


if __name__ == "__main__":
    main()
