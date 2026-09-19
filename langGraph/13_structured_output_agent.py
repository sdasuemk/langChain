"""
=============================================================================
Phase 2: Tool Calling & Dynamic Routing - Lesson 13: Structured Output with Tools
=============================================================================

Pattern:
                     START
                       │
                       ▼
                 ┌───────────┐
         ┌──────►│   agent   │ (Uses research tools, then final structured tool)
         │       └─────┬─────┘
         │             │
         │      [custom_router]
         │        /    │    \
         │  (research) │   (final_schema_tool)
         │      /      │      \
         │     ▼       │       ▼
         └── ToolNode  │   extract_structured_data
                       ▼       │
                      END      ▼
                              END

Key Concepts:
1. Enforcing Typed/Structured Outputs:
   Instead of unreliable regex parsing on free-text strings, agents call a
   designated "Final Answer" tool bound to a Pydantic schema.
2. Custom Router for Schema Tools:
   The conditional edge distinguishes between intermediate helper tools
   (e.g., search, math) and the final structured delivery tool.
=============================================================================
"""

import sys
import io
from typing import Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode

# Ensure proper utf-8 encoding for Windows terminals
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


# --- 1. Define Pydantic Schema for Structured Result ---
class IncidentReport(BaseModel):
    incident_id: str = Field(description="Unique ID for incident")
    severity: str = Field(description="Severity: 'LOW', 'MEDIUM', or 'CRITICAL'")
    root_cause: str = Field(description="Identified root cause")
    action_items: list[str] = Field(description="Recommended recovery steps")


# --- 2. Define Intermediate Helper Tools and Final Submission Tool ---
@tool
def lookup_server_telemetry(server_name: str) -> str:
    """Helper tool: Reads CPU and memory telemetry from servers."""
    print(f"\n  [Helper Tool: lookup_server_telemetry] Inspecting {server_name}...")
    return f"{server_name} CPU: 99.8%, Memory: 98.4%, Out-of-memory error code 137 in syslog."


@tool
def submit_incident_report(
    incident_id: str,
    severity: str,
    root_cause: str,
    action_items: list[str]
) -> str:
    """Final Tool: Submits the formalized incident report."""
    print(f"\n  [Final Tool: submit_incident_report] Formally registering report {incident_id}...")
    return "Report submitted."


research_tools = [lookup_server_telemetry]
all_tools = [lookup_server_telemetry, submit_incident_report]


# --- 3. Define Graph State with Structured Field ---
class IncidentState(MessagesState):
    structured_report: Optional[IncidentReport]


# --- 4. Custom Conditional Router ---
def route_agent_output(state: IncidentState) -> str:
    """
    Checks if the last message invoked the final structured tool:
    - If submit_incident_report called -> route to 'extract_report'
    - If intermediate helper tools called -> route to 'research_tools'
    - Otherwise -> route to END
    """
    last_msg = state["messages"][-1]
    if not hasattr(last_msg, "tool_calls") or not last_msg.tool_calls:
        return "finish"

    tool_names = [call["name"] for call in last_msg.tool_calls]

    if "submit_incident_report" in tool_names:
        print("\n--> [Router]: Detected final structured submission tool! Routing to extraction node.")
        return "extract_report"

    print("\n--> [Router]: Detected research helper tools. Routing to ToolNode.")
    return "research_tools"


def extract_report_node(state: IncidentState) -> dict:
    """Extracts tool arguments from submit_incident_report into a validated Pydantic model."""
    last_msg = state["messages"][-1]
    submission_call = next(
        c for c in last_msg.tool_calls if c["name"] == "submit_incident_report"
    )
    validated = IncidentReport(**submission_call["args"])
    print(f"\n[Extraction Node] Pydantic Model Successfully Validated!")
    return {
        "structured_report": validated,
        "messages": [AIMessage(content=f"Incident {validated.incident_id} logged with severity {validated.severity}.")]
    }


def main():
    print("=" * 65)
    print("  LangGraph Phase 2: Lesson 13 - Structured Output via Tool Calling")
    print("=" * 65)

    # --- 5. Assemble Graph ---
    graph = StateGraph(IncidentState)

    # Simulated agent node that first investigates, then submits structured schema
    def agent_node(state: IncidentState) -> dict:
        messages = state["messages"]
        last_msg = messages[-1]

        # Step 2: Telemetry received, now submit formalized report
        if last_msg.__class__.__name__ == "ToolMessage":
            print("[Agent Node] Telemetry received. Calling submit_incident_report tool...")
            call = {
                "name": "submit_incident_report",
                "args": {
                    "incident_id": "INC-8832",
                    "severity": "CRITICAL",
                    "root_cause": "Out-of-memory leak during batch import.",
                    "action_items": ["Restart container srv-01", "Increase memory limit to 8GB", "Patch memory leak"]
                },
                "id": "call_sub_1"
            }
            return {"messages": [AIMessage(content="", tool_calls=[call])]}

        # Step 1: Initial query -> call telemetry tool first
        print("[Agent Node] Investigating issue. Calling lookup_server_telemetry tool...")
        call = {
            "name": "lookup_server_telemetry",
            "args": {"server_name": "srv-prod-01"},
            "id": "call_tel_1"
        }
        return {"messages": [AIMessage(content="", tool_calls=[call])]}

    graph.add_node("agent", agent_node)
    graph.add_node("research_tools", ToolNode(research_tools))
    graph.add_node("extract_report", extract_report_node)

    graph.add_edge(START, "agent")

    graph.add_conditional_edges(
        "agent",
        route_agent_output,
        {
            "research_tools": "research_tools",
            "extract_report": "extract_report",
            "finish": END
        }
    )

    # Research tools loop back to agent
    graph.add_edge("research_tools", "agent")

    # Final report node leads to END
    graph.add_edge("extract_report", END)

    app = graph.compile()

    # --- 6. Run Execution ---
    initial_input: IncidentState = {
        "messages": [HumanMessage(content="Server srv-prod-01 is crashing, please analyze and submit an incident report.")],
        "structured_report": None
    }

    final_state = app.invoke(initial_input)

    print("\n" + "=" * 65)
    print("Execution Complete! Validated Pydantic Object in State:")
    print("=" * 65)
    report: IncidentReport = final_state["structured_report"]
    print(f"Incident ID : {report.incident_id}")
    print(f"Severity    : {report.severity}")
    print(f"Root Cause  : {report.root_cause}")
    print("Action Items:")
    for item in report.action_items:
        print(f"  - {item}")


if __name__ == "__main__":
    main()
