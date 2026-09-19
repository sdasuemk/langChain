"""
=============================================================================
Phase 7: Advanced Production Patterns - Lesson 33: Modern Command Control Flow
=============================================================================

Pattern:
                     START
                       │
                       ▼
                 triage_node
                       │  (Returns Command(update={...}, goto="..."))
              ┌────────┴────────┐
              ▼                 ▼
          refund_node      tech_node
              │                 │
              └────────┬────────┘
                       ▼
                      END

Key Concepts:
1. Unified Node Return (`Command`):
   Traditionally, node returns updated state dicts, and separate `add_conditional_edges`
   functions handle routing.
   In modern LangGraph (v0.2.20+), a node can return `Command(update={...}, goto="destination")`,
   combining state mutation AND dynamic routing in a single line!
2. Eliminating Routing Boilerplate:
   Greatly simplifies dynamic agents and multi-turn loops.
=============================================================================
"""

import sys
import io
from typing import Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

# Ensure proper utf-8 encoding for Windows terminals
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


# --- 1. Define State Schema ---
class CommandState(TypedDict):
    customer_request: str
    route_chosen: str
    reply: str


def main():
    print("=" * 65)
    print("  LangGraph Advanced: Lesson 33 - Modern Command Control Flow")
    print("=" * 65)

    try:
        from langgraph.types import Command
    except ImportError:
        print("Note: Command control flow requires modern LangGraph (v0.2.20+).")
        return

    # --- 2. Define Nodes Returning Command Directly ---
    def triage_node(state: CommandState) -> Command[Literal["refund", "tech"]]:
        req = state["customer_request"].lower()
        print(f"\n[Triage Node] Inspecting: '{state['customer_request']}'")

        if "money" in req or "charge" in req or "refund" in req:
            target = "refund"
            explanation = "Billing issue detected."
        else:
            target = "tech"
            explanation = "Technical glitch detected."

        print(f"  --> Returning Command(goto='{target}') directly from node!")
        return Command(
            update={"route_chosen": target},
            goto=target  # Directly controls next node execution!
        )

    def refund_node(state: CommandState) -> dict:
        print("\n[Refund Node] Processing credit card reimbursement...")
        return {"reply": "Reimbursement processed to your card."}

    def tech_node(state: CommandState) -> dict:
        print("\n[Tech Node] Checking network connection and server latency...")
        return {"reply": "Server diagnostic completed. Latency normalized."}

    # --- 3. Assemble Graph ---
    graph = StateGraph(CommandState)

    graph.add_node("triage", triage_node)
    graph.add_node("refund", refund_node)
    graph.add_node("tech", tech_node)

    # Note: No need for add_conditional_edges! triage_node handles it via Command.
    graph.add_edge(START, "triage")
    graph.add_edge("refund", END)
    graph.add_edge("tech", END)

    app = graph.compile()

    # --- 4. Test Scenarios ---
    test_cases = [
        "Please refund my accidental second charge.",
        "The web portal is timing out on checkout."
    ]

    for i, test in enumerate(test_cases, start=1):
        print("\n" + "=" * 55)
        print(f"TEST {i}: '{test}'")
        print("=" * 55)

        result = app.invoke({
            "customer_request": test,
            "route_chosen": "",
            "reply": ""
        })

        print(f"Final Destination : {result['route_chosen']}")
        print(f"Final Reply       : {result['reply']}")


if __name__ == "__main__":
    main()
