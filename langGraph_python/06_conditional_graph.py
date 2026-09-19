"""
=============================================================================
Phase 1: Core Graph Patterns - Lesson 6: Conditional (Dynamic Routing) Graph
=============================================================================

Pattern:
                     START
                       |
               classify_ticket
                       |
               [routing_decision]  (add_conditional_edges)
              /        |         \
         "billing" "technical"  "general"
            /          |           \
     handle_billing  handle_tech  handle_general
            \          |           /
             --------->+<----------
                       |
                  send_reply
                       |
                      END

Characteristics:
- Conditional edges evaluate state dynamically at runtime to pick the next node.
- Implemented via `graph.add_conditional_edges(source_node, route_function, mapping_dict)`.
- The `route_function` inspects the current state and returns a key.
- The `mapping_dict` maps the returned key to the corresponding destination node.
=============================================================================
"""

import sys
import io
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

# Ensure proper utf-8 encoding for Windows terminals
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


# --- 1. Define State Schema ---
class TicketState(TypedDict):
    ticket_text: str
    category: str
    department_response: str
    ticket_status: str


# --- 2. Define Processing & Specialist Nodes ---
def classify_ticket_node(state: TicketState) -> dict:
    """Classifies the ticket category based on keywords."""
    print(f"\n[Classifier] Inspecting ticket: '{state['ticket_text']}'")
    text = state["ticket_text"].lower()

    if any(w in text for w in ["invoice", "charge", "refund", "credit card", "payment"]):
        category = "billing"
    elif any(w in text for w in ["bug", "crash", "error", "broken", "exception"]):
        category = "technical"
    else:
        category = "general"

    print(f"[Classifier] Decided Category: -> {category.upper()}")
    return {"category": category}


def handle_billing_node(state: TicketState) -> dict:
    print("[Specialist: Billing Dept] Processing invoice and refund policy details...")
    return {
        "department_response": "Billing: We have reviewed your invoice. Any eligible refund will reflect within 3 business days.",
        "ticket_status": "Resolved by Billing"
    }


def handle_technical_node(state: TicketState) -> dict:
    print("[Specialist: Tech Support] Checking server logs and error stack traces...")
    return {
        "department_response": "Tech Support: We replicated the issue and escalated ticket #912 to engineering.",
        "ticket_status": "Escalated to Engineering"
    }


def handle_general_node(state: TicketState) -> dict:
    print("[Specialist: General FAQ] Looking up knowledge base articles...")
    return {
        "department_response": "Support: Please check our documentation at docs.example.com for standard guidelines.",
        "ticket_status": "Answered via FAQ"
    }


def send_reply_node(state: TicketState) -> dict:
    """Final common node that sends formatted answer back to customer."""
    print(f"[Notifier] Dispatching response to user. Status: {state['ticket_status']}")
    return {}


# --- 3. Define Conditional Routing Function ---
def route_ticket(state: TicketState) -> str:
    """
    Examines state['category'] and returns the string route key.
    This function is passed directly into add_conditional_edges.
    """
    return state["category"]


def main():
    print("=" * 65)
    print("  LangGraph Pattern: Conditional Routing Flow")
    print("=" * 65)

    # --- 4. Assemble the Conditional Graph ---
    graph = StateGraph(TicketState)

    # Register nodes
    graph.add_node("classify_ticket", classify_ticket_node)
    graph.add_node("handle_billing", handle_billing_node)
    graph.add_node("handle_technical", handle_technical_node)
    graph.add_node("handle_general", handle_general_node)
    graph.add_node("send_reply", send_reply_node)

    # Start at classifier
    graph.add_edge(START, "classify_ticket")

    # Connect conditional edge:
    # Source: "classify_ticket"
    # Router function: route_ticket
    # Path map: { return_value : target_node_name }
    graph.add_conditional_edges(
        "classify_ticket",
        route_ticket,
        {
            "billing": "handle_billing",
            "technical": "handle_technical",
            "general": "handle_general",
        }
    )

    # All specialized handlers converge into send_reply -> END
    graph.add_edge("handle_billing", "send_reply")
    graph.add_edge("handle_technical", "send_reply")
    graph.add_edge("handle_general", "send_reply")
    graph.add_edge("send_reply", END)

    # Compile
    app = graph.compile()

    # --- 5. Test Multiple Tickets Through the Router ---
    test_tickets = [
        "I was charged twice on my credit card for invoice #4421.",
        "Application throws 500 error when clicking upload button.",
        "What are your working hours and office location?"
    ]

    for index, ticket in enumerate(test_tickets, start=1):
        print("\n" + "-" * 55)
        print(f"TEST CASE {index}:")
        initial_input: TicketState = {
            "ticket_text": ticket,
            "category": "",
            "department_response": "",
            "ticket_status": ""
        }
        result = app.invoke(initial_input)
        print(f"Final Reply Delivered:\n  '{result['department_response']}'")


if __name__ == "__main__":
    main()
