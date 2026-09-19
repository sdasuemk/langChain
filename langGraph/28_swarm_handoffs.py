"""
=============================================================================
Phase 6: Multi-Agent Systems & Production - Lesson 28: Swarm / Network Direct Handoffs
=============================================================================

Pattern:
                     START
                       │
                       ▼
                 triage_agent ────────┐ (Direct Handoff)
                       │              │
                (if tech query)       │ (if billing query)
                       │              ▼
                       │        billing_agent ────┐ (Resolved or escalate)
                       ▼                          │
                 tech_support ◄───────────────────┘ (Handoff to tech)
                       │
                       ▼
                      END

Key Concepts:
1. Decentralized Peer-to-Peer Handoffs:
   Unlike the centralized Supervisor pattern (Hub & Spoke), Swarm/Network
   agents communicate as peers, handing off the conversation context directly
   to the next agent without a central supervisor intermediary.
2. Dynamic Handoff Routing:
   Nodes dynamically designate the `current_agent`, and routing edges navigate
   directly to the target peer.
=============================================================================
"""

import sys
import io
from typing import List
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

# Ensure proper utf-8 encoding for Windows terminals
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


# --- 1. Define Swarm State ---
class SwarmState(TypedDict):
    customer_query: str
    current_agent: str
    conversation_trail: List[str]
    resolution: str


# --- 2. Define Peer Agents ---
def triage_agent_node(state: SwarmState) -> dict:
    q = state["customer_query"].lower()
    print(f"\n[Agent: Triage] Evaluating incoming query: '{state['customer_query']}'")

    if "refund" in q or "invoice" in q or "charge" in q:
        next_agent = "billing"
        note = "Triage: Identified billing query. Handing off to Billing Specialist."
    else:
        next_agent = "tech"
        note = "Triage: Identified technical query. Handing off to Tech Support."

    print(f"  --> Handoff -> [{next_agent.upper()} AGENT]")
    return {
        "current_agent": next_agent,
        "conversation_trail": state["conversation_trail"] + [note]
    }


def billing_agent_node(state: SwarmState) -> dict:
    print(f"\n[Agent: Billing Specialist] Reviewing account billing details...")
    q = state["customer_query"].lower()

    # If billing specialist determines a software bug caused the double charge, hand off to tech!
    if "system glitch" in q or "api error" in q:
        note = "Billing: Refund processed, but bug caused the issue. Handing off to Tech Support for RCA."
        print(f"  --> Secondary Handoff -> [TECH SUPPORT AGENT]")
        return {
            "current_agent": "tech",
            "conversation_trail": state["conversation_trail"] + [note]
        }

    note = "Billing: Successfully issued $50 refund to customer credit card."
    print("  --> Issue resolved within Billing department.")
    return {
        "current_agent": "FINISH",
        "conversation_trail": state["conversation_trail"] + [note],
        "resolution": "Billing resolved ticket."
    }


def tech_support_agent_node(state: SwarmState) -> dict:
    print(f"\n[Agent: Tech Support Specialist] Diagnosing technical problem...")
    note = "Tech Support: Investigated logs, resolved system configuration, and verified connectivity."
    print("  --> Technical issue diagnosed and resolved.")
    return {
        "current_agent": "FINISH",
        "conversation_trail": state["conversation_trail"] + [note],
        "resolution": "Tech Support resolved ticket."
    }


# --- 3. Dynamic Peer Router ---
def route_swarm(state: SwarmState) -> str:
    return state["current_agent"]


def main():
    print("=" * 65)
    print("  LangGraph Phase 6: Lesson 28 - Swarm / Peer Direct Handoffs")
    print("=" * 65)

    # --- 4. Assemble Swarm Graph ---
    graph = StateGraph(SwarmState)

    graph.add_node("triage", triage_agent_node)
    graph.add_node("billing", billing_agent_node)
    graph.add_node("tech", tech_support_agent_node)

    graph.add_edge(START, "triage")

    # Triage dynamic handoff
    graph.add_conditional_edges(
        "triage",
        route_swarm,
        {
            "billing": "billing",
            "tech": "tech"
        }
    )

    # Billing dynamic handoff (can resolve or escalate to Tech)
    graph.add_conditional_edges(
        "billing",
        route_swarm,
        {
            "tech": "tech",
            "FINISH": END
        }
    )

    # Tech support resolves to END
    graph.add_edge("tech", END)

    app = graph.compile()

    # --- 5. Test Multi-Hop Peer Handoffs ---
    scenarios = [
        "Please help, I was charged twice due to an api error system glitch during checkout.",
        "My payment card expired and I need to update my monthly invoice.",
        "The server keeps returning timeout errors on the dashboard."
    ]

    for idx, prompt in enumerate(scenarios, start=1):
        print("\n" + "=" * 65)
        print(f"CASE {idx}: '{prompt}'")
        print("=" * 65)

        initial_state: SwarmState = {
            "customer_query": prompt,
            "current_agent": "triage",
            "conversation_trail": [],
            "resolution": ""
        }

        final_res = app.invoke(initial_state)

        print("\nAudit Trail of Handoffs:")
        for step in final_res["conversation_trail"]:
            print(f"  * {step}")


if __name__ == "__main__":
    main()
