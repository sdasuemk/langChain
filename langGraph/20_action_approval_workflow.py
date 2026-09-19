"""
=============================================================================
Phase 4: Human-in-the-Loop (HITL) - Lesson 20: Full Approval Workflow (Approve/Edit/Reject)
=============================================================================

Pattern:
                     START
                       │
                       ▼
                 propose_action
                       │
                 [BREAKPOINT]   (Human inspects proposed action)
                       │
                [approval_router]
               /       │        \
          "approve"  "edit"    "reject"
             │         │          │
             ▼         ▼          ▼
        execute_action execute_   abort_action
                       modified
             │         │          │
             └─────────┼──────────┘
                       │
                       ▼
                      END

Key Concepts:
1. The 3 Pillars of Human Oversight:
   - **Approve**: Run the action exactly as the AI proposed.
   - **Edit**: Modify parameters via `app.update_state()` before executing.
   - **Reject**: Abort the action completely and route to an escalation/cancellation node.
=============================================================================
"""

import sys
import io
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# Ensure proper utf-8 encoding for Windows terminals
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


# --- 1. Define State Schema ---
class ApprovalState(TypedDict):
    database_name: str
    proposed_sql: str
    human_decision: str  # "approve", "edit", "reject"
    execution_result: str


# --- 2. Define Nodes ---
def propose_action_node(state: ApprovalState) -> dict:
    print(f"\n[AI Node: propose_action] Analyzing database maintenance tasks for '{state['database_name']}'...")
    sql = "DELETE FROM user_logs WHERE created_at < '2024-01-01' AND status = 'archived';"
    print(f"  Proposed SQL Command: '{sql}'")
    return {
        "proposed_sql": sql,
        "human_decision": "pending"
    }


def execute_action_node(state: ApprovalState) -> dict:
    print(f"\n[Execution Node] EXECUTING SQL on {state['database_name']}:")
    print(f"  --> {state['proposed_sql']}")
    return {"execution_result": "Success: 1,420 rows purged."}


def abort_action_node(state: ApprovalState) -> dict:
    print(f"\n[Abort Node] Operation aborted by human supervisor.")
    return {"execution_result": "Cancelled: No changes were applied to the database."}


# --- 3. Routing Condition ---
def route_approval(state: ApprovalState) -> str:
    decision = state.get("human_decision", "reject")
    if decision in ["approve", "edit"]:
        return "execute"
    return "abort"


def run_scenario(scenario_name: str, human_choice: str, edited_sql: str = None):
    print("\n" + "=" * 65)
    print(f"SCENARIO: {scenario_name} (Decision: '{human_choice}')")
    print("=" * 65)

    graph = StateGraph(ApprovalState)
    graph.add_node("propose_action", propose_action_node)
    graph.add_node("execute_action", execute_action_node)
    graph.add_node("abort_action", abort_action_node)

    graph.add_edge(START, "propose_action")

    # Halt right before executing or aborting
    graph.add_conditional_edges(
        "propose_action",
        route_approval,
        {
            "execute": "execute_action",
            "abort": "abort_action"
        }
    )
    graph.add_edge("execute_action", END)
    graph.add_edge("abort_action", END)

    checkpointer = MemorySaver()
    # Interrupt before routing or executing
    app = graph.compile(
        checkpointer=checkpointer,
        interrupt_after=["propose_action"]
    )

    config = {"configurable": {"thread_id": f"session_{scenario_name.lower().replace(' ', '_')}"}}

    # Step 1: AI proposes SQL and halts
    app.invoke(
        {"database_name": "prod_users_db", "proposed_sql": "", "human_decision": "pending", "execution_result": ""},
        config=config
    )

    # Step 2: Human reviews and takes action
    snapshot = app.get_state(config)
    print(f"\n[Human Review]: AI proposed: '{snapshot.values['proposed_sql']}'")

    if human_choice == "approve":
        print("[Human Decision]: APPROVED without changes.")
        app.update_state(config, {"human_decision": "approve"})
    elif human_choice == "edit":
        print(f"[Human Decision]: EDITED SQL to safer command: '{edited_sql}'")
        app.update_state(config, {"proposed_sql": edited_sql, "human_decision": "edit"})
    elif human_choice == "reject":
        print("[Human Decision]: REJECTED action.")
        app.update_state(config, {"human_decision": "reject"})

    # Step 3: Resume execution
    final_state = app.invoke(None, config=config)
    print(f"Result: {final_state['execution_result']}")


def main():
    print("=" * 65)
    print("  LangGraph Phase 4: Lesson 20 - Full Action Approval Workflow")
    print("=" * 65)

    # Test Case 1: Approve
    run_scenario("Case 1 - Direct Approval", human_choice="approve")

    # Test Case 2: Edit before execution
    run_scenario(
        "Case 2 - Human Edit",
        human_choice="edit",
        edited_sql="DELETE FROM user_logs WHERE created_at < '2023-01-01' LIMIT 100;"
    )

    # Test Case 3: Reject
    run_scenario("Case 3 - Rejection", human_choice="reject")


if __name__ == "__main__":
    main()
