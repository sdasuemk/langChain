"""
=============================================================================
Phase 4: Human-in-the-Loop (HITL) - Lesson 18: Static Breakpoints
=============================================================================

Pattern:
                     START
                       │
                       ▼
                 prepare_transfer
                       │
                 [BREAKPOINT]  (interrupt_before=["execute_transfer"])
                       │
                       ▼
                execute_transfer
                       │
                       ▼
                      END

Key Concepts:
1. `interrupt_before`:
   Compiling a graph with `graph.compile(checkpointer=..., interrupt_before=[node_name])`
   instructs LangGraph to pause execution right before entering the specified node.
2. Inspecting Paused State:
   `app.get_state(config).next` will indicate that the paused node is scheduled next.
3. Resuming Execution:
   Calling `app.invoke(None, config=config)` seamlessly resumes execution from the
   exact point where it was paused.
=============================================================================
"""

import sys
import io
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver

# Ensure proper utf-8 encoding for Windows terminals
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


# --- 1. Define State Schema ---
class TransferState(TypedDict):
    sender: str
    recipient: str
    amount: float
    status: str
    confirmation_code: str


# --- 2. Define Nodes ---
def prepare_transfer_node(state: TransferState) -> dict:
    print(f"\n[Step 1: prepare_transfer] Validating transfer details...")
    print(f"  Transfer: ${state['amount']} from {state['sender']} -> {state['recipient']}")
    return {"status": "Pending Approval"}


def execute_transfer_node(state: TransferState) -> dict:
    print(f"\n[Step 2: execute_transfer] EXECUTING CRITICAL FINANCIAL OPERATION...")
    print(f"  Sending ${state['amount']} to {state['recipient']}...")
    return {
        "status": "Transferred Successfully",
        "confirmation_code": "TX-998822"
    }


def main():
    print("=" * 65)
    print("  LangGraph Phase 4: Lesson 18 - Static Breakpoints (HITL)")
    print("=" * 65)

    # --- 3. Assemble Graph ---
    graph = StateGraph(TransferState)
    graph.add_node("prepare_transfer", prepare_transfer_node)
    graph.add_node("execute_transfer", execute_transfer_node)

    graph.add_edge(START, "prepare_transfer")
    graph.add_edge("prepare_transfer", "execute_transfer")
    graph.add_edge("execute_transfer", END)

    # A checkpointer is MANDATORY for Human-in-the-Loop workflows
    checkpointer = MemorySaver()

    # Compile with static breakpoint BEFORE execute_transfer
    app = graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["execute_transfer"]
    )

    thread_config = {"configurable": {"thread_id": "transfer_session_001"}}

    # --- 4. Launch Workflow (Runs until Breakpoint) ---
    print("\n--- STAGE 1: Initiating transfer ---")
    initial_input: TransferState = {
        "sender": "Alice",
        "recipient": "Bob",
        "amount": 5000.0,
        "status": "Initiated",
        "confirmation_code": ""
    }

    # The graph will run prepare_transfer and halt BEFORE execute_transfer
    result_phase1 = app.invoke(initial_input, config=thread_config)

    # --- 5. Inspect the Paused State ---
    print("\n--- STAGE 2: Graph Paused at Breakpoint ---")
    snapshot = app.get_state(thread_config)
    print(f"Current State Values : {snapshot.values}")
    print(f"Next Node Scheduled  : {snapshot.next}")

    if "execute_transfer" in snapshot.next:
        print("\n>>> HUMAN REVIEW REQUIRED <<<")
        print(f"Human Operator: 'Reviewed transfer of ${snapshot.values['amount']} to {snapshot.values['recipient']}.'")
        print("Human Operator: 'Transfer APPROVED.'")

    # --- 6. Resume Graph Execution ---
    print("\n--- STAGE 3: Resuming Graph Execution ---")
    # Passing None as input tells LangGraph to continue with the current state
    final_result = app.invoke(None, config=thread_config)

    print("\n--- Workflow Completed ---")
    print(f"Final Status            : {final_result['status']}")
    print(f"Final Confirmation Code : {final_result['confirmation_code']}")


if __name__ == "__main__":
    main()
