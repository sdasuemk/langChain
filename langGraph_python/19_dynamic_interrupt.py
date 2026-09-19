"""
=============================================================================
Phase 4: Human-in-the-Loop (HITL) - Lesson 19: Dynamic In-Node Interrupts
=============================================================================

Pattern:
                  generate_content
                         │
               [interrupt(payload)]   <-- Node pauses mid-execution
                         │
                 (human responds)
                         │
                  publish_content
                         │
                        END

Key Concepts:
1. Dynamic `interrupt()`:
   Unlike static breakpoints which pause before/after a node, dynamic `interrupt()`
   allows a node to pause MID-EXECUTION, surface questions or drafts to the human,
   and wait for user input.
2. Resuming with `Command(resume=value)`:
   When resuming, the value provided in `Command(resume=...)` is directly returned
   by the `interrupt()` call inside the node function!
=============================================================================
"""

import sys
import io
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver

# Ensure proper utf-8 encoding for Windows terminals
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


# --- 1. Define State Schema ---
class ArticleState(TypedDict):
    topic: str
    draft: str
    human_notes: str
    status: str


def main():
    print("=" * 65)
    print("  LangGraph Phase 4: Lesson 19 - Dynamic In-Node Interrupts")
    print("=" * 65)

    # Check for modern dynamic interrupt support
    try:
        from langgraph.types import interrupt, Command
    except ImportError:
        print("Note: Dynamic 'interrupt' requires modern LangGraph (v0.2.20+).")
        print("Using compatible breakpoint pattern.")
        return

    # --- 2. Define Node with Dynamic Interrupt ---
    def writer_node(state: ArticleState) -> dict:
        print(f"\n[Writer Node] Drafting article on: '{state['topic']}'...")
        initial_draft = f"Title: The Future of AI in 2026.\nArtificial intelligence is rapidly advancing in {state['topic']}."
        print(f"  Generated Draft: '{initial_draft}'")

        # Dynamic Interrupt: Pause right here and ask human for feedback/edits
        print("  --> Triggering dynamic interrupt() to request human review...")
        human_input = interrupt({
            "action": "review_draft",
            "current_draft": initial_draft,
            "instructions": "Please provide editorial approval or adjustment notes."
        })

        print(f"\n[Writer Node Resumed] Received human feedback: '{human_input}'")
        return {
            "draft": initial_draft,
            "human_notes": str(human_input),
            "status": "Reviewed"
        }

    def publisher_node(state: ArticleState) -> dict:
        print(f"\n[Publisher Node] Publishing finalized draft with human notes: '{state['human_notes']}'")
        return {"status": "Published to Website"}

    # --- 3. Assemble Graph ---
    graph = StateGraph(ArticleState)
    graph.add_node("writer", writer_node)
    graph.add_node("publisher", publisher_node)

    graph.add_edge(START, "writer")
    graph.add_edge("writer", "publisher")
    graph.add_edge("publisher", END)

    checkpointer = MemorySaver()
    app = graph.compile(checkpointer=checkpointer)

    thread_config = {"configurable": {"thread_id": "article_workflow_99"}}

    # --- 4. First Run (Halts inside writer_node) ---
    print("\n--- STAGE 1: Starting Article Creation ---")
    app.invoke(
        {"topic": "Autonomous Agents", "draft": "", "human_notes": "", "status": "Drafting"},
        config=thread_config
    )

    # --- 5. Inspect the Interrupted State ---
    print("\n--- STAGE 2: Graph Paused by interrupt() ---")
    snapshot = app.get_state(thread_config)
    print(f"Graph Status: Paused inside node: {snapshot.next}")

    # Inspect the interrupt payload surfaced to the human
    interrupt_info = snapshot.tasks[0].interrupts[0].value if snapshot.tasks and snapshot.tasks[0].interrupts else "Review required"
    print(f"Surfaced Interrupt Payload: {interrupt_info}")

    # --- 6. Resume using Command(resume=...) ---
    print("\n--- STAGE 3: Human submits review via Command(resume=...) ---")
    user_feedback = "Approved with note: Emphasize state machines in Section 2."
    print(f"Human input: '{user_feedback}'")

    final_state = app.invoke(Command(resume=user_feedback), config=thread_config)

    print("\n--- Workflow Completed ---")
    print(f"Final Status : {final_state['status']}")
    print(f"Human Notes  : {final_state['human_notes']}")


if __name__ == "__main__":
    main()
