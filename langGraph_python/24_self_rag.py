"""
=============================================================================
Phase 5: Agentic RAG Patterns - Lesson 24: Self-RAG (Self-Reflection & Hallucination Grading)
=============================================================================

Pattern:
                     START
                       │
                       ▼
                    retrieve
                       │
                       ▼
                 generate_answer ◄─────────┐ (Regenerate if hallucinated)
                       │                   │
                       ▼                   │
               grade_hallucination ────────┘
                       │
                 (is grounded)
                       │
                       ▼
                 grade_usefulness ─────────┐ (Rewrite query & re-retrieve if unhelpful)
                       │                   │
                 (is useful)               │
                       │                   ▼
                      END            rewrite_query ──► retrieve

Key Concepts:
1. Two-Tier Self-Reflection:
   - **Groundedness (Hallucination Check)**: Validates whether the LLM's claims are
     strictly supported by the retrieved context.
   - **Usefulness**: Validates whether the response directly answers the user's question.
2. Cyclic Self-Correction:
   Instead of outputting hallucinations to users, the graph traps flaws in an
   evaluation loop and self-corrects until quality criteria are met.
=============================================================================
"""

import sys
import io
from typing import List
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

# Ensure proper utf-8 encoding for Windows terminals
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


# --- 1. Define State Schema ---
class SelfRAGState(TypedDict):
    question: str
    documents: List[str]
    generation: str
    is_grounded: bool
    is_useful: bool
    attempt_count: int


# --- 2. Define Nodes ---
def retrieve_node(state: SelfRAGState) -> dict:
    print(f"\n[Step 1: retrieve] Fetching documents for: '{state['question']}'")
    docs = [
        "Company Policy: Travel expenses must be submitted within 30 days of trip completion.",
        "Receipts are mandatory for all single expenditures exceeding $25."
    ]
    return {"documents": docs}


def generate_answer_node(state: SelfRAGState) -> dict:
    attempt = state.get("attempt_count", 0) + 1
    print(f"\n[Step 2: generate_answer] Generation Attempt #{attempt}...")

    # Simulate: first attempt hallucinates; second attempt is strictly grounded
    if attempt == 1:
        # Hallucinated answer (claims $100 without receipts and 60 days)
        draft = "You can submit expenses within 60 days, and you don't need receipts under $100."
    else:
        # Grounded answer
        draft = "Expenses must be submitted within 30 days, and receipts are required for items over $25."

    print(f"  Generated Draft: '{draft}'")
    return {"generation": draft, "attempt_count": attempt}


def grade_hallucination_node(state: SelfRAGState) -> dict:
    print(f"\n[Step 3: grade_hallucination] Checking groundedness against context...")
    context_text = " ".join(state["documents"])
    draft = state["generation"]

    # Simulated hallucination grader
    if "60 days" in draft or "under $100" in draft:
        print("  --> Grader: [HALLUCINATION DETECTED] Claims contradict retrieved policy.")
        grounded = False
    else:
        print("  --> Grader: [GROUNDED] All claims faithfully supported by documents.")
        grounded = True

    return {"is_grounded": grounded}


def grade_usefulness_node(state: SelfRAGState) -> dict:
    print(f"\n[Step 4: grade_usefulness] Checking if answer addresses user question...")
    q = state["question"].lower()
    draft = state["generation"].lower()

    # Checks if response addresses submission timeframe or receipts
    is_useful = "days" in draft and "receipts" in draft
    status_label = "USEFUL & COMPLETE" if is_useful else "INCOMPLETE"
    print(f"  --> Grader: [{status_label}]")

    return {"is_useful": is_useful}


# --- 3. Conditional Decision Functions ---
def check_hallucination(state: SelfRAGState) -> str:
    if not state["is_grounded"] and state["attempt_count"] < 3:
        print("\n--> [Router]: Draft failed hallucination check -> Looping back to GENERATE.")
        return "regenerate"
    print("\n--> [Router]: Draft passed groundedness check -> Proceeding to USEFULNESS evaluation.")
    return "grade_usefulness"


def check_usefulness(state: SelfRAGState) -> str:
    if not state["is_useful"] and state["attempt_count"] < 3:
        print("\n--> [Router]: Answer failed usefulness check -> Looping back to RETRIEVE.")
        return "retrieve"
    print("\n--> [Router]: Answer PASSED all reflection filters! Directing to END.")
    return "finish"


def main():
    print("=" * 65)
    print("  LangGraph Phase 5: Lesson 24 - Self-RAG Reflection Loop")
    print("=" * 65)

    # --- 4. Assemble Graph ---
    graph = StateGraph(SelfRAGState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate_answer", generate_answer_node)
    graph.add_node("grade_hallucination", grade_hallucination_node)
    graph.add_node("grade_usefulness", grade_usefulness_node)

    # Sequence
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate_answer")
    graph.add_edge("generate_answer", "grade_hallucination")

    # Hallucination evaluation edge
    graph.add_conditional_edges(
        "grade_hallucination",
        check_hallucination,
        {
            "regenerate": "generate_answer",
            "grade_usefulness": "grade_usefulness"
        }
    )

    # Usefulness evaluation edge
    graph.add_conditional_edges(
        "grade_usefulness",
        check_usefulness,
        {
            "retrieve": "retrieve",
            "finish": END
        }
    )

    app = graph.compile()

    # --- 5. Run Execution ---
    initial_input: SelfRAGState = {
        "question": "What are the rules and deadlines for submitting business expenses?",
        "documents": [],
        "generation": "",
        "is_grounded": False,
        "is_useful": False,
        "attempt_count": 0
    }

    final_state = app.invoke(initial_input)

    print("\n" + "=" * 65)
    print("Self-RAG Workflow Complete!")
    print(f"Total Generation Attempts : {final_state['attempt_count']}")
    print(f"Final Groundedness Status : {final_state['is_grounded']}")
    print(f"Final Usefulness Status   : {final_state['is_useful']}")
    print(f"\nFinal Verified Output:\n'{final_state['generation']}'")
    print("=" * 65)


if __name__ == "__main__":
    main()
