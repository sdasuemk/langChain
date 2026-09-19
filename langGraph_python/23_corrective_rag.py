"""
=============================================================================
Phase 5: Agentic RAG Patterns - Lesson 23: Corrective RAG (CRAG)
=============================================================================

Pattern:
                     START
                       │
                       ▼
                   retrieve
                       │
                       ▼
                 grade_documents
                       │
               [relevance_decision]
              /                    \
       (relevant docs)       (poor / no docs)
             │                     │
             │                     ▼
             │               rewrite_query
             │                     │
             │                     ▼
             │             web_search_fallback
             │                     │
             └──────────┬──────────┘
                        │
                        ▼
                 generate_answer
                        │
                       END

Key Concepts:
1. Document Relevance Grading:
   Instead of blindly passing retrieved chunks to the generator, CRAG grades
   the semantic quality and relevance of each retrieved document.
2. Self-Correction Fallback:
   If the internal vector store fails to provide relevant context, the agent
   automatically rewrites the query and falls back to web search before generating.
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
class CRAGState(TypedDict):
    question: str
    original_docs: List[str]
    filtered_docs: List[str]
    needs_web_fallback: bool
    rewritten_query: str
    generation: str


# --- 2. Define Nodes ---
def retrieve_node(state: CRAGState) -> dict:
    q = state["question"].lower()
    print(f"\n[Step 1: retrieve] Searching vector database for: '{state['question']}'...")

    # Simulated vector store lookup
    if "quantum computing" in q:
        # Irrelevant documents simulated (poor vector retrieval)
        raw_docs = [
            "Ancient Greek philosophers debated the nature of atoms.",
            "Photosynthesis is the process by which green plants make food."
        ]
    else:
        # Relevant documents found
        raw_docs = [
            "LangGraph is a framework for building stateful, multi-actor LLM applications.",
            "LangGraph uses state machines to enable loops, branching, and human-in-the-loop workflows."
        ]

    return {"original_docs": raw_docs}


def grade_documents_node(state: CRAGState) -> dict:
    print("\n[Step 2: grade_documents] Evaluating document relevance...")
    q = state["question"].lower()
    relevant_chunks = []

    for idx, doc in enumerate(state["original_docs"], start=1):
        # Simulated relevance grader LLM
        is_relevant = any(w in doc.lower() for w in q.split() if len(w) > 4)

        if is_relevant:
            print(f"  Doc #{idx}: [RELEVANT] -> Retaining chunk.")
            relevant_chunks.append(doc)
        else:
            print(f"  Doc #{idx}: [IRRELEVANT] -> Discarded.")

    needs_fallback = len(relevant_chunks) == 0
    return {
        "filtered_docs": relevant_chunks,
        "needs_web_fallback": needs_fallback
    }


def rewrite_query_node(state: CRAGState) -> dict:
    print(f"\n[Corrective Step: rewrite_query] Retrieval failed. Transforming query for web search...")
    # Simulated query rewriter
    refined = f"{state['question']} latest research developments overview"
    print(f"  Transformed Query: '{refined}'")
    return {"rewritten_query": refined}


def web_search_fallback_node(state: CRAGState) -> dict:
    print(f"\n[Fallback Step: web_search] Querying web with rewritten search: '{state['rewritten_query']}'...")
    web_docs = [
        f"Web Search Summary for '{state['rewritten_query']}': Breakthrough in 1,000-qubit quantum processors announced."
    ]
    return {"filtered_docs": web_docs}


def generate_answer_node(state: CRAGState) -> dict:
    print("\n[Final Step: generate_answer] Synthesizing response from validated context...")
    context = "\n".join(state["filtered_docs"])
    answer = f"Answer to '{state['question']}':\n{context}"
    return {"generation": answer}


# --- 3. Conditional Router ---
def decide_to_generate(state: CRAGState) -> str:
    if state["needs_web_fallback"]:
        print("\n--> [Router Decision]: No relevant internal docs found -> Triggering FALLBACK pipeline.")
        return "fallback"
    print("\n--> [Router Decision]: Relevant internal docs verified -> Proceeding directly to GENERATION.")
    return "generate"


def main():
    print("=" * 65)
    print("  LangGraph Phase 5: Lesson 23 - Corrective RAG (CRAG)")
    print("=" * 65)

    # --- 4. Assemble Graph ---
    graph = StateGraph(CRAGState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade_documents", grade_documents_node)
    graph.add_node("rewrite_query", rewrite_query_node)
    graph.add_node("web_search_fallback", web_search_fallback_node)
    graph.add_node("generate_answer", generate_answer_node)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade_documents")

    # Conditional decision after grading
    graph.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "generate": "generate_answer",
            "fallback": "rewrite_query"
        }
    )

    graph.add_edge("rewrite_query", "web_search_fallback")
    graph.add_edge("web_search_fallback", "generate_answer")
    graph.add_edge("generate_answer", END)

    app = graph.compile()

    # --- 5. Test Both Branches ---
    # Case A: Good internal retrieval (No fallback needed)
    print("\n" + "=" * 60)
    print(">>> CASE A: Query matches internal documentation")
    print("=" * 60)
    result_a = app.invoke({
        "question": "What is LangGraph state machine?",
        "original_docs": [],
        "filtered_docs": [],
        "needs_web_fallback": False,
        "rewritten_query": "",
        "generation": ""
    })
    print("\nFinal Generation:\n", result_a["generation"])

    # Case B: Poor internal retrieval (Triggers CRAG rewrite & fallback)
    print("\n" + "=" * 60)
    print(">>> CASE B: Query has missing internal docs (Triggers CRAG)")
    print("=" * 60)
    result_b = app.invoke({
        "question": "What is quantum computing?",
        "original_docs": [],
        "filtered_docs": [],
        "needs_web_fallback": False,
        "rewritten_query": "",
        "generation": ""
    })
    print("\nFinal Generation:\n", result_b["generation"])


if __name__ == "__main__":
    main()
