"""
=============================================================================
Phase 5: Agentic RAG Patterns - Lesson 25: Adaptive RAG (Strategy Routing)
=============================================================================

Pattern:
                     START
                       │
                       ▼
                 analyze_query
                       │
               [strategy_router]
              /        │        \
        "simple"   "standard"  "complex"
           │           │          │
           ▼           ▼          ▼
       direct_llm single_hop  decompose_multihop
           │           │          │
           │           ▼          ▼
           │       retrieve   iterative_retrieval
           │           │          │
           └───────────┼──────────┘
                       │
                       ▼
                 generate_final
                       │
                      END

Key Concepts:
1. Adaptive Strategy Selection:
   Instead of applying one uniform architecture to every question, Adaptive RAG
   routes queries to the most cost-effective and accurate strategy:
   - Direct: Fast & cheap for general logic or simple trivia.
   - Standard: Fast vector search for standard factual lookups.
   - Multi-Hop: Query decomposition into sub-questions for multi-step reasoning.
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
class AdaptiveRAGState(TypedDict):
    query: str
    complexity: str  # "simple", "standard", "complex"
    sub_queries: List[str]
    retrieved_facts: List[str]
    final_response: str


# --- 2. Define Query Analysis Node ---
def analyze_query_node(state: AdaptiveRAGState) -> dict:
    q = state["query"].lower()
    print(f"\n[Strategy Analyzer] Assessing query complexity: '{state['query']}'")

    if any(w in q for w in ["compare", "difference between", "trend over", "and how does it affect"]):
        complexity = "complex"
    elif any(w in q for w in ["what is", "where is", "when did", "who is", "definition"]):
        complexity = "standard"
    else:
        complexity = "simple"

    print(f"  Selected Strategy -> {complexity.upper()}")
    return {"complexity": complexity}


# --- 3. Strategy Execution Nodes ---
def direct_llm_strategy_node(state: AdaptiveRAGState) -> dict:
    print("  [Direct Strategy] Answering directly without document retrieval...")
    return {
        "retrieved_facts": ["Direct Model Knowledge"],
        "final_response": f"Direct Answer to: '{state['query']}'."
    }


def standard_rag_strategy_node(state: AdaptiveRAGState) -> dict:
    print("  [Standard RAG Strategy] Performing single-hop vector retrieval...")
    facts = [f"Retrieved factual context for '{state['query']}' from vector store."]
    return {
        "retrieved_facts": facts,
        "final_response": f"Standard RAG Answer:\nBased on document: {facts[0]}"
    }


def decompose_multihop_node(state: AdaptiveRAGState) -> dict:
    print("  [Complex Strategy: Decomposition] Breaking question into sub-queries...")
    # Simulated query decomposition
    sub_q = [
        f"Sub-query 1: Extract metrics for aspect A of '{state['query'][:25]}...'",
        f"Sub-query 2: Extract metrics for aspect B of '{state['query'][:25]}...'"
    ]
    for s in sub_q:
        print(f"    * {s}")
    return {"sub_queries": sub_q}


def iterative_retrieval_node(state: AdaptiveRAGState) -> dict:
    print("  [Complex Strategy: Iterative Retrieval] Resolving all sub-queries sequentially...")
    facts = []
    for sq in state["sub_queries"]:
        fact = f"Fact discovered for [{sq}]: Metric confirmed."
        facts.append(fact)
        print(f"    --> Found: {fact}")

    synthesis = "Multi-Hop Synthesized Analysis:\n" + "\n".join(facts)
    return {
        "retrieved_facts": facts,
        "final_response": synthesis
    }


# --- 4. Strategy Routing Function ---
def route_strategy(state: AdaptiveRAGState) -> str:
    return state["complexity"]


def main():
    print("=" * 65)
    print("  LangGraph Phase 5: Lesson 25 - Adaptive RAG Architecture")
    print("=" * 65)

    # --- 5. Assemble Graph ---
    graph = StateGraph(AdaptiveRAGState)

    graph.add_node("analyze_query", analyze_query_node)
    graph.add_node("direct_strategy", direct_llm_strategy_node)
    graph.add_node("standard_strategy", standard_rag_strategy_node)
    graph.add_node("decompose", decompose_multihop_node)
    graph.add_node("iterative_retrieval", iterative_retrieval_node)

    graph.add_edge(START, "analyze_query")

    # Conditional branching to strategy
    graph.add_conditional_edges(
        "analyze_query",
        route_strategy,
        {
            "simple": "direct_strategy",
            "standard": "standard_strategy",
            "complex": "decompose"
        }
    )

    graph.add_edge("decompose", "iterative_retrieval")

    # Convergence
    graph.add_edge("direct_strategy", END)
    graph.add_edge("standard_strategy", END)
    graph.add_edge("iterative_retrieval", END)

    app = graph.compile()

    # --- 6. Test Various Complexities ---
    test_prompts = [
        "Hi, how are you today?",
        "What is the definition of retrieval augmented generation?",
        "Compare the performance difference between model A and model B and how does it affect memory consumption?"
    ]

    for i, prompt in enumerate(test_prompts, start=1):
        print("\n" + "=" * 60)
        print(f"TEST {i}: '{prompt}'")
        print("=" * 60)

        res = app.invoke({
            "query": prompt,
            "complexity": "",
            "sub_queries": [],
            "retrieved_facts": [],
            "final_response": ""
        })

        print("\nResult Output:")
        print(res["final_response"])


if __name__ == "__main__":
    main()
