"""
=============================================================================
Phase 5: Agentic RAG Patterns - Lesson 22: Router RAG
=============================================================================

Pattern:
                     START
                       │
                       ▼
                 classify_query
                       │
                 [router_logic]
                /      │       \
          (vector)   (web)    (direct)
             │         │         │
             ▼         ▼         ▼
          vector_db  web_search direct_llm
             │         │         │
             └─────────┼─────────┘
                       │
                       ▼
                 generate_answer
                       │
                      END

Key Concepts:
1. Intent-Based Routing:
   Naive RAG always queries the vector store, even for simple greetings or
   real-time news. Router RAG analyzes query intent to pick the optimal source:
   - Proprietary company knowledge -> Vector DB
   - Recent live world events -> Web Search
   - General conversation / math -> Direct LLM
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
class RouterRAGState(TypedDict):
    question: str
    datasource: str  # "vectorstore", "web_search", "direct"
    retrieved_docs: List[str]
    answer: str


# --- 2. Define Query Classifier Node ---
def classify_query_node(state: RouterRAGState) -> dict:
    q = state["question"].lower()
    print(f"\n[Classifier Node] Analyzing intent for: '{state['question']}'")

    if any(w in q for w in ["internal policy", "handbook", "pto", "company 401k", "quarterly revenue"]):
        datasource = "vectorstore"
    elif any(w in q for w in ["today", "current weather", "stock price now", "latest news", "olympics 2026"]):
        datasource = "web_search"
    else:
        datasource = "direct"

    print(f"  Decision -> Route to: {datasource.upper()}")
    return {"datasource": datasource}


# --- 3. Define Specialized Retrieval / Generation Nodes ---
def vector_retriever_node(state: RouterRAGState) -> dict:
    print("  [Vector Store] Searching company knowledge base index...")
    mock_docs = [
        "Policy Doc: Employees receive 20 days PTO annually.",
        "Benefits Doc: 401(k) matches up to 5% of base salary."
    ]
    return {"retrieved_docs": mock_docs}


def web_search_node(state: RouterRAGState) -> dict:
    print("  [Web Search] Querying live search API for real-time information...")
    mock_web_results = [
        "Live Web Result: Today's weather is 22°C with clear skies.",
        "Live Web Result: Market indices gained 1.2% in morning trading."
    ]
    return {"retrieved_docs": mock_web_results}


def direct_llm_node(state: RouterRAGState) -> dict:
    print("  [Direct LLM] No external retrieval required for general knowledge.")
    return {"retrieved_docs": ["No external documents needed."]}


def generate_answer_node(state: RouterRAGState) -> dict:
    print(f"\n[Synthesizer Node] Compiling final answer using datasource: '{state['datasource']}'")
    context = "\n".join(state["retrieved_docs"])
    answer = f"Answer to '{state['question']}':\nBased on {state['datasource']}:\n{context}"
    return {"answer": answer}


# --- 4. Routing Function ---
def route_query(state: RouterRAGState) -> str:
    return state["datasource"]


def main():
    print("=" * 65)
    print("  LangGraph Phase 5: Lesson 22 - Router RAG Architecture")
    print("=" * 65)

    # --- 5. Assemble Graph ---
    graph = StateGraph(RouterRAGState)

    graph.add_node("classify_query", classify_query_node)
    graph.add_node("vectorstore", vector_retriever_node)
    graph.add_node("web_search", web_search_node)
    graph.add_node("direct", direct_llm_node)
    graph.add_node("generate_answer", generate_answer_node)

    graph.add_edge(START, "classify_query")

    # Dynamic conditional edge based on datasource
    graph.add_conditional_edges(
        "classify_query",
        route_query,
        {
            "vectorstore": "vectorstore",
            "web_search": "web_search",
            "direct": "direct"
        }
    )

    # All paths converge on answer generation
    graph.add_edge("vectorstore", "generate_answer")
    graph.add_edge("web_search", "generate_answer")
    graph.add_edge("direct", "generate_answer")
    graph.add_edge("generate_answer", END)

    app = graph.compile()

    # --- 6. Test Various Query Types ---
    test_queries = [
        "What is our internal policy on PTO days?",
        "What is today's current weather in Tokyo?",
        "Explain what a state machine is."
    ]

    for idx, q in enumerate(test_queries, start=1):
        print("\n" + "=" * 60)
        print(f"TEST CASE {idx}: '{q}'")
        print("=" * 60)

        result = app.invoke({
            "question": q,
            "datasource": "",
            "retrieved_docs": [],
            "answer": ""
        })

        print("\n--- Final Generated Answer ---")
        print(result["answer"])


if __name__ == "__main__":
    main()
