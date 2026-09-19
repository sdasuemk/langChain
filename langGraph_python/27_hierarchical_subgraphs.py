"""
=============================================================================
Phase 6: Multi-Agent Systems & Production - Lesson 27: Hierarchical Subgraphs
=============================================================================

Pattern:
                     PARENT GRAPH
                   ┌──────────────┐
                   │    START     │
                   └──────┬───────┘
                          │
                          ▼
                   ┌──────────────┐
                   │  plan_task   │
                   └──────┬───────┘
                          │
                          ▼
            ╔═════════════════════════════════╗
            ║     QA SUBSYSTEM (SUBGRAPH)     ║
            ║                                 ║
            ║    START -> run_linter          ║
            ║                │                ║
            ║                ▼                ║
            ║           run_tests -> END      ║
            ╚═════════════════════════════════╝
                          │
                          ▼
                   ┌──────────────┐
                   │deploy_release│
                   └──────┬───────┘
                          │
                          ▼
                   ┌──────────────┐
                   │     END      │
                   └──────────────┘

Key Concepts:
1. Subgraphs as First-Class Nodes:
   In LangGraph, any compiled graph (`subgraph.compile()`) can be added directly
   as a node in a parent graph: `parent_graph.add_node("qa_team", compiled_subgraph)`.
2. State Isolation & Composability:
   Subgraphs can manage their own internal loops, states, and checkpointers,
   keeping high-level orchestration clean and maintainable.
=============================================================================
"""

import sys
import io
from typing import List
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

# Ensure proper utf-8 encoding for Windows terminals
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


# ---------------------------------------------------------------------------
# 1. Define Child Subgraph (QA Verification Team)
# ---------------------------------------------------------------------------
class SubgraphState(TypedDict):
    feature_name: str
    qa_reports: List[str]
    is_qa_approved: bool


def linter_node(state: SubgraphState) -> dict:
    print(f"    [Subgraph: Linter] Checking code styling for '{state['feature_name']}'...")
    return {
        "qa_reports": state["qa_reports"] + ["Linter Check: 0 syntax or PEP8 errors."]
    }


def unit_test_node(state: SubgraphState) -> dict:
    print(f"    [Subgraph: Unit Tests] Running test suites...")
    return {
        "qa_reports": state["qa_reports"] + ["Unit Tests: 42 passed, 0 failed."],
        "is_qa_approved": True
    }


def build_qa_subgraph():
    """Assembles and compiles the self-contained QA subgraph."""
    subgraph = StateGraph(SubgraphState)
    subgraph.add_node("linter", linter_node)
    subgraph.add_node("unit_tests", unit_test_node)

    subgraph.add_edge(START, "linter")
    subgraph.add_edge("linter", "unit_tests")
    subgraph.add_edge("unit_tests", END)

    return subgraph.compile()


# ---------------------------------------------------------------------------
# 2. Define Parent Graph (Release Management)
# ---------------------------------------------------------------------------
# The parent graph shares the common keys that the subgraph expects
class ParentState(TypedDict):
    feature_name: str
    qa_reports: List[str]
    is_qa_approved: bool
    deployment_status: str


def planning_node(state: ParentState) -> dict:
    print(f"\n[Parent Graph: Planning] Initializing rollout for: '{state['feature_name']}'...")
    return {
        "qa_reports": [],
        "is_qa_approved": False,
        "deployment_status": "Planning Complete"
    }


def deploy_node(state: ParentState) -> dict:
    print(f"\n[Parent Graph: Deploy] Verifying QA approval...")
    if state["is_qa_approved"]:
        print("  --> QA Sign-off verified! Deploying to production cluster.")
        status = "Deployed to Production (v2.4.0)"
    else:
        status = "Deployment Rejected: QA failed"
    return {"deployment_status": status}


def main():
    print("=" * 65)
    print("  LangGraph Phase 6: Lesson 27 - Hierarchical Subgraphs")
    print("=" * 65)

    # 1. Compile the QA Subgraph
    compiled_qa_subgraph = build_qa_subgraph()

    # 2. Assemble the Parent Graph
    parent_graph = StateGraph(ParentState)

    parent_graph.add_node("planning", planning_node)
    # Notice: The compiled subgraph is plugged in directly as a single node!
    parent_graph.add_node("qa_subsystem", compiled_qa_subgraph)
    parent_graph.add_node("deploy", deploy_node)

    # Connect parent flow: planning -> qa_subsystem -> deploy -> END
    parent_graph.add_edge(START, "planning")
    parent_graph.add_edge("planning", "qa_subsystem")
    parent_graph.add_edge("qa_subsystem", "deploy")
    parent_graph.add_edge("deploy", END)

    parent_app = parent_graph.compile()

    # 3. Execute Parent Pipeline
    initial_input: ParentState = {
        "feature_name": "OAuth2 Social Login",
        "qa_reports": [],
        "is_qa_approved": False,
        "deployment_status": "Pending"
    }

    print("\nStarting execution of parent workflow...")
    final_state = parent_app.invoke(initial_input)

    print("\n" + "=" * 65)
    print("Parent Workflow Completed!")
    print(f"Feature Name       : {final_state['feature_name']}")
    print(f"Deployment Status  : {final_state['deployment_status']}")
    print("QA Subsystem Reports:")
    for report in final_state["qa_reports"]:
        print(f"  * {report}")
    print("=" * 65)


if __name__ == "__main__":
    main()
