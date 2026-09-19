"""
=============================================================================
Phase 1: Core Graph Patterns - Lesson 7: Loop (Cyclic) Graph & Recursion Limit
=============================================================================

Pattern:
                     START
                       |
                 generate_code  <---------------+ (Loop back with feedback)
                       |                        |
                   test_code                    |
                       |                        |
               [evaluator_router] --------------+
                  /          \
            (all pass)    (max attempts reached)
                /              \
               +--------+-------+
                        |
                       END

Characteristics:
- Cycles are the signature capability of LangGraph (impossible in linear LCEL DAGs).
- Enables iterative self-correction: LLM generates, code/critic evaluates,
  and if validation fails, state loops back for another revision.
- Loop Safety:
  1. Internal Guard: Track `attempt_count` in state and exit when a maximum is reached.
  2. LangGraph Safety Net: `recursion_limit` in config prevents runaway infinite loops.
=============================================================================
"""

import sys
import io
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

# Ensure proper utf-8 encoding for Windows terminals
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


# --- 1. Define State with Iteration Tracking ---
class CodeRefinementState(TypedDict):
    task_description: str
    current_code: str
    test_result: str
    feedback: str
    iteration: int
    max_iterations: int


# --- 2. Define Generator and Tester Nodes ---
def generate_code_node(state: CodeRefinementState) -> dict:
    current_iter = state.get("iteration", 0) + 1
    feedback = state.get("feedback", "None")

    print(f"\n[Generator Node] Iteration {current_iter}: Writing code...")
    print(f"  Incorporating feedback: '{feedback}'")

    # Simulate progressive improvement over iterations
    if current_iter == 1:
        # First attempt: has a bug (missing return)
        code = "def solve():\n    x = 42"
    elif current_iter == 2:
        # Second attempt: has a type error
        code = "def solve():\n    return '42' + 1"
    else:
        # Third attempt: correct
        code = "def solve():\n    return 42 + 1"

    return {
        "current_code": code,
        "iteration": current_iter
    }


def test_code_node(state: CodeRefinementState) -> dict:
    code = state["current_code"]
    print(f"[Tester Node] Running unit tests on:\n{code}")

    # Simulated test runner
    if "return" not in code:
        test_result = "FAIL"
        feedback = "Error: Function must return a value."
    elif "'42' + 1" in code:
        test_result = "FAIL"
        feedback = "TypeError: Can only concatenate str to str, not int."
    else:
        test_result = "PASS"
        feedback = "All 3 unit tests passed successfully."

    print(f"  Result: {test_result} | Feedback: {feedback}")

    return {
        "test_result": test_result,
        "feedback": feedback
    }


# --- 3. Define Conditional Loop Router ---
def should_continue(state: CodeRefinementState) -> str:
    """
    Decides whether to loop back to 'generate_code' or terminate at END.
    """
    if state["test_result"] == "PASS":
        print("\n--> [Router]: Tests PASSED! Directing to END.")
        return "finish"

    if state["iteration"] >= state["max_iterations"]:
        print(f"\n--> [Router]: Max iterations ({state['max_iterations']}) reached. Exiting to END.")
        return "finish"

    print("\n--> [Router]: Tests FAILED. Looping back to 'generate_code'...")
    return "retry"


def main():
    print("=" * 65)
    print("  LangGraph Pattern: Cyclic (Loop) Execution Flow")
    print("=" * 65)

    # --- 4. Assemble Cyclic Graph ---
    graph = StateGraph(CodeRefinementState)

    # Register nodes
    graph.add_node("generate_code", generate_code_node)
    graph.add_node("test_code", test_code_node)

    # Sequence
    graph.add_edge(START, "generate_code")
    graph.add_edge("generate_code", "test_code")

    # Conditional Cycle:
    # From 'test_code', evaluate test results:
    # - "retry"  -> loops back to "generate_code"
    # - "finish" -> routes to END
    graph.add_conditional_edges(
        "test_code",
        should_continue,
        {
            "retry": "generate_code",
            "finish": END
        }
    )

    # Compile
    app = graph.compile()

    # --- 5. Run the Iterative Refinement Loop ---
    initial_input: CodeRefinementState = {
        "task_description": "Write a function solve() that calculates 42 + 1",
        "current_code": "",
        "test_result": "",
        "feedback": "",
        "iteration": 0,
        "max_iterations": 5
    }

    # config recursion_limit protects against runaway execution
    config = {"recursion_limit": 15}

    print("\nStarting execution with recursion_limit=15...")
    final_state = app.invoke(initial_input, config=config)

    print("\n" + "=" * 65)
    print("Execution Finished!")
    print(f"Total Iterations: {final_state['iteration']}")
    print(f"Final Test Result: {final_state['test_result']}")
    print(f"Final Approved Code:\n{final_state['current_code']}")
    print("=" * 65)


if __name__ == "__main__":
    main()
