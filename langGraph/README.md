# LangGraph Mastery - Phase 1: Core Fundamentals

Welcome to **Phase 1** of your LangGraph journey. In this phase, we master the core engine of LangGraph before adding tools, memory, or complex agents.

---

## 1. The Core Mental Model

In traditional LangChain (LCEL), everything flows in one direction (DAG):
```
PromptTemplate -> LLM -> OutputParser
```

In **LangGraph**, workflows are modeled as **State Machines**:
1. **State**: The single source of truth (data container) passed across nodes.
2. **Nodes**: Pure Python functions that receive the current state and return an update dictionary.
3. **Edges**: Direct connections between nodes (`START`, custom nodes, `END`).
4. **Reducers**: Rules that determine how node outputs are merged into the state (e.g. overwrite vs append).

```
   +-----------+
   |   START   |
   +-----+-----+
         |
         v
   +-----------+
   |   Node    | <-----+ (Cycles possible!)
   +-----+-----+       |
         |             |
         v             |
   +-----------+       |
   | Condition |-------+
   +-----+-----+
         |
         v
   +-----------+
   |    END    |
   +-----------+
```

---

## 2. Phase 1 Code Walkthrough

We have created 3 progressive lessons in this directory:

### [01_simple_state_graph.py](file:///c:/Coding/langChain/langGraph/01_simple_state_graph.py)
* **Focus**: Pure Python state machine with no external APIs.
* **Concepts**:
  * Defining state with `typing_extensions.TypedDict`.
  * Creating nodes: `def node_func(state: MyState) -> dict`.
  * Connecting nodes with `graph.add_edge(START, "node_a")` and `graph.add_edge("node_a", END)`.
  * Compiling with `graph.compile()`.
  * Running with `app.invoke(initial_state)`.

### [02_reducers_and_messages.py](file:///c:/Coding/langChain/langGraph/02_reducers_and_messages.py)
* **Focus**: How state updates are merged (Reducers).
* **Concepts**:
  * **Default behavior**: When a node returns `{"key": new_val}`, it **replaces** `key`.
  * **List Append Reducer**: `Annotated[list, operator.add]` ensures returning `[item]` appends to the list instead of erasing previous items.
  * **`add_messages` Reducer**: The cornerstone of chat agents. It appends new messages, tracks message IDs, and updates existing messages if an ID matches.

### [03_llm_state_graph.py](file:///c:/Coding/langChain/langGraph/03_llm_state_graph.py)
* **Focus**: Connecting an LLM with `MessagesState`.
* **Concepts**:
  * Using LangGraph's prebuilt `MessagesState`.
  * Streaming node updates with `app.stream(..., stream_mode="updates")`.
  * Subclassing `MessagesState` to add custom tracking fields (e.g., user profile, step counter).

---

## 3. Core Graph Topologies & Patterns

### [04_serial_graph.py](file:///c:/Coding/langChain/langGraph/04_serial_graph.py) — Linear Pipeline
* **Topology**: `START -> clean_text -> count_words -> generate_summary -> END`
* **When to use**: Sequential data pipelines where each step directly relies on the output of the previous step.

### [05_parallel_graph.py](file:///c:/Coding/langChain/langGraph/05_parallel_graph.py) — Fan-Out / Fan-In
* **Topology**: 
  ```
                START
                  │
             input_reader
              ┌───┴───┐
              ▼       ▼
          sentiment  keywords  (Concurrent execution)
              └───┬───┘
                  ▼
              aggregator
                  │
                 END
  ```
* **When to use**: Independent tasks that can execute at the same time. The aggregator automatically waits for all incoming branches (barrier synchronization).
* **State safety**: Uses `Annotated[List[str], operator.add]` so parallel nodes can append to a shared audit log without collision.

### [06_conditional_graph.py](file:///c:/Coding/langChain/langGraph/06_conditional_graph.py) — Dynamic Routing
* **Topology**:
  ```
                     START
                       │
                 classify_ticket
                       │
               [routing_decision]  (via add_conditional_edges)
              ┌────────┼────────┐
              ▼        ▼        ▼
           billing  technical  general
              └────────┼────────┘
                       │
                   send_reply
                       │
                      END
  ```
* **When to use**: Intelligent triage, agent handoffs, or intent-based routing.

### [07_loop_graph.py](file:///c:/Coding/langChain/langGraph/07_loop_graph.py) — Cyclic Execution & Self-Correction
* **Topology**:
  ```
                     START
                       │
                 generate_code ◄────────┐ (Loop back with feedback)
                       │                │
                   test_code            │
                       │                │
               [evaluator_router] ──────┘
                       │
                     (pass)
                       │
                      END
  ```
* **When to use**: Iterative generation (code synthesis, document drafting, self-reflection).
* **Safety**: State-based iteration guards and LangGraph's `recursion_limit` safeguard against infinite loops.

---

## 4. How to Run

Activate your virtual environment and run the lessons:

```powershell
# In Windows PowerShell:
# 1. Fundamentals
.\.venv\Scripts\python.exe langGraph/01_simple_state_graph.py
.\.venv\Scripts\python.exe langGraph/02_reducers_and_messages.py
.\.venv\Scripts\python.exe langGraph/03_llm_state_graph.py

# 2. Graph Topologies & Patterns
.\.venv\Scripts\python.exe langGraph/04_serial_graph.py
.\.venv\Scripts\python.exe langGraph/05_parallel_graph.py
.\.venv\Scripts\python.exe langGraph/06_conditional_graph.py
.\.venv\Scripts\python.exe langGraph/07_loop_graph.py
```

---

---

## 4. Phase 2: Tool Calling & Dynamic Routing

### [08_custom_react_agent.py](file:///c:/Coding/langChain/langGraph/08_custom_react_agent.py) — Custom ReAct Agent from Scratch
* **Topology**:
  ```
                     START
                       │
                       ▼
                 ┌───────────┐
         ┌──────►│   agent   │
         │       └─────┬─────┘
         │             │
         │     [tools_condition]
         │        /         \
         │   (has tools)   (no tools)
         │      /             \
         │     ▼               ▼
         └── ToolNode         END
  ```
* **Concepts**:
  * Using `ToolNode(tools)` from `langgraph.prebuilt`.
  * Using `tools_condition` for automatic branching based on `AIMessage.tool_calls`.
  * Cycling tool output back to the agent for final answer synthesis.

### [09_prebuilt_react_agent.py](file:///c:/Coding/langChain/langGraph/09_prebuilt_react_agent.py) — Production Shortcut
* **Concepts**:
  * Using `create_react_agent(model, tools, prompt=...)`.
  * How the prebuilt helper bundles state management, tool binding, and loop edges in one line.

### [10_advanced_tool_routing.py](file:///c:/Coding/langChain/langGraph/10_advanced_tool_routing.py) — Segregated Tools & Error Recovery
* **Topology**:
  ```
                     START
                       │
                       ▼
                     agent
                       │
             [route_tools_by_risk]
            /          │          \
       (read_tools) (write_tools) (END)
            │          │            │
            ▼          ▼            │
        SafeTools  MutationTools    │
            │          │            │
            └──────────┴────────────┘
                       │
                       ▼
                     agent
  ```
* **Concepts**:
  * Splitting tools into multiple `ToolNode` instances according to permission or risk level.
  * Setting `ToolNode(..., handle_tool_errors=True)` so exceptions turn into feedback for the agent rather than crashing.

---

## 5. How to Run All Lessons

Activate your virtual environment and run the lessons:

```powershell
# In Windows PowerShell:
# Phase 1: Core Fundamentals & Patterns
.\.venv\Scripts\python.exe langGraph/01_simple_state_graph.py
.\.venv\Scripts\python.exe langGraph/02_reducers_and_messages.py
.\.venv\Scripts\python.exe langGraph/03_llm_state_graph.py
.\.venv\Scripts\python.exe langGraph/04_serial_graph.py
.\.venv\Scripts\python.exe langGraph/05_parallel_graph.py
.\.venv\Scripts\python.exe langGraph/06_conditional_graph.py
.\.venv\Scripts\python.exe langGraph/07_loop_graph.py

# Phase 2: Tool Calling & Dynamic Routing
.\.venv\Scripts\python.exe langGraph/08_custom_react_agent.py
.\.venv\Scripts\python.exe langGraph/09_prebuilt_react_agent.py
.\.venv\Scripts\python.exe langGraph/10_advanced_tool_routing.py
```

---

## Ready for Phase 3?
Next up is **Phase 3: Persistence, Memory & Checkpointers** (`MemorySaver`, `SqliteSaver`, `thread_id`, state rewind, and multi-user sessions).
