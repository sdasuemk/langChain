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

## 3. How to Run Phase 1

Activate your virtual environment and run the lessons:

```powershell
# In Windows PowerShell:
.\.venv\Scripts\python.exe langGraph/01_simple_state_graph.py
.\.venv\Scripts\python.exe langGraph/02_reducers_and_messages.py
.\.venv\Scripts\python.exe langGraph/03_llm_state_graph.py
```

---

## Ready for Phase 2?
Once you run and review these three scripts, we move to **Phase 2: Conditional Routing & Tools**, where we build a **custom ReAct Agent** with conditional edges (`tools_condition`) and `ToolNode`.
