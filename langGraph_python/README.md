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

### [11_parallel_tool_calling.py](file:///c:/Coding/langChain/langGraph/11_parallel_tool_calling.py) — Parallel Tool Execution & Recursion Limits
* **Concepts**:
  * Handling multiple tool calls emitted in a single turn.
  * `ToolNode` executing multiple tool calls concurrently.
  * Protecting agent workflows against runaway loops using `config={"recursion_limit": N}` and catching `GraphRecursionError`.

### [12_dynamic_tool_selection.py](file:///c:/Coding/langChain/langGraph/12_dynamic_tool_selection.py) — Dynamic Tool Selection & RBAC
* **Concepts**:
  * Context-aware tool provisioning: filtering available tools based on user roles (Admin vs Guest) or conversation state.
  * Subclassing `MessagesState` with domain metadata (`user_role`, `authorized_tools`).

### [13_structured_output_agent.py](file:///c:/Coding/langChain/langGraph/13_structured_output_agent.py) — Structured Output via Tool Schemas
* **Topology**:
  ```
                     START
                       │
                       ▼
                 ┌───────────┐
         ┌──────►│   agent   │ (Research tools, then final submission tool)
         │       └─────┬─────┘
         │             │
         │      [custom_router]
         │        /    │    \
         │  (research) │   (final_schema_tool)
         │      /      │      \
         │     ▼       │       ▼
         └── ToolNode  │   extract_structured_data
                       ▼       │
                      END      ▼
                              END
  ```
* **Concepts**:
  * Forcing typed outputs by binding Pydantic models as final submission tools.
  * Routing dynamically between intermediate helper tools and final schema extraction nodes.

---

---

## 5. Phase 3: Persistence, Memory & Checkpointers

### [14_in_memory_persistence.py](file:///c:/Coding/langChain/langGraph/14_in_memory_persistence.py) — In-Memory Checkpointing (`MemorySaver`)
* **Concepts**:
  * Using `MemorySaver` to checkpoint state after each super-step.
  * Isolating conversations using `thread_id`: `config={"configurable": {"thread_id": "session_123"}}`.
  * Passing only NEW messages while LangGraph automatically restores prior conversation context.

### [15_sqlite_disk_persistence.py](file:///c:/Coding/langChain/langGraph/15_sqlite_disk_persistence.py) — SQLite Disk Persistence (`SqliteSaver`)
* **Concepts**:
  * Persisting agent state to disk via SQLite (`state_checkpoints.db`).
  * Surviving process terminations and server restarts: resuming threads across independent runs.

### [16_state_inspection_and_history.py](file:///c:/Coding/langChain/langGraph/16_state_inspection_and_history.py) — State Inspection & History Audit
* **Concepts**:
  * Inspecting current state snapshots with `app.get_state(config)`.
  * Reading checkpoint metadata, scheduled next nodes, and message stacks.
  * Traversing the complete chronological timeline with `app.get_state_history(config)`.

### [17_time_travel_and_state_editing.py](file:///c:/Coding/langChain/langGraph/17_time_travel_and_state_editing.py) — Time Travel & State Editing
* **Concepts**:
  * Editing state manually with `app.update_state(config, values, as_node=...)`.
  * Time travel: selecting a historical checkpoint ID and forking an alternate conversation timeline.

---

## 6. Phase 4: Human-in-the-Loop (HITL)

### [18_static_breakpoints.py](file:///c:/Coding/langChain/langGraph/18_static_breakpoints.py) — Static Breakpoints (`interrupt_before`)
* **Concepts**:
  * Halting graph execution before sensitive nodes with `interrupt_before=[node_name]`.
  * Inspecting paused status via `app.get_state(config).next`.
  * Resuming execution by calling `app.invoke(None, config=config)`.

### [19_dynamic_interrupt.py](file:///c:/Coding/langChain/langGraph/19_dynamic_interrupt.py) — Dynamic In-Node Interrupts
* **Concepts**:
  * Pausing mid-node using modern `interrupt(payload)` to surface questions or drafts to the human.
  * Resuming execution and injecting human responses directly back into the node using `Command(resume=value)`.

### [20_action_approval_workflow.py](file:///c:/Coding/langChain/langGraph/20_action_approval_workflow.py) — Full Approval Workflow (Approve/Edit/Reject)
* **Concepts**:
  * The 3 core human oversight branches:
    1. **Approve**: Execute unchanged.
    2. **Edit**: Modify parameters via `app.update_state()` before allowing execution.
    3. **Reject**: Abort and route to an escalation/cancellation node.

### [21_tool_call_interception.py](file:///c:/Coding/langChain/langGraph/21_tool_call_interception.py) — Tool Call Interception & Verification
* **Concepts**:
  * Segregating safe read-only tools from high-stakes mutation tools.
  * Pausing execution before `ToolNode` runs for sensitive tool calls.
  * Modifying tool arguments in-flight prior to resumption.

---

---

## 7. Phase 5: Agentic RAG Patterns

### [22_router_rag.py](file:///c:/Coding/langChain/langGraph/22_router_rag.py) — Router RAG
* **Concepts**:
  * Intent-based query classification to direct traffic between Vector DB, Web Search, or Direct LLM answering.
  * Prevents wasteful vector lookups for general chit-chat and out-of-domain questions.

### [23_corrective_rag.py](file:///c:/Coding/langChain/langGraph/23_corrective_rag.py) — Corrective RAG (CRAG)
* **Concepts**:
  * Evaluating document relevance before passing chunks to generation.
  * Self-correction loop: if retrieved context is poor, the agent rewrites the query and triggers a web search fallback.

### [24_self_rag.py](file:///c:/Coding/langChain/langGraph/24_self_rag.py) — Self-RAG (Reflection Loop)
* **Concepts**:
  * Two-tier quality validation:
    1. **Hallucination Grader**: Ensures response is grounded in retrieved facts.
    2. **Usefulness Grader**: Verifies response directly answers the question.
  * Automatic cyclic regeneration if claims contradict the source text.

### [25_adaptive_rag.py](file:///c:/Coding/langChain/langGraph/25_adaptive_rag.py) — Adaptive RAG (Strategy Selection)
* **Concepts**:
  * Assessing query complexity dynamically:
    - **Simple**: Fast direct answer.
    - **Standard**: Single-hop vector retrieval.
    - **Complex**: Multi-hop query decomposition with sequential iterative retrieval.

---

## 8. How to Run All Lessons

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
.\.venv\Scripts\python.exe langGraph/11_parallel_tool_calling.py
.\.venv\Scripts\python.exe langGraph/12_dynamic_tool_selection.py
.\.venv\Scripts\python.exe langGraph/13_structured_output_agent.py

# Phase 3: Persistence, Memory & Checkpointers
.\.venv\Scripts\python.exe langGraph/14_in_memory_persistence.py
.\.venv\Scripts\python.exe langGraph/15_sqlite_disk_persistence.py
.\.venv\Scripts\python.exe langGraph/16_state_inspection_and_history.py
.\.venv\Scripts\python.exe langGraph/17_time_travel_and_state_editing.py

# Phase 4: Human-in-the-Loop (HITL)
.\.venv\Scripts\python.exe langGraph/18_static_breakpoints.py
.\.venv\Scripts\python.exe langGraph/19_dynamic_interrupt.py
.\.venv\Scripts\python.exe langGraph/20_action_approval_workflow.py
.\.venv\Scripts\python.exe langGraph/21_tool_call_interception.py

# Phase 5: Agentic RAG Patterns
.\.venv\Scripts\python.exe langGraph/22_router_rag.py
.\.venv\Scripts\python.exe langGraph/23_corrective_rag.py
.\.venv\Scripts\python.exe langGraph/24_self_rag.py
.\.venv\Scripts\python.exe langGraph/25_adaptive_rag.py
```

---

## 8. Phase 6: Multi-Agent Systems & Production

### [26_supervisor_multi_agent.py](file:///c:/Coding/langChain/langGraph/26_supervisor_multi_agent.py) — Supervisor Pattern
* **Concepts**:
  * Centralized orchestrator LLM (Supervisor) assessing overall goals and delegating sub-tasks to specialized workers (Researcher, Coder).
  * Workers complete tasks and report back to the supervisor until `FINISH` is triggered.

### [27_hierarchical_subgraphs.py](file:///c:/Coding/langChain/langGraph/27_hierarchical_subgraphs.py) — Hierarchical Subgraphs
* **Concepts**:
  * Nesting compiled graphs as first-class nodes inside parent graphs: `parent_graph.add_node("qa_subsystem", compiled_subgraph)`.
  * State isolation: encapsulates internal QA pipelines and linting checks away from the parent release management graph.

### [28_swarm_handoffs.py](file:///c:/Coding/langChain/langGraph/28_swarm_handoffs.py) — Swarm / Network Peer Handoffs
* **Concepts**:
  * Decentralized peer-to-peer agent handoffs without a central bottleneck.
  * Triage Agent handing off to Billing, which dynamically escalates multi-hop to Tech Support.

### [29_production_streaming.py](file:///c:/Coding/langChain/langGraph/29_production_streaming.py) — Production Streaming Modes
* **Concepts**:
  * `stream_mode="updates"`: Yields step-by-step state diffs produced by each node.
  * `stream_mode="values"`: Yields full state snapshots after every super-step.
  * Token-by-token streaming: Streaming real-time token chunks for responsive frontends.

---

---

## 9. Phase 7: Advanced Production Engineering

### [30_map_reduce_send_api.py](file:///c:/Coding/langChain/langGraph/30_map_reduce_send_api.py) — Dynamic Map-Reduce (`Send` API)
* **Concepts**:
  * Unlike static parallel branches, the `Send(node, payload)` API dynamically spawns $N$ worker instances at runtime based on task generation.
  * Automatic fan-in reduction via an `operator.add` reducer list.

### [31_memory_trim_and_summary.py](file:///c:/Coding/langChain/langGraph/31_memory_trim_and_summary.py) — Memory Compaction & Summarization
* **Concepts**:
  * Overcoming the context-window overflow problem in long-running persistent threads.
  * Conditional summarization: condensing older messages into a running `summary` and purging older raw records using `RemoveMessage`.

### [32_cross_thread_memory_store.py](file:///c:/Coding/langChain/langGraph/32_cross_thread_memory_store.py) — Cross-Thread Memory (`Store`)
* **Concepts**:
  * Checkpointers preserve state *within* a thread. LangGraph's `Store` (`InMemoryStore`) preserves global user profiles, facts, and preferences *across different threads*.
  * Hierarchical namespacing: `store.put(("users", user_id), "preferences", {...})`.

### [33_modern_command_control.py](file:///c:/Coding/langChain/langGraph/33_modern_command_control.py) — Modern `Command` Control Flow
* **Concepts**:
  * Unifying state updates and edge routing into a single return: `return Command(update={...}, goto="target_node")`.
  * Eliminates boilerplate conditional edge mapping when decisions are computed directly inside nodes.

### [34_async_graph_execution.py](file:///c:/Coding/langChain/langGraph/34_async_graph_execution.py) — Asynchronous Execution (`ainvoke` / `astream`)
* **Concepts**:
  * Writing non-blocking asynchronous graphs with `async def` nodes.
  * Running concurrent workflows in web environments (FastAPI/Starlette) with `await app.ainvoke(...)` and `async for event in app.astream(...)`.

---

## 10. Complete Master Curriculum (34 Lessons)

Activate your virtual environment and run any lesson:

```powershell
# In Windows PowerShell:
# Phase 1: Core Fundamentals & Topologies
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
.\.venv\Scripts\python.exe langGraph/11_parallel_tool_calling.py
.\.venv\Scripts\python.exe langGraph/12_dynamic_tool_selection.py
.\.venv\Scripts\python.exe langGraph/13_structured_output_agent.py

# Phase 3: Persistence, Memory & Checkpointers
.\.venv\Scripts\python.exe langGraph/14_in_memory_persistence.py
.\.venv\Scripts\python.exe langGraph/15_sqlite_disk_persistence.py
.\.venv\Scripts\python.exe langGraph/16_state_inspection_and_history.py
.\.venv\Scripts\python.exe langGraph/17_time_travel_and_state_editing.py

# Phase 4: Human-in-the-Loop (HITL)
.\.venv\Scripts\python.exe langGraph/18_static_breakpoints.py
.\.venv\Scripts\python.exe langGraph/19_dynamic_interrupt.py
.\.venv\Scripts\python.exe langGraph/20_action_approval_workflow.py
.\.venv\Scripts\python.exe langGraph/21_tool_call_interception.py

# Phase 5: Agentic RAG Patterns
.\.venv\Scripts\python.exe langGraph/22_router_rag.py
.\.venv\Scripts\python.exe langGraph/23_corrective_rag.py
.\.venv\Scripts\python.exe langGraph/24_self_rag.py
.\.venv\Scripts\python.exe langGraph/25_adaptive_rag.py

# Phase 6: Multi-Agent Systems & Production
.\.venv\Scripts\python.exe langGraph/26_supervisor_multi_agent.py
.\.venv\Scripts\python.exe langGraph/27_hierarchical_subgraphs.py
.\.venv\Scripts\python.exe langGraph/28_swarm_handoffs.py
.\.venv\Scripts\python.exe langGraph/29_production_streaming.py

# Phase 7: Advanced Production Engineering
.\.venv\Scripts\python.exe langGraph/30_map_reduce_send_api.py
.\.venv\Scripts\python.exe langGraph/31_memory_trim_and_summary.py
.\.venv\Scripts\python.exe langGraph/32_cross_thread_memory_store.py
.\.venv\Scripts\python.exe langGraph/33_modern_command_control.py
.\.venv\Scripts\python.exe langGraph/34_async_graph_execution.py
```

---

## Congratulations!
You now possess a complete 34-lesson, 7-phase master curriculum for LangGraph covering state machines, reducers, custom ReAct agents, checkpointer memory, human-in-the-loop approvals, agentic RAG, multi-agent architectures, dynamic map-reduce (`Send`), cross-thread stores, and async production streaming!
