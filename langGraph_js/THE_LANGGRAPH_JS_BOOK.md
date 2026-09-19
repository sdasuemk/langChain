# The LangGraph.js Handbook: Zero to Hero
*The Complete JavaScript & TypeScript Guide to Building Stateful, Multi-Agent AI Systems*

---

## Welcome to the JavaScript Kitchen

If you have built AI apps using traditional chains (like standard LCEL or prompt pipes), you built an **assembly line**:
```
User Input -> Prompt Template -> Model -> Output Parser
```
This is a **DAG (Directed Acyclic Graph)**. It flows in one direction.

### Why DAGs Break in Real-World Web Apps
* What if the LLM output is malformed and you need to loop back to retry?
* What if an AI agent needs to pause and ask the user for approval via a web UI before deleting a database record?
* What if an agent needs to maintain multi-turn memory per user session across multiple browser requests?

Conveyor belts cannot loop backward or pause gracefully.

**LangGraph.js is a Michelin-Star Restaurant Kitchen modeled as a State Machine.**
* The **Order Ticket** is the **State** (defined with `Annotation.Root`).
* The **Chefs** at specialized prep stations are **Nodes** (JavaScript functions).
* The **Runners** moving tickets between stations are **Edges** (`graph.addEdge`).
* The **Expeditor** triaging orders to different stations is the **Router** (`graph.addConditionalEdges`).
* The **Head Chef** tasting food before serving is **Human-in-the-Loop** (`interruptBefore`).

Let’s master LangGraph.js from Zero to Hero.

---

# Phase 1: Core Fundamentals & Topologies in JavaScript

### 1.1 The "Hello World" of LangGraph.js (The Clean 20-Line Example)

Here is the absolute simplest, cleanest LangGraph.js application you can run in Node.js:

```javascript
import { Annotation, StateGraph, START, END } from "@langchain/langgraph";

// 1. Define State (The shared data container)
const SimpleStateAnnotation = Annotation.Root({
  message: Annotation(),
});

// 2. Define a Node (A pure function that updates state)
function greetNode(state) {
  return { message: `${state.message} -> Processed by JavaScript Chef!` };
}

// 3. Assemble the Graph
const graph = new StateGraph(SimpleStateAnnotation);
graph.addNode("greeter", greetNode);

// 4. Connect the Edges
graph.addEdge(START, "greeter");
graph.addEdge("greeter", END);

// 5. Compile & Run
const app = graph.compile();
const result = await app.invoke({ message: "Hello LangGraph.js" });

console.log(result.message);
// Output: Hello LangGraph.js -> Processed by JavaScript Chef!
```

---

### 1.2 Breaking Down the Mental Model in JavaScript

#### 1. Defining State with `Annotation.Root`
In Python, LangGraph uses `TypedDict`. In JavaScript/TypeScript, LangGraph uses **`Annotation.Root`**. This defines both the TypeScript types and the runtime reducer rules:

```javascript
const OrderAnnotation = Annotation.Root({
  dishName: Annotation(),
  spiceLevel: Annotation({
    reducer: (current, update) => (update !== undefined ? update : current),
    default: () => 1,
  }),
  chefNotes: Annotation({
    // Concatenates items instead of overwriting!
    reducer: (current, update) => current.concat(update),
    default: () => [],
  }),
});
```

#### 2. Adding Nodes with `graph.addNode`
A node is just a JavaScript function (sync or `async`). It receives the current `state` and returns an object of updates:

```javascript
function grillStation(state) {
  return {
    spiceLevel: state.spiceLevel + 1,
    chefNotes: ["Steak seared on iron skillet."],
  };
}

// Register the node with a unique string ID:
graph.addNode("grill", grillStation);
```

#### 3. Connecting Static Edges with `graph.addEdge`
* `START`: Where the request enters.
* `END`: Where the completed response exits.
* `graph.addEdge("nodeA", "nodeB")`: Connects Node A directly to Node B.

```javascript
graph.addEdge(START, "prep");
graph.addEdge("prep", "grill");
graph.addEdge("grill", END);
```

#### 4. Conditional Routing with `graph.addConditionalEdges`
When traffic needs to branch dynamically based on state:

> **The Expeditor Story:** When an order arrives, the expeditor checks if it is vegan. If vegan $\to$ Salad Bar; otherwise $\to$ Grill.

```javascript
// Step A: The pure Router function
function routeByDiet(state) {
  return state.isVegan ? "salad" : "meat";
}

// Step B: Register conditional edge
graph.addConditionalEdges(
  "orderTriage",  // Source node
  routeByDiet,    // Router function
  {
    salad: "saladStation",  // router return key -> target node
    meat: "grillStation",
  }
);
```

---

### 1.3 The State Reducer Mystery in JavaScript

> **The Rookie Waiter Scenario:**
> A customer says: *"I want a burger."* The waiter writes: `Order: Burger`.
> Five minutes later, they add: *"And fries please."*
> If the waiter erases `Burger` and only writes `Fries`, the customer starves. That is the **default overwrite behavior**.

In JavaScript:
```javascript
// Overwrite (Default):
reducer: (curr, update) => update ?? curr

// Append (Reducer):
reducer: (curr, update) => curr.concat(update)
```

#### Prebuilt `MessagesAnnotation`
For chat applications, LangGraph.js provides `MessagesAnnotation` out of the box:
```javascript
import { MessagesAnnotation, StateGraph } from "@langchain/langgraph";

const graph = new StateGraph(MessagesAnnotation);
// Automatically handles HumanMessage, AIMessage, and tool results!
```

---

### 1.4 The 4 Fundamental Topologies

```
1. SERIAL (Pipeline)
   START ──► Clean ──► Transform ──► Store ──► END

2. PARALLEL (Fan-Out / Fan-In)
              ┌──► Analyze Sentiment ──┐
   START ─────┤                        ├────► Aggregator ──► END
              └──► Extract Keywords ───┘

3. CONDITIONAL (Dynamic Routing)
                   ┌──► Billing Node
   START ──► Router ──► Technical Node
                   └──► General FAQ

4. LOOP (Cyclic Refinement)
             ┌──────────────┐
             ▼              │
   START ──► Draft ──► Critique ──► (Passed?) ──► END
                         │              ▲
                         └── (Failed) ──┘
```

---

# Phase 2: Tool Calling & The ReAct Pattern in JS

An LLM is a **brain in a jar**. It cannot query Postgres or check current stocks on its own.

### The ReAct Cycle in LangGraph.js:
```
                     START
                       │
                       ▼
                 ┌───────────┐
         ┌──────►│   agent   │  (Calls LLM, returns AIMessage with tool_calls)
         │       └─────┬─────┘
         │             │
         │     [toolsCondition]  (Did the LLM request tools?)
         │        /         \
         │   (has tools)   (no tools)
         │      /             \
         │     ▼               ▼
         └── ToolNode         END
          (Executes tool,
           produces ToolMessage)
```

### Defining Tools with Zod Validation:
```javascript
import { tool } from "@langchain/core/tools";
import { z } from "zod";

const weatherTool = tool(
  async ({ city }) => `Weather in ${city}: 22°C and sunny.`,
  {
    name: "get_weather",
    description: "Returns weather for a given city.",
    schema: z.object({
      city: z.string().describe("Target city name"),
    }),
  }
);
```

### Assembling the ReAct Graph:
```javascript
import { ToolNode, toolsCondition } from "@langchain/langgraph/prebuilt";

const graph = new StateGraph(MessagesAnnotation);
graph.addNode("agent", agentNode);
graph.addNode("tools", new ToolNode([weatherTool]));

graph.addEdge(START, "agent");
graph.addConditionalEdges("agent", toolsCondition);
graph.addEdge("tools", "agent"); // Loop back!

const app = graph.compile();
```

---

# Phase 3: Persistence & Memory (`MemorySaver`)

Without persistence, graph state is wiped clean as soon as an invocation finishes.

### Multi-Turn Sessions with `thread_id`:
```javascript
import { MemorySaver } from "@langchain/langgraph";

const checkpointer = new MemorySaver();
const app = graph.compile({ checkpointer });

// User Alice's isolated thread
const aliceSession = { configurable: { thread_id: "alice_session_1" } };

// Turn 1
await app.invoke(
  { messages: [new HumanMessage("Hello, my name is Alice.")] },
  aliceSession
);

// Turn 2: Only pass the new question! LangGraph automatically restores Turn 1!
const res = await app.invoke(
  { messages: [new HumanMessage("What is my name?")] },
  aliceSession
);
console.log(res.messages.at(-1).content); // "Your name is Alice!"
```

---

# Phase 4: Human-in-the-Loop & Breakpoints

Never let an autonomous AI agent execute real payments or drop database tables without a human sign-off.

### Static Breakpoints with `interruptBefore`:
```javascript
const app = graph.compile({
  checkpointer,
  interruptBefore: ["executeWireTransfer"],
});

const config = { configurable: { thread_id: "tx_99" } };

// 1. Run until breakpoint:
await app.invoke({ amount: 5000 }, config);

// 2. Inspect paused snapshot:
const snapshot = await app.getState(config);
console.log("Paused before:", snapshot.next); // ['executeWireTransfer']

// 3. Human in-flight edit:
await app.updateState(config, { amount: 3500 });

// 4. Resume execution by passing null:
const finalState = await app.invoke(null, config);
```

---

# Phase 5: Agentic RAG Patterns in JavaScript

### 1. Router RAG:
Inspects question intent before querying costly vector databases. Routes:
* Company policy $\to$ Vector Store
* Live world news $\to$ Web Search API
* General math $\to$ Direct LLM

### 2. Corrective RAG (CRAG):
* **Grade Documents:** An evaluation node checks if retrieved chunks are semantically relevant.
* **Query Rewrite & Fallback:** If internal docs are irrelevant, CRAG rewrites the query and triggers a web search fallback before synthesizing an answer.

---

# Phase 6: Multi-Agent Architectures

### 1. Supervisor Pattern (Hub & Spoke):
A central **Supervisor Agent** evaluates high-level goals and delegates sub-tasks to specialized workers (**Researcher**, **Coder**), aggregating their deliverables until the project is marked `"FINISH"`.

### 2. Swarm / Network Pattern (Peer-to-Peer Handoffs):
No central bottleneck. An incoming customer query reaches **Triage**, which hands off directly to **Billing**, which escalates directly to **Tech Support**.

---

# Python vs. JavaScript Syntax Rosetta Stone

| Concept | Python (`langgraph`) | JavaScript (`@langchain/langgraph`) |
| :--- | :--- | :--- |
| **State Schema** | `class State(TypedDict):` | `const State = Annotation.Root({ ... })` |
| **Chat State** | `from langgraph.graph import MessagesState` | `import { MessagesAnnotation } from "@langchain/langgraph"` |
| **Graph Creation** | `graph = StateGraph(State)` | `const graph = new StateGraph(State)` |
| **Add Node** | `graph.add_node("name", fn)` | `graph.addNode("name", fn)` |
| **Add Edge** | `graph.add_edge("a", "b")` | `graph.addEdge("a", "b")` |
| **Conditional Edge**| `graph.add_conditional_edges(src, fn, map)` | `graph.addConditionalEdges(src, fn, map)` |
| **Prebuilt Tools** | `from langgraph.prebuilt import ToolNode, tools_condition` | `import { ToolNode, toolsCondition } from "@langchain/langgraph/prebuilt"` |
| **Checkpointer** | `MemorySaver()` | `new MemorySaver()` |
| **Compile** | `app = graph.compile(checkpointer=...)` | `const app = graph.compile({ checkpointer })` |
| **Inspect State** | `app.get_state(config)` | `await app.getState(config)` |
| **Update State** | `app.update_state(config, { ... })` | `await app.updateState(config, { ... })` |
| **Invocation** | `app.invoke(input, config)` | `await app.invoke(input, config)` |

---

## How to Run the JavaScript Lessons

Navigate into the `langGraph_js` directory and install dependencies:

```bash
cd langGraph_js
npm install

# Run any lesson:
npm run lesson:01   # Basic StateGraph
npm run lesson:02   # Reducers & MessagesAnnotation
npm run lesson:03   # 4 Core Topologies
npm run lesson:04   # Custom ReAct Agent with ToolNode
npm run lesson:05   # MemorySaver & Thread Isolation
npm run lesson:06   # Human-in-the-Loop & Breakpoints
npm run lesson:07   # Corrective RAG (CRAG)
npm run lesson:08   # Supervisor Multi-Agent System
```
