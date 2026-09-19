/**
 * ============================================================================
 * LangGraph.js - Lesson 1: Basic StateGraph (Pure JavaScript)
 * ============================================================================
 * Key Concepts:
 * 1. Annotation.Root: Defines the shape and reducer rules of the state container.
 * 2. Nodes: Regular JavaScript functions (sync or async) that take state and return updates.
 * 3. Edges: Connect nodes together (START -> Node -> END).
 * 4. Compilation: graph.compile() returns an executable runnable app.
 * ============================================================================
 */

import { Annotation, StateGraph, START, END } from "@langchain/langgraph";

// --- 1. Define State Schema using Annotation.Root ---
const SimpleStateAnnotation = Annotation.Root({
  userName: Annotation(),
  greeting: Annotation(),
  stepCount: Annotation({
    reducer: (current, update) => (update !== undefined ? update : current),
    default: () => 0,
  }),
});

// --- 2. Define Node Functions ---
function welcomeNode(state) {
  console.log(`\n[Node: welcomeNode] Processing user: '${state.userName}'`);
  return {
    greeting: `Hello, ${state.userName}! Welcome to LangGraph.js.`,
    stepCount: state.stepCount + 1,
  };
}

function uppercaseNode(state) {
  console.log(`[Node: uppercaseNode] Transforming greeting to UPPERCASE...`);
  return {
    greeting: state.greeting.toUpperCase(),
    stepCount: state.stepCount + 1,
  };
}

async function main() {
  console.log("=".repeat(65));
  console.log("  LangGraph.js Lesson 1: Basic StateGraph Flow (Pure JS)");
  console.log("=".repeat(65));

  // --- 3. Assemble the Graph ---
  const graph = new StateGraph(SimpleStateAnnotation);

  // Add nodes: (name, function)
  graph.addNode("welcome", welcomeNode);
  graph.addNode("uppercase", uppercaseNode);

  // Connect edges: START -> welcome -> uppercase -> END
  graph.addEdge(START, "welcome");
  graph.addEdge("welcome", "uppercase");
  graph.addEdge("uppercase", END);

  // --- 4. Compile the Graph ---
  const app = graph.compile();

  // --- 5. Invoke the Graph ---
  const initialInput = {
    userName: "Alex",
    greeting: "",
    stepCount: 0,
  };

  console.log(`\nInitial Input:`, initialInput);
  const finalState = await app.invoke(initialInput);

  console.log(`\n--- Execution Complete ---`);
  console.log(`Final Result State:`, finalState);
}

main().catch(console.error);
