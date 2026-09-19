/**
 * ============================================================================
 * LangGraph.js - Lesson 3: The 4 Core Topologies in JavaScript
 * ============================================================================
 * 1. Serial (Pipeline): START -> clean -> count -> summarize -> END
 * 2. Parallel (Fan-out / Fan-in): START -> reader -> (worker1 & worker2) -> aggregate -> END
 * 3. Conditional: graph.addConditionalEdges(source, routerFn, pathMap)
 * 4. Loop (Cyclic): Refinement loop with recursionLimit protection
 * ============================================================================
 */

import { Annotation, StateGraph, START, END } from "@langchain/langgraph";

// ----------------------------------------------------------------------------
// 1. SERIAL GRAPH
// ----------------------------------------------------------------------------
const PipelineAnnotation = Annotation.Root({
  rawText: Annotation(),
  cleanedText: Annotation(),
  wordCount: Annotation(),
});

function cleanNode(state) {
  return { cleanedText: state.rawText.trim().replace(/\s+/g, " ") };
}

function countNode(state) {
  return { wordCount: state.cleanedText.split(" ").length };
}

function buildSerialGraph() {
  const graph = new StateGraph(PipelineAnnotation);
  graph.addNode("clean", cleanNode);
  graph.addNode("count", countNode);

  graph.addEdge(START, "clean");
  graph.addEdge("clean", "count");
  graph.addEdge("count", END);

  return graph.compile();
}

// ----------------------------------------------------------------------------
// 2. PARALLEL GRAPH (Fan-out / Fan-in)
// ----------------------------------------------------------------------------
const ParallelAnnotation = Annotation.Root({
  text: Annotation(),
  sentiment: Annotation(),
  keywords: Annotation(),
  auditLog: Annotation({
    reducer: (curr, update) => curr.concat(update),
    default: () => [],
  }),
});

function sentimentWorker(state) {
  const isPos = state.text.toLowerCase().includes("great");
  return {
    sentiment: isPos ? "Positive" : "Neutral",
    auditLog: ["Sentiment analysis complete."],
  };
}

function keywordWorker(state) {
  const words = state.text.split(" ").filter((w) => w.length > 5);
  return {
    keywords: words,
    auditLog: ["Keyword extraction complete."],
  };
}

function aggregatorNode(state) {
  return {
    auditLog: ["Aggregator combined sentiment and keywords successfully."],
  };
}

function buildParallelGraph() {
  const graph = new StateGraph(ParallelAnnotation);
  graph.addNode("sentiment", sentimentWorker);
  graph.addNode("keywords", keywordWorker);
  graph.addNode("aggregator", aggregatorNode);

  // Fan-out from START to both workers in parallel
  graph.addEdge(START, "sentiment");
  graph.addEdge(START, "keywords");

  // Fan-in: Aggregator waits for both
  graph.addEdge("sentiment", "aggregator");
  graph.addEdge("keywords", "aggregator");
  graph.addEdge("aggregator", END);

  return graph.compile();
}

// ----------------------------------------------------------------------------
// 3. CONDITIONAL ROUTING GRAPH
// ----------------------------------------------------------------------------
const TicketAnnotation = Annotation.Root({
  query: Annotation(),
  dept: Annotation(),
  response: Annotation(),
});

function triageNode(state) {
  const q = state.query.toLowerCase();
  const dept = q.includes("bill") || q.includes("refund") ? "billing" : "tech";
  return { dept };
}

function buildConditionalGraph() {
  const graph = new StateGraph(TicketAnnotation);
  graph.addNode("triage", triageNode);
  graph.addNode("billing", () => ({ response: "Billing: Refund issued." }));
  graph.addNode("tech", () => ({ response: "Tech Support: Ticket opened." }));

  graph.addEdge(START, "triage");

  // Conditional edge: evaluates state.dept to select next node
  graph.addConditionalEdges("triage", (state) => state.dept, {
    billing: "billing",
    tech: "tech",
  });

  graph.addEdge("billing", END);
  graph.addEdge("tech", END);

  return graph.compile();
}

// ----------------------------------------------------------------------------
// 4. CYCLIC LOOP GRAPH
// ----------------------------------------------------------------------------
const LoopAnnotation = Annotation.Root({
  iteration: Annotation({
    reducer: (c, u) => u ?? c,
    default: () => 0,
  }),
  qualityPassed: Annotation({
    reducer: (c, u) => u ?? c,
    default: () => false,
  }),
});

function draftNode(state) {
  const nextIter = state.iteration + 1;
  console.log(`  [Loop Draft] Iteration #${nextIter}...`);
  return {
    iteration: nextIter,
    qualityPassed: nextIter >= 3, // Passes on 3rd attempt
  };
}

function buildLoopGraph() {
  const graph = new StateGraph(LoopAnnotation);
  graph.addNode("draft", draftNode);

  graph.addEdge(START, "draft");

  // If quality passed -> END, else loop back to "draft"
  graph.addConditionalEdges("draft", (state) => (state.qualityPassed ? "finish" : "retry"), {
    retry: "draft",
    finish: END,
  });

  return graph.compile();
}

async function main() {
  console.log("=".repeat(65));
  console.log("  LangGraph.js Lesson 3: 4 Core Topologies in Action");
  console.log("=".repeat(65));

  // Test Serial
  console.log("\n>>> 1. Testing Serial Graph:");
  const serialApp = buildSerialGraph();
  const serialRes = await serialApp.invoke({ rawText: "   Hello   LangGraph    JavaScript   " });
  console.log("Cleaned:", serialRes.cleanedText, "| Words:", serialRes.wordCount);

  // Test Parallel
  console.log("\n>>> 2. Testing Parallel Fan-Out / Fan-In:");
  const parallelApp = buildParallelGraph();
  const parallelRes = await parallelApp.invoke({ text: "This is a great product with awesome capability." });
  console.log("Sentiment:", parallelRes.sentiment);
  console.log("Keywords :", parallelRes.keywords);
  console.log("Logs     :", parallelRes.auditLog);

  // Test Conditional
  console.log("\n>>> 3. Testing Conditional Router:");
  const condApp = buildConditionalGraph();
  const condRes = await condApp.invoke({ query: "I need a refund for my bill." });
  console.log("Response :", condRes.response);

  // Test Loop
  console.log("\n>>> 4. Testing Cyclic Refinement Loop:");
  const loopApp = buildLoopGraph();
  const loopRes = await loopApp.invoke({}, { recursionLimit: 10 });
  console.log("Total Iterations Completed:", loopRes.iteration);
}

main().catch(console.error);
