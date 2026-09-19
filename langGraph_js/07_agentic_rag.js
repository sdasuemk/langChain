/**
 * ============================================================================
 * LangGraph.js - Lesson 7: Agentic RAG Patterns (Router & Corrective RAG)
 * ============================================================================
 * Key Concepts:
 * 1. Router RAG: Directs questions to Vector Store, Web Search, or Direct LLM.
 * 2. Corrective RAG (CRAG): Evaluates retrieved doc relevance; triggers
 *    automatic query rewrite and fallback web search if docs are irrelevant.
 * ============================================================================
 */

import { Annotation, StateGraph, START, END } from "@langchain/langgraph";

const CRAGAnnotation = Annotation.Root({
  question: Annotation(),
  retrievedDocs: Annotation({
    reducer: (curr, update) => update ?? curr,
    default: () => [],
  }),
  isRelevant: Annotation({
    reducer: (curr, update) => update ?? curr,
    default: () => false,
  }),
  rewrittenQuery: Annotation({
    reducer: (curr, update) => update ?? curr,
    default: () => "",
  }),
  generation: Annotation({
    reducer: (curr, update) => update ?? curr,
    default: () => "",
  }),
});

function retrieveNode(state) {
  console.log(`\n[Step 1: Retrieve] Querying internal vector store for: '${state.question}'...`);
  if (state.question.toLowerCase().includes("quantum")) {
    // Simulate irrelevant document retrieval
    return { retrievedDocs: ["Ancient philosophy of particles."] };
  }
  return { retrievedDocs: ["LangGraph.js coordinates stateful multi-actor applications using graphs."] };
}

function gradeDocsNode(state) {
  console.log(`[Step 2: Grade Docs] Evaluating semantic relevance...`);
  const relevant = state.retrievedDocs.some((d) =>
    d.toLowerCase().includes("langgraph")
  );
  console.log(`  --> Relevance Grade: ${relevant ? "RELEVANT" : "IRRELEVANT"}`);
  return { isRelevant: relevant };
}

function rewriteQueryNode(state) {
  console.log(`[Step 3: Rewrite Query] Vector docs irrelevant. Rewriting for web search...`);
  const transformed = `${state.question} latest 2026 tech developments`;
  console.log(`  --> Refined Query: '${transformed}'`);
  return { rewrittenQuery: transformed };
}

function webSearchFallbackNode(state) {
  console.log(`[Step 4: Web Search] Fetching live web results for: '${state.rewrittenQuery}'...`);
  return {
    retrievedDocs: [`Live Web Result for ${state.rewrittenQuery}: Breakthrough in quantum scaling announced.`],
  };
}

function generateAnswerNode(state) {
  console.log(`[Step 5: Synthesize] Generating final grounded answer...`);
  const context = state.retrievedDocs.join("\n");
  return {
    generation: `Answer to "${state.question}":\nContext: ${context}`,
  };
}

async function main() {
  console.log("=".repeat(65));
  console.log("  LangGraph.js Lesson 7: Corrective RAG (CRAG)");
  console.log("=".repeat(65));

  const graph = new StateGraph(CRAGAnnotation);

  graph.addNode("retrieve", retrieveNode);
  graph.addNode("gradeDocs", gradeDocsNode);
  graph.addNode("rewriteQuery", rewriteQueryNode);
  graph.addNode("webSearch", webSearchFallbackNode);
  graph.addNode("generate", generateAnswerNode);

  graph.addEdge(START, "retrieve");
  graph.addEdge("retrieve", "gradeDocs");

  // Conditional routing after grading: if relevant -> generate, else -> rewriteQuery
  graph.addConditionalEdges(
    "gradeDocs",
    (state) => (state.isRelevant ? "generate" : "fallback"),
    {
      generate: "generate",
      fallback: "rewriteQuery",
    }
  );

  graph.addEdge("rewriteQuery", "webSearch");
  graph.addEdge("webSearch", "generate");
  graph.addEdge("generate", END);

  const app = graph.compile();

  // Test Case A: Matches internal docs
  console.log("\n>>> CASE A: Internal Knowledge Match");
  const resA = await app.invoke({ question: "What is LangGraph.js?" });
  console.log(resA.generation);

  // Test Case B: Irrelevant docs triggers CRAG rewrite & fallback
  console.log("\n>>> CASE B: Missing Internal Knowledge (Triggers Fallback)");
  const resB = await app.invoke({ question: "What is quantum computing?" });
  console.log(resB.generation);
}

main().catch(console.error);
