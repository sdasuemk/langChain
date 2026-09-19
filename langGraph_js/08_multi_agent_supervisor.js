/**
 * ============================================================================
 * LangGraph.js - Lesson 8: Supervisor Multi-Agent System
 * ============================================================================
 * Key Concepts:
 * 1. Centralized Supervisor: Directs high-level workflow and delegates tasks.
 * 2. Specialized Workers: Researcher and Coder handle domain-specific execution.
 * 3. Handoff Loop: Workers report back to supervisor until "FINISH" is chosen.
 * ============================================================================
 */

import { Annotation, StateGraph, START, END } from "@langchain/langgraph";

const TeamAnnotation = Annotation.Root({
  projectGoal: Annotation(),
  deliverables: Annotation({
    reducer: (curr, update) => curr.concat(update),
    default: () => [],
  }),
  nextWorker: Annotation({
    reducer: (curr, update) => update ?? curr,
    default: () => "supervisor",
  }),
  iteration: Annotation({
    reducer: (curr, update) => (update !== undefined ? update : curr),
    default: () => 0,
  }),
});

function supervisorNode(state) {
  const currentIter = state.iteration + 1;
  console.log(`\n[Supervisor] Reviewing project deliverables (Cycle #${currentIter})...`);

  const hasResearch = state.deliverables.some((d) => d.includes("Research"));
  const hasCode = state.deliverables.some((d) => d.includes("Code"));

  let next = "FINISH";
  if (!hasResearch) {
    next = "researcher";
    console.log("  --> Decision: Task Researcher to gather specifications.");
  } else if (!hasCode) {
    next = "coder";
    console.log("  --> Decision: Task Coder to implement module.");
  } else {
    console.log("  --> Decision: All deliverables satisfied! Concluding project.");
  }

  return {
    nextWorker: next,
    iteration: currentIter,
  };
}

function researcherNode(state) {
  console.log(`  [Worker: Researcher] Researching requirements for: '${state.projectGoal}'...`);
  return {
    deliverables: ["Research Report: Redis distributed lock algorithm confirmed."],
  };
}

function coderNode(state) {
  console.log(`  [Worker: Coder] Writing JavaScript implementation...`);
  return {
    deliverables: ["Code Deliverable: const lock = new Redlock([redisClient]);"],
  };
}

async function main() {
  console.log("=".repeat(65));
  console.log("  LangGraph.js Lesson 8: Supervisor Multi-Agent Architecture");
  console.log("=".repeat(65));

  const graph = new StateGraph(TeamAnnotation);

  graph.addNode("supervisor", supervisorNode);
  graph.addNode("researcher", researcherNode);
  graph.addNode("coder", coderNode);

  graph.addEdge(START, "supervisor");

  // Supervisor routes to workers or finishes
  graph.addConditionalEdges("supervisor", (state) => state.nextWorker, {
    researcher: "researcher",
    coder: "coder",
    FINISH: END,
  });

  // Workers always return back to the supervisor
  graph.addEdge("researcher", "supervisor");
  graph.addEdge("coder", "supervisor");

  const app = graph.compile();

  const finalState = await app.invoke({
    projectGoal: "Implement distributed concurrency lock in Node.js",
  });

  console.log("\n" + "=".repeat(65));
  console.log("Multi-Agent Project Completed! Deliverables Stack:");
  console.log("=".repeat(65));
  finalState.deliverables.forEach((item, idx) => {
    console.log(`Deliverable ${idx + 1}: ${item}`);
  });
}

main().catch(console.error);
