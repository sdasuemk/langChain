/**
 * ============================================================================
 * LangGraph.js - Lesson 6: Human-in-the-Loop (HITL) & Breakpoints
 * ============================================================================
 * Key Concepts:
 * 1. `interruptBefore`: Pauses execution right before a sensitive node runs.
 * 2. State Inspection: `app.getState(config).next` shows which node is paused.
 * 3. Human In-Flight Edits: Modifying state with `app.updateState()`.
 * 4. Resumption: Passing `null` as input to continue from the breakpoint.
 * ============================================================================
 */

import { Annotation, StateGraph, MemorySaver, START, END } from "@langchain/langgraph";

const PaymentAnnotation = Annotation.Root({
  sender: Annotation(),
  recipient: Annotation(),
  amount: Annotation({
    reducer: (curr, update) => update ?? curr,
    default: () => 0,
  }),
  status: Annotation({
    reducer: (curr, update) => update ?? curr,
    default: () => "Pending",
  }),
  confirmationId: Annotation({
    reducer: (curr, update) => update ?? curr,
    default: () => "",
  }),
});

function prepareTransferNode(state) {
  console.log(`\n[Node: prepareTransfer] Validating $${state.amount} transfer to ${state.recipient}...`);
  return { status: "Pending Human Approval" };
}

function executeTransferNode(state) {
  console.log(`\n[Node: executeTransfer] EXECUTING CRITICAL WIRE: Sending $${state.amount} to ${state.recipient}!`);
  return {
    status: "Transferred Successfully",
    confirmationId: "TX-JS-9921",
  };
}

async function main() {
  console.log("=".repeat(65));
  console.log("  LangGraph.js Lesson 6: Human-in-the-Loop & Breakpoints");
  console.log("=".repeat(65));

  const graph = new StateGraph(PaymentAnnotation);
  graph.addNode("prepareTransfer", prepareTransferNode);
  graph.addNode("executeTransfer", executeTransferNode);

  graph.addEdge(START, "prepareTransfer");
  graph.addEdge("prepareTransfer", "executeTransfer");
  graph.addEdge("executeTransfer", END);

  const checkpointer = new MemorySaver();

  // Compile with static breakpoint BEFORE executeTransfer
  const app = graph.compile({
    checkpointer,
    interruptBefore: ["executeTransfer"],
  });

  const threadConfig = { configurable: { thread_id: "payment_thread_001" } };

  // --- Stage 1: Run until Breakpoint ---
  console.log("\n--- STAGE 1: Initiating Transfer (Runs until Breakpoint) ---");
  await app.invoke(
    {
      sender: "Alice",
      recipient: "Bob",
      amount: 5000,
    },
    threadConfig
  );

  // --- Stage 2: Inspect Paused State ---
  console.log("\n--- STAGE 2: Inspecting Paused State ---");
  const snapshot = await app.getState(threadConfig);
  console.log("Scheduled Next Node:", snapshot.next); // ['executeTransfer']
  console.log("Proposed Amount     :", snapshot.values.amount);

  // --- Stage 3: Human Intervention (Edit Amount before Approving) ---
  console.log("\n>>> SUPERVISOR INTERVENTION <<<");
  console.log("Supervisor: 'Reducing transfer limit from $5,000 to $3,500.'");

  await app.updateState(threadConfig, {
    amount: 3500,
  });

  // --- Stage 4: Resume Execution ---
  console.log("\n--- STAGE 3: Resuming Execution (Passing null) ---");
  const finalState = await app.invoke(null, threadConfig);

  console.log("\n--- Execution Completed ---");
  console.log("Final Status        :", finalState.status);
  console.log("Final Amount Sent   :", finalState.amount);
  console.log("Confirmation ID     :", finalState.confirmationId);
}

main().catch(console.error);
