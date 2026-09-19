/**
 * ============================================================================
 * LangGraph.js - Lesson 2: Reducers & Message Handling
 * ============================================================================
 * Key Concepts:
 * 1. Default vs. Reducer:
 *    By default, returning a key overwrites it. To append, you define a reducer:
 *    reducer: (existing, update) => existing.concat(update)
 * 2. MessagesAnnotation:
 *    Prebuilt Annotation provided by LangGraph.js that handles conversation
 *    histories, message IDs, and seamless message merging.
 * ============================================================================
 */

import { Annotation, StateGraph, MessagesAnnotation, START, END } from "@langchain/langgraph";
import { HumanMessage, AIMessage } from "@langchain/core/messages";

// --- Part A: Custom Reducer for Append-Only Audit Logs ---
const LogStateAnnotation = Annotation.Root({
  // Array reducer: concats new items instead of erasing previous ones
  activityLogs: Annotation({
    reducer: (current, update) => current.concat(update),
    default: () => [],
  }),
  currentStatus: Annotation({
    reducer: (current, update) => (update !== undefined ? update : current),
    default: () => "Pending",
  }),
});

function stepOneNode(state) {
  return {
    activityLogs: ["[Step 1]: Initialized user session."],
    currentStatus: "Step 1 Done",
  };
}

function stepTwoNode(state) {
  return {
    activityLogs: ["[Step 2]: Processed authentication payload."],
    currentStatus: "Step 2 Done",
  };
}

// --- Part B: Chat Messages with Prebuilt MessagesAnnotation ---
function assistantNode(state) {
  const lastMessage = state.messages[state.messages.length - 1];
  console.log(`\n[Assistant Node] Received prompt: '${lastMessage.content}'`);

  const reply = new AIMessage({
    content: `Acknowledged: "${lastMessage.content}". LangGraph.js maintains chat history!`,
  });

  return { messages: [reply] };
}

async function main() {
  console.log("=".repeat(65));
  console.log("  LangGraph.js Lesson 2: Reducers & MessagesAnnotation");
  console.log("=".repeat(65));

  // --- Run Part A: Custom Log Concatenation Reducer ---
  console.log("\n>>> PART A: Demonstrating concat reducer for logs");
  const graphA = new StateGraph(LogStateAnnotation);
  graphA.addNode("stepOne", stepOneNode);
  graphA.addNode("stepTwo", stepTwoNode);

  graphA.addEdge(START, "stepOne");
  graphA.addEdge("stepOne", "stepTwo");
  graphA.addEdge("stepTwo", END);

  const appA = graphA.compile();
  const resA = await appA.invoke({
    activityLogs: ["System Boot."],
    currentStatus: "Booting",
  });

  console.log("\nAccumulated Activity Logs (Appended by reducer):");
  resA.activityLogs.forEach((log) => console.log(`  * ${log}`));
  console.log(`Final Status: ${resA.currentStatus}`);

  // --- Run Part B: MessagesAnnotation for Chat Applications ---
  console.log("\n" + "-".repeat(65));
  console.log(">>> PART B: Demonstrating MessagesAnnotation");
  const graphB = new StateGraph(MessagesAnnotation);
  graphB.addNode("assistant", assistantNode);
  graphB.addEdge(START, "assistant");
  graphB.addEdge("assistant", END);

  const appB = graphB.compile();

  // Turn 1
  const turn1 = await appB.invoke({
    messages: [new HumanMessage("Hello! I am learning LangGraph in JavaScript.")],
  });

  console.log("\nConversation History after Turn 1:");
  turn1.messages.forEach((m) => console.log(`  [${m._getType()}]: ${m.content}`));

  // Turn 2
  const turn2 = await appB.invoke({
    messages: turn1.messages.concat([new HumanMessage("What are reducers?")]),
  });

  console.log("\nConversation History after Turn 2:");
  turn2.messages.forEach((m) => console.log(`  [${m._getType()}]: ${m.content}`));
}

main().catch(console.error);
