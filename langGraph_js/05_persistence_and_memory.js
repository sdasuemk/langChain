/**
 * ============================================================================
 * LangGraph.js - Lesson 5: Persistence & Memory (MemorySaver & thread_id)
 * ============================================================================
 * Key Concepts:
 * 1. Checkpointer: Saves graph state after every step.
 * 2. `thread_id`: Partitions conversations into isolated sessions.
 * 3. `app.getState(config)`: Reads current state snapshot and next node queue.
 * ============================================================================
 */

import { StateGraph, MessagesAnnotation, MemorySaver, START, END } from "@langchain/langgraph";
import { HumanMessage, AIMessage } from "@langchain/core/messages";

function assistantNode(state) {
  const history = state.messages;
  const lastMsg = history[history.length - 1].content;
  console.log(`\n  [Assistant Node] Current Query: '${lastMsg}'`);
  console.log(`  [Assistant Node] Total turns in memory: ${history.length}`);

  const pastText = history.slice(0, -1).map((m) => m.content).join(" ");

  if (lastMsg.toLowerCase().includes("my name is")) {
    const name = lastMsg.split("name is")[1].trim().replace(".", "");
    return { messages: [new AIMessage(`Nice to meet you, ${name}! I've saved your name.`)] };
  } else if (lastMsg.toLowerCase().includes("what is my name")) {
    if (pastText.includes("Alice")) {
      return { messages: [new AIMessage("Your name is Alice! I remember from earlier.")] };
    }
    return { messages: [new AIMessage("I do not know your name yet in this session.")] };
  }

  return { messages: [new AIMessage(`Received: '${lastMsg}'. How can I assist?`)] };
}

async function main() {
  console.log("=".repeat(65));
  console.log("  LangGraph.js Lesson 5: In-Memory Persistence & Checkpoints");
  console.log("=".repeat(65));

  const graph = new StateGraph(MessagesAnnotation);
  graph.addNode("assistant", assistantNode);
  graph.addEdge(START, "assistant");
  graph.addEdge("assistant", END);

  // Initialize Checkpointer
  const checkpointer = new MemorySaver();
  const app = graph.compile({ checkpointer });

  // --- Session 1: Alice ---
  console.log("\n" + "=".repeat(55));
  console.log(">>> THREAD 1: Alice's Conversation");
  console.log("=".repeat(55));

  const aliceConfig = { configurable: { thread_id: "alice_session_101" } };

  console.log("\nTurn 1: Alice introduces herself:");
  const turn1 = await app.invoke(
    { messages: [new HumanMessage("Hello, my name is Alice.")] },
    aliceConfig
  );
  console.log("AI:", turn1.messages[turn1.messages.length - 1].content);

  console.log("\nTurn 2: Alice asks for her name (only sending new query):");
  const turn2 = await app.invoke(
    { messages: [new HumanMessage("What is my name?")] },
    aliceConfig
  );
  console.log("AI:", turn2.messages[turn2.messages.length - 1].content);

  // --- Session 2: Bob (Verifying Isolation) ---
  console.log("\n" + "=".repeat(55));
  console.log(">>> THREAD 2: Bob's Conversation (Isolated)");
  console.log("=".repeat(55));

  const bobConfig = { configurable: { thread_id: "bob_session_202" } };

  console.log("\nTurn 1: Bob asks if AI knows him:");
  const bobTurn1 = await app.invoke(
    { messages: [new HumanMessage("What is my name?")] },
    bobConfig
  );
  console.log("AI (to Bob):", bobTurn1.messages[bobTurn1.messages.length - 1].content);

  // --- State Inspection with getState ---
  console.log("\n" + "=".repeat(55));
  console.log(">>> Inspecting State Snapshot with app.getState()");
  console.log("=".repeat(55));

  const snapshot = await app.getState(aliceConfig);
  console.log("Alice's Checkpoint ID:", snapshot.config.configurable.checkpoint_id);
  console.log("Total Messages Stored :", snapshot.values.messages.length);
}

main().catch(console.error);
