/**
 * ============================================================================
 * LangGraph.js - Lesson 4: Tool Calling & The ReAct Pattern
 * ============================================================================
 * Key Concepts:
 * 1. Defining Tools using `tool()` from @langchain/core/tools with Zod schemas.
 * 2. `ToolNode`: Prebuilt node executing tool calls in the latest AIMessage.
 * 3. `toolsCondition`: Built-in router directing to "tools" or END.
 * 4. Cyclic ReAct Loop: `tools -> agent` sends execution results back to model.
 * ============================================================================
 */

import { StateGraph, MessagesAnnotation, START, END } from "@langchain/langgraph";
import { ToolNode, toolsCondition } from "@langchain/langgraph/prebuilt";
import { tool } from "@langchain/core/tools";
import { AIMessage, HumanMessage } from "@langchain/core/messages";
import { z } from "zod";

// --- 1. Define Tools using Zod Schema Validation ---
const addNumbersTool = tool(
  async ({ a, b }) => {
    console.log(`\n  [Tool: addNumbers] Adding ${a} + ${b}...`);
    return String(a + b);
  },
  {
    name: "add_numbers",
    description: "Adds two numbers together.",
    schema: z.object({
      a: z.number().describe("First number"),
      b: z.number().describe("Second number"),
    }),
  }
);

const stockLookupTool = tool(
  async ({ ticker }) => {
    console.log(`\n  [Tool: stockLookup] Checking price for ${ticker.toUpperCase()}...`);
    const prices = { AAPL: "$185.50", NVDA: "$125.80", GOOG: "$175.20" };
    return prices[ticker.toUpperCase()] || "Ticker not found.";
  },
  {
    name: "get_stock_price",
    description: "Fetches current stock price for a given ticker.",
    schema: z.object({
      ticker: z.string().describe("Stock ticker symbol"),
    }),
  }
);

const tools = [addNumbersTool, stockLookupTool];

// --- 2. Simulated LLM Node (Standalone runnable without API keys) ---
function agentNode(state) {
  const lastMsg = state.messages[state.messages.length - 1];
  console.log(`\n[Agent Node] Inspecting message: '${lastMsg.content}'`);

  // If last message was a Tool response, synthesize final answer
  if (lastMsg._getType() === "tool") {
    return {
      messages: [
        new AIMessage(`Based on the tool output ("${lastMsg.content}"), here is your answer!`),
      ],
    };
  }

  const query = String(lastMsg.content).toLowerCase();

  // Simulate LLM producing tool calls
  if (query.includes("nvda") || query.includes("stock")) {
    return {
      messages: [
        new AIMessage({
          content: "",
          tool_calls: [
            {
              name: "get_stock_price",
              args: { ticker: "NVDA" },
              id: "call_stock_101",
            },
          ],
        }),
      ],
    };
  }

  return {
    messages: [new AIMessage("I can look up stocks and calculate numbers using tools.")],
  };
}

async function main() {
  console.log("=".repeat(65));
  console.log("  LangGraph.js Lesson 4: Custom ReAct Agent with ToolNode");
  console.log("=".repeat(65));

  // --- 3. Assemble ReAct Graph ---
  const graph = new StateGraph(MessagesAnnotation);

  graph.addNode("agent", agentNode);
  graph.addNode("tools", new ToolNode(tools));

  graph.addEdge(START, "agent");

  // Conditional Edge: If tool_calls exist -> "tools", else -> END
  graph.addConditionalEdges("agent", toolsCondition);

  // Return cycle: tools -> agent
  graph.addEdge("tools", "agent");

  const app = graph.compile();

  // --- 4. Invoke ReAct Agent ---
  const query = "What is the stock price of NVDA?";
  console.log(`User Query: '${query}'`);

  const finalState = await app.invoke({
    messages: [new HumanMessage(query)],
  });

  console.log("\n" + "=".repeat(65));
  console.log("ReAct Loop Completed! Full Message Stack:");
  console.log("=".repeat(65));
  finalState.messages.forEach((msg) => {
    if (msg.tool_calls && msg.tool_calls.length > 0) {
      console.log(`[AIMessage (Tool Call)]:`, msg.tool_calls);
    } else {
      console.log(`[${msg._getType()}]:`, msg.content);
    }
  });
}

main().catch(console.error);
