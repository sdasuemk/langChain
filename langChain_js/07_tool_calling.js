/**
 * ============================================================================
 * LangChain.js - Lesson 7: Tool Calling & bindTools
 * ============================================================================
 * Key Concepts:
 * 1. Defining Tools: Using `tool()` from @langchain/core/tools with Zod schemas.
 * 2. bindTools(): Binds schema definitions to Chat Models so the model can emit
 *    structured tool invocation requests.
 * 3. Tool Execution Loop: Extracting `tool_calls`, executing the matching function,
 *    and passing back a `ToolMessage`.
 * ============================================================================
 */

import { tool } from "@langchain/core/tools";
import { HumanMessage, AIMessage, ToolMessage } from "@langchain/core/messages";
import { z } from "zod";

// --- 1. Define Tools with Zod Schemas ---
const calculateTaxTool = tool(
  async ({ amount, taxRate }) => {
    console.log(`\n  [Tool Execution: calculate_tax] Calculating tax on $${amount} at ${taxRate * 100}%...`);
    const tax = amount * taxRate;
    return String(tax.toFixed(2));
  },
  {
    name: "calculate_tax",
    description: "Calculates total sales tax given an amount and a tax rate percentage.",
    schema: z.object({
      amount: z.number().describe("Subtotal dollar amount"),
      taxRate: z.number().describe("Tax rate expressed as decimal e.g. 0.08 for 8%"),
    }),
  }
);

const tools = [calculateTaxTool];
const toolsMap = { calculate_tax: calculateTaxTool };

// --- 2. Simulated Chat Model with bindTools capability ---
class SimulatedChatModelWithTools {
  bindTools(toolsList) {
    this.boundTools = toolsList;
    return this;
  }

  async invoke(messages) {
    const lastMsg = messages[messages.length - 1];

    // Stage 2: Model receives ToolMessage and delivers final human response
    if (lastMsg._getType() === "tool") {
      return new AIMessage(
        `Based on the calculation tool, the total sales tax is $${lastMsg.content}.`
      );
    }

    // Stage 1: Initial query generates tool call
    return new AIMessage({
      content: "",
      tool_calls: [
        {
          name: "calculate_tax",
          args: { amount: 250, taxRate: 0.085 },
          id: "call_tax_99",
        },
      ],
    });
  }
}

async function main() {
  console.log("=".repeat(65));
  console.log("  LangChain.js Lesson 7: Structured Tool Calling & bindTools");
  console.log("=".repeat(65));

  const model = new SimulatedChatModelWithTools().bindTools(tools);

  const query = "What is the sales tax on a $250 purchase at an 8.5% tax rate?";
  console.log(`User Query: "${query}"`);

  const conversationHistory = [new HumanMessage(query)];

  // --- Step 1: Query Model to Generate Tool Call ---
  console.log("\n--- STAGE 1: Sending Query to Model ---");
  const response1 = await model.invoke(conversationHistory);
  conversationHistory.push(response1);

  console.log("Model Response:", response1.tool_calls ? "Generated Tool Calls!" : "Direct Response");
  console.log("Tool Calls:", response1.tool_calls);

  // --- Step 2: Execute Requested Tools ---
  if (response1.tool_calls && response1.tool_calls.length > 0) {
    console.log("\n--- STAGE 2: Executing Tools Locally ---");

    for (const toolCall of response1.tool_calls) {
      const selectedTool = toolsMap[toolCall.name];
      const resultText = await selectedTool.invoke(toolCall.args);

      // Package result as a ToolMessage referencing toolCall.id
      const toolMessage = new ToolMessage({
        content: resultText,
        tool_call_id: toolCall.id,
      });

      conversationHistory.push(toolMessage);
    }

    // --- Step 3: Call Model with Tool Return to Generate Final Answer ---
    console.log("\n--- STAGE 3: Feeding Tool Results Back to Model ---");
    const finalAnswer = await model.invoke(conversationHistory);

    console.log("\n" + "=".repeat(65));
    console.log("Final Synthesized Answer:");
    console.log("=".repeat(65));
    console.log(finalAnswer.content);
  }
}

main().catch(console.error);
