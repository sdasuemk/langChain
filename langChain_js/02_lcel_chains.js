/**
 * ============================================================================
 * LangChain.js - Lesson 2: LCEL (LangChain Expression Language) Chains
 * ============================================================================
 * Key Concepts:
 * 1. The .pipe() method: Links Runnables together (Prompt -> Model -> Parser).
 * 2. RunnableSequence: Composes multiple steps into a single executable pipeline.
 * 3. RunnablePassthrough: Passes original input through untouched for parallel steps.
 * ============================================================================
 */

import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { RunnableSequence, RunnablePassthrough } from "@langchain/core/runnables";
import { AIMessage } from "@langchain/core/messages";

// --- 1. Define Simulated Model ---
class SimulatedLLM {
  async invoke(messages) {
    const promptText = messages[messages.length - 1].content;
    if (promptText.includes("rhyme")) {
      return new AIMessage("Coding with delight, under starry night.");
    }
    return new AIMessage("JavaScript pipelines make AI orchestration effortless.");
  }
}

async function main() {
  console.log("=".repeat(65));
  console.log("  LangChain.js Lesson 2: LCEL Chains (.pipe() & Sequences)");
  console.log("=".repeat(65));

  const prompt = ChatPromptTemplate.fromTemplate(
    "Write a short {style} about {topic}."
  );
  const model = new SimulatedLLM();
  const outputParser = new StringOutputParser();

  // --- 2. Build Pipeline using .pipe() ---
  // In Python: chain = prompt | model | outputParser
  // In JavaScript: chain = prompt.pipe(model).pipe(outputParser)
  const pipeChain = prompt.pipe(model).pipe(outputParser);

  console.log("\n>>> 1. Invoking Chain via .pipe():");
  const result1 = await pipeChain.invoke({
    style: "rhyme",
    topic: "software architecture",
  });
  console.log("Result:", result1);

  // --- 3. Alternative: RunnableSequence.from([...]) ---
  const sequenceChain = RunnableSequence.from([
    prompt,
    model,
    outputParser,
  ]);

  console.log("\n>>> 2. Invoking Chain via RunnableSequence.from():");
  const result2 = await sequenceChain.invoke({
    style: "summary",
    topic: "concurrency",
  });
  console.log("Result:", result2);

  // --- 4. RunnablePassthrough (Preserving Inputs) ---
  const passthroughChain = RunnableSequence.from([
    {
      originalQuery: new RunnablePassthrough(),
      style: () => "technical summary",
      topic: (input) => input.text,
    },
    prompt,
    model,
    outputParser,
  ]);

  console.log("\n>>> 3. Invoking Chain with RunnablePassthrough transformation:");
  const result3 = await passthroughChain.invoke({ text: "microservices" });
  console.log("Result:", result3);
}

main().catch(console.error);
