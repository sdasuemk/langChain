/**
 * ============================================================================
 * LangChain.js - Lesson 3: Output Parsers (String, JSON, Structured Zod)
 * ============================================================================
 * Key Concepts:
 * 1. StringOutputParser: Extracts text string directly from AIMessage.
 * 2. JsonOutputParser: Parses raw JSON strings into JavaScript objects.
 * 3. StructuredOutputParser with Zod: Enforces TypeScript/Zod typed schemas
 *    with automatic formatting instructions for the prompt.
 * ============================================================================
 */

import { StringOutputParser, JsonOutputParser } from "@langchain/core/output_parsers";
import { StructuredOutputParser } from "@langchain/core/output_parsers";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { AIMessage } from "@langchain/core/messages";
import { z } from "zod";

async function main() {
  console.log("=".repeat(65));
  console.log("  LangChain.js Lesson 3: Output Parsers");
  console.log("=".repeat(65));

  // --- 1. StringOutputParser ---
  console.log("\n>>> 1. StringOutputParser:");
  const stringParser = new StringOutputParser();
  const rawAiMessage = new AIMessage("Clean extracted response content.");
  const parsedString = await stringParser.invoke(rawAiMessage);
  console.log("Extracted String:", parsedString);

  // --- 2. JsonOutputParser ---
  console.log("\n>>> 2. JsonOutputParser:");
  const jsonParser = new JsonOutputParser();
  const rawJsonMsg = new AIMessage('{"status": "success", "userId": 42, "role": "admin"}');
  const parsedJson = await jsonParser.invoke(rawJsonMsg);
  console.log("Parsed JS Object:", parsedJson);
  console.log(`User ID: ${parsedJson.userId}, Role: ${parsedJson.role}`);

  // --- 3. StructuredOutputParser with Zod ---
  console.log("\n>>> 3. StructuredOutputParser with Zod Schema:");

  // Define strict Zod schema
  const userFeedbackSchema = z.object({
    sentiment: z.enum(["positive", "neutral", "negative"]).describe("Overall sentiment"),
    confidence: z.number().min(0).max(1).describe("Confidence score between 0 and 1"),
    keyPoints: z.array(z.string()).describe("Key takeaways mentioned"),
  });

  const structuredParser = StructuredOutputParser.fromZodSchema(userFeedbackSchema);

  // Auto-generate instructions for the LLM prompt
  const formatInstructions = structuredParser.getFormatInstructions();
  console.log("Generated Prompt Format Instructions:");
  console.log(formatInstructions);

  // Simulated LLM adhering to format instructions
  const simulatedLlmOutput = new AIMessage(
    JSON.stringify({
      sentiment: "positive",
      confidence: 0.95,
      keyPoints: ["Fast load times", "Intuitive dark mode UI"],
    })
  );

  const validatedResult = await structuredParser.invoke(simulatedLlmOutput);
  console.log("\nValidated Typed Object Result:");
  console.log(validatedResult);
  console.log(`Sentiment: ${validatedResult.sentiment.toUpperCase()}`);
}

main().catch(console.error);
