/**
 * ============================================================================
 * LangChain.js - Lesson 6: Retrievers & RAG (Retrieval-Augmented Generation)
 * ============================================================================
 * Key Concepts:
 * 1. asRetriever(): Converts any vector store into a standard Runnable.
 * 2. LCEL RAG Pipeline:
 *    { context: retriever.pipe(formatDocs), question: new RunnablePassthrough() }
 *    -> prompt -> model -> StringOutputParser
 * 3. Grounding LLM generations in authoritative retrieved documentation.
 * ============================================================================
 */

import { Document } from "@langchain/core/documents";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { RunnableSequence, RunnablePassthrough } from "@langchain/core/runnables";
import { AIMessage } from "@langchain/core/messages";

// --- 1. Deterministic Standalone Embeddings & Simulated LLM ---
class SimulatedEmbeddings {
  async embedDocuments(docs) {
    return docs.map((d) => [d.toLowerCase().includes("policy") ? 1.0 : 0.2, 0.5]);
  }
  async embedQuery(q) {
    return [q.toLowerCase().includes("policy") || q.toLowerCase().includes("pto") ? 1.0 : 0.2, 0.5];
  }
}

class SimulatedRAGChatModel {
  async invoke(messages) {
    const fullPrompt = messages[messages.length - 1].content;
    console.log("\n[LLM Received Prompt with Context Augmented]:");
    console.log(fullPrompt);

    return new AIMessage(
      "Based on the company HR policy, full-time employees accrue 20 days of paid time off (PTO) annually."
    );
  }
}

async function main() {
  console.log("=".repeat(65));
  console.log("  LangChain.js Lesson 6: End-to-End RAG with LCEL");
  console.log("=".repeat(65));

  // --- 2. Seed Vector Store with Knowledge Base ---
  const hrHandbook = [
    new Document({
      pageContent: "Company Policy Section 4: Full-time employees accrue 20 days of paid time off (PTO) annually.",
      metadata: { doc: "handbook.pdf", page: 12 },
    }),
    new Document({
      pageContent: "Company Policy Section 5: Unused PTO rolls over up to a maximum of 5 calendar days.",
      metadata: { doc: "handbook.pdf", page: 13 },
    }),
    new Document({
      pageContent: "Office Security: Badge access is mandatory between 7pm and 6am on weekdays.",
      metadata: { doc: "security.pdf", page: 2 },
    }),
  ];

  const embeddings = new SimulatedEmbeddings();
  const vectorStore = await MemoryVectorStore.fromDocuments(hrHandbook, embeddings);

  // --- 3. Convert Vector Store into a Runnable Retriever ---
  const retriever = vectorStore.asRetriever(2);

  // --- 4. Define Context Formatter ---
  const formatDocs = (docs) => docs.map((d) => `[Source: ${d.metadata.doc}]: ${d.pageContent}`).join("\n");

  // --- 5. Define RAG Prompt Template ---
  const ragPrompt = ChatPromptTemplate.fromMessages([
    [
      "system",
      `You are an authoritative enterprise HR assistant.
Answer the question strictly using the retrieved context provided below.
If the context does not contain the answer, reply "I do not have information on that."

Context:
{context}`,
    ],
    ["user", "{question}"],
  ]);

  const model = new SimulatedRAGChatModel();
  const outputParser = new StringOutputParser();

  // --- 6. Assemble LCEL RAG Pipeline ---
  const ragChain = RunnableSequence.from([
    {
      context: retriever.pipe(formatDocs),
      question: new RunnablePassthrough(),
    },
    ragPrompt,
    model,
    outputParser,
  ]);

  // --- 7. Execute Query ---
  const query = "What is our company policy on annual PTO days?";
  console.log(`\nUser Question: "${query}"\n`);

  const answer = await ragChain.invoke(query);

  console.log("\n" + "=".repeat(65));
  console.log("Final Generated RAG Answer:");
  console.log("=".repeat(65));
  console.log(answer);
}

main().catch(console.error);
