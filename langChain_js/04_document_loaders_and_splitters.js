/**
 * ============================================================================
 * LangChain.js - Lesson 4: Documents & Text Splitters
 * ============================================================================
 * Key Concepts:
 * 1. Document: The core container holding `pageContent` and `metadata`.
 * 2. RecursiveCharacterTextSplitter: Splits large text chunks intelligently
 *    using paragraph, newline, and space delimiters while preserving context.
 * 3. Chunk Overlap: Ensures continuity between adjacent split fragments.
 * ============================================================================
 */

import { Document } from "@langchain/core/documents";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";

async function main() {
  console.log("=".repeat(65));
  console.log("  LangChain.js Lesson 4: Document Chunking & Text Splitters");
  console.log("=".repeat(65));

  // --- 1. Creating Raw Documents ---
  const rawDocument = new Document({
    pageContent: `LangChain is a framework for developing applications powered by large language models (LLMs).
It simplifies the entire application lifecycle:
- Development: Build using building blocks like prompts, models, and retrievers.
- Productionization: Inspect and monitor chains with LangSmith.
- Deployment: Deploy chains with LangServe and state machines with LangGraph.

LangGraph is a library within LangChain specifically designed for building stateful, multi-actor applications with LLMs.
It treats workflows as cyclic state graphs instead of simple linear chains.`,
    metadata: { source: "langchain_overview.md", author: "LangChain Team", year: 2026 },
  });

  console.log("\nOriginal Document Length:", rawDocument.pageContent.length, "characters");
  console.log("Document Metadata       :", rawDocument.metadata);

  // --- 2. Initializing RecursiveCharacterTextSplitter ---
  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 150,      // Max characters per chunk
    chunkOverlap: 30,    // Shared characters between adjacent chunks for semantic continuity
  });

  // Split documents
  const splitChunks = await textSplitter.splitDocuments([rawDocument]);

  console.log(`\nSplit into ${splitChunks.length} chunks:\n`);

  splitChunks.forEach((chunk, index) => {
    console.log(`--- Chunk #${index + 1} (${chunk.pageContent.length} chars) ---`);
    console.log(`Content : "${chunk.pageContent.replace(/\n/g, " ")}"`);
    console.log(`Metadata:`, chunk.metadata);
    console.log();
  });
}

main().catch(console.error);
