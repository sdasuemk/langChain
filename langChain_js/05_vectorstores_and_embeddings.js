/**
 * ============================================================================
 * LangChain.js - Lesson 5: Vector Stores & Embeddings
 * ============================================================================
 * Key Concepts:
 * 1. Embeddings: Transforms text into numeric vectors capturing semantic meaning.
 * 2. Vector Stores: Index and search embeddings using cosine similarity.
 * 3. MemoryVectorStore: In-memory vector database provided by LangChain.
 * 4. Similarity Search with Scores: Finding top-K closest matching documents.
 * ============================================================================
 */

import { Document } from "@langchain/core/documents";
import { MemoryVectorStore } from "langchain/vectorstores/memory";

// --- 1. Deterministic Standalone Embeddings (No API keys needed) ---
class SimulatedEmbeddings {
  async embedDocuments(documents) {
    return documents.map((doc) => this._hashText(doc));
  }

  async embedQuery(query) {
    return this._hashText(query);
  }

  _hashText(text) {
    // Generate a normalized 4-dimensional semantic vector
    const lower = text.toLowerCase();
    const v1 = lower.includes("cat") || lower.includes("kitten") ? 1.0 : 0.1;
    const v2 = lower.includes("dog") || lower.includes("puppy") ? 1.0 : 0.1;
    const v3 = lower.includes("code") || lower.includes("javascript") ? 1.0 : 0.1;
    const v4 = lower.includes("food") || lower.includes("pizza") ? 1.0 : 0.1;
    return [v1, v2, v3, v4];
  }
}

async function main() {
  console.log("=".repeat(65));
  console.log("  LangChain.js Lesson 5: MemoryVectorStore & Similarity Search");
  console.log("=".repeat(65));

  const embeddings = new SimulatedEmbeddings();

  // --- 2. Seed Knowledge Documents ---
  const sampleDocs = [
    new Document({
      pageContent: "JavaScript is a versatile programming language for web development.",
      metadata: { topic: "coding", id: 1 },
    }),
    new Document({
      pageContent: "Puppies are energetic young dogs that love to play fetch.",
      metadata: { topic: "animals", id: 2 },
    }),
    new Document({
      pageContent: "Kittens are playful young cats that enjoy chasing yarn.",
      metadata: { topic: "animals", id: 3 },
    }),
    new Document({
      pageContent: "Neapolitan pizza is baked in wood-fired ovens with mozzarella cheese.",
      metadata: { topic: "cooking", id: 4 },
    }),
  ];

  // --- 3. Index Documents in MemoryVectorStore ---
  console.log("\nIndexing documents into MemoryVectorStore...");
  const vectorStore = await MemoryVectorStore.fromDocuments(sampleDocs, embeddings);
  console.log("Vector store indexing complete!");

  // --- 4. Perform Similarity Search ---
  const query = "Tell me about young cats";
  console.log(`\nQuery: "${query}"`);

  const results = await vectorStore.similaritySearch(query, 2);

  console.log(`\nTop ${results.length} Matches Found:`);
  results.forEach((doc, idx) => {
    console.log(`\n[Match #${idx + 1}] Topic: ${doc.metadata.topic}`);
    console.log(`Content: "${doc.pageContent}"`);
  });

  // --- 5. Similarity Search With Distance Scores ---
  const scoredResults = await vectorStore.similaritySearchWithScore(query, 2);
  console.log(`\nMatches with Cosine Similarity Distance:`);
  scoredResults.forEach(([doc, score], idx) => {
    console.log(`  * #${idx + 1}: Score = ${score.toFixed(4)} | "${doc.pageContent.slice(0, 40)}..."`);
  });
}

main().catch(console.error);
