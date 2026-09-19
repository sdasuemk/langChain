# The LangChain.js Handbook: Zero to Hero
*The Complete Conceptual & Practical Guide to Building LLM Applications in JavaScript & TypeScript*

---

## Welcome to the Language Factory

Imagine you are building a modern automated workshop:
* **The Prompts** are your **Blueprints**: Reusable design templates with dynamic placeholders.
* **The Models (LLMs)** are your **Engines**: The raw intelligence generating answers.
* **The LCEL (LangChain Expression Language)** is your **Pipe Plumbing**: Seamlessly directing the output of one machine directly into the input of the next.
* **The Text Splitter** is your **Paper Shredder & Binders**: Chopping massive 500-page manuals into digestible index cards.
* **The Vector Store** is your **Semantic Library**: Finding the exact right card based on *meaning* rather than exact keyword matches.
* **The RAG Chain** is an **Open-Book Exam**: Feeding the LLM exact excerpts before it generates an answer, eliminating hallucinations.
* **The Tools** are **Robotic Hands**: Allowing the LLM to run JavaScript functions, call APIs, and calculate numbers.

Let’s master LangChain.js from Zero to Hero.

---

# Phase 1: Blueprints & Engines (Prompts & Models)

### 1.1 The Anatomy of Chat Messages

Modern LLMs don't just take single strings; they converse using distinct **Roles**:
1. `SystemMessage`: The instructions, personality, and guardrails given to the assistant.
2. `HumanMessage`: What the user types into the chat box.
3. `AIMessage`: What the LLM generates in response.
4. `ToolMessage`: The data returned when an external function or API executes.

```javascript
import { HumanMessage, SystemMessage } from "@langchain/core/messages";

const conversation = [
  new SystemMessage("You are an expert full-stack TypeScript engineer."),
  new HumanMessage("How do I type a generic Promise in TypeScript?"),
];
```

---

### 1.2 ChatPromptTemplate: Reusable Blueprints

> **The Real-Life Scenario:**
> Instead of manually copying and pasting prompt text and string-interpolating inputs with `${userInput}`, `ChatPromptTemplate` prevents prompt injection, cleanly structures message roles, and validates that all variables exist before sending to the model.

```javascript
import { ChatPromptTemplate } from "@langchain/core/prompts";

const promptTemplate = ChatPromptTemplate.fromMessages([
  ["system", "You are a customer support agent for {companyName}."],
  ["user", "I need help with my order #{orderId}: {issueDescription}"],
]);

// Format into real message objects:
const messages = await promptTemplate.formatMessages({
  companyName: "Acme Cloud",
  orderId: "8841",
  issueDescription: "The server is returning HTTP 504 gateway timeout.",
});
```

---

# Phase 2: The Pipe Plumbing (LCEL: LangChain Expression Language)

In Python LangChain, chains are chained using the bitwise OR pipe operator:
```python
# Python syntax:
chain = prompt | model | output_parser
```

In JavaScript, because JS does not support operator overloading for `|`, we use the **`.pipe()`** method or **`RunnableSequence`**:

### 2.1 The `.pipe()` Paradigm
Every core component in LangChain.js is a **Runnable**. Every Runnable implements `.pipe()`:

```javascript
import { StringOutputParser } from "@langchain/core/output_parsers";

// The Cleanest JavaScript LCEL Chain:
const chain = promptTemplate
  .pipe(chatModel)
  .pipe(new StringOutputParser());

const result = await chain.invoke({
  companyName: "Acme Cloud",
  orderId: "8841",
  issueDescription: "Refund request",
});

console.log(result); // Clean string output!
```

### 2.2 Preserving Inputs with `RunnablePassthrough`
What if a downstream step needs the original user query alongside an intermediate calculation?
`RunnablePassthrough` passes the original input untouched:

```javascript
import { RunnablePassthrough, RunnableSequence } from "@langchain/core/runnables";

const chain = RunnableSequence.from([
  {
    // Fetches docs based on input:
    context: retriever.pipe(formatDocs),
    // Passes the raw user query through untouched:
    question: new RunnablePassthrough(),
  },
  prompt,
  model,
  new StringOutputParser(),
]);
```

---

# Phase 3: The Filter (Output Parsers)

LLMs output raw `AIMessage` objects containing token metadata and text. Output Parsers convert this output into usable JavaScript formats:

### 3.1 StringOutputParser
Extracts the text payload directly:
```javascript
import { StringOutputParser } from "@langchain/core/output_parsers";
const parser = new StringOutputParser();
const text = await parser.invoke(aiMessage); // "Here is your answer..."
```

### 3.2 JsonOutputParser
Parses JSON strings into native JavaScript objects:
```javascript
import { JsonOutputParser } from "@langchain/core/output_parsers";
const parser = new JsonOutputParser();
const obj = await parser.invoke(new AIMessage('{"status": "ok", "code": 200}'));
console.log(obj.status); // "ok"
```

### 3.3 StructuredOutputParser with Zod (Type Safety)
> **The Enterprise Contract:**
> When building APIs, you cannot gamble on whether the LLM outputs valid JSON. With Zod, you define a TypeScript schema, and LangChain auto-generates formatting instructions and validates the response.

```javascript
import { StructuredOutputParser } from "@langchain/core/output_parsers";
import { z } from "zod";

const schema = z.object({
  sentiment: z.enum(["positive", "neutral", "negative"]),
  score: z.number().min(0).max(1),
  tags: z.array(z.string()),
});

const parser = StructuredOutputParser.fromZodSchema(schema);

// Inject instructions into your prompt:
const prompt = ChatPromptTemplate.fromTemplate(
  "Analyze this review: {review}\n{format_instructions}"
);

const chain = prompt
  .pipe(model)
  .pipe(parser);
```

---

# Phase 4: Slicing the Books (Documents & Text Splitters)

> **The Library Dilemma:**
> If you feed a 300-page PDF into an LLM prompt, you exceed token limits, incur high costs, and encounter the *"Lost in the Middle"* problem where models ignore key facts buried in long contexts.

### 4.1 The `Document` Class
Every document in LangChain has two fields:
* `pageContent`: The raw text snippet.
* `metadata`: Key-value metadata (source file, author, page number, date).

### 4.2 RecursiveCharacterTextSplitter
Unlike naive character slicing (which cuts words in half), `RecursiveCharacterTextSplitter` tries to split on:
1. Double newlines (paragraphs)
2. Single newlines (lines)
3. Spaces (words)
4. Characters (individual letters)

```javascript
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";

const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 500,     // Target size in characters
  chunkOverlap: 50,   // Shared characters between chunks to preserve context
});

const chunks = await splitter.splitDocuments([rawDocument]);
```

---

# Phase 5: The Semantic Index (Vector Stores & Embeddings)

Traditional SQL searches look for exact matches: `WHERE text LIKE '%dog%'`.
If the document says *"puppy"*, SQL misses it completely!

### 5.1 What are Embeddings?
An Embedding model converts text into a high-dimensional array of numbers (e.g. 1,536 floats).
* In this vector space, the vector for *"Puppy"* is extremely close to *"Dog"*.
* The vector for *"Apple"* is far away from *"Quantum Mechanics"*.

### 5.2 MemoryVectorStore
Stores and indexes document vectors for fast cosine similarity search:

```javascript
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { OpenAIEmbeddings } from "@langchain/openai"; // Or any provider

// Index documents:
const vectorStore = await MemoryVectorStore.fromDocuments(chunks, new OpenAIEmbeddings());

// Query by semantic meaning:
const topMatches = await vectorStore.similaritySearch("How do I request time off?", 3);
```

---

# Phase 6: The Open-Book Exam (Retrieval-Augmented Generation - RAG)

> **The Open-Book Analogy:**
> An LLM without RAG is a student taking a closed-book medical exam from memory—they might hallucinate facts.
> **RAG** gives the student the exact relevant medical textbook pages right before answering!

```
User Question: "What is our PTO rollover policy?"
                      │
                      ▼
               [Embed Question]
                      │
                      ▼
         [Vector Store Similarity Search]
                      │
                      ▼
               Retrieved Chunks:
  "Section 5: Unused PTO rolls over up to 5 days."
                      │
                      ▼
            [Augment Prompt Template]
  "Context: Unused PTO rolls over up to 5 days.
   Question: What is our PTO rollover policy?"
                      │
                      ▼
                   [Model]
                      │
                      ▼
         Authoritative, Grounded Answer!
```

### The Complete JavaScript RAG Chain:
```javascript
const retriever = vectorStore.asRetriever(3);

const formatDocs = (docs) => docs.map((d) => d.pageContent).join("\n\n");

const ragPrompt = ChatPromptTemplate.fromMessages([
  ["system", "Answer strictly using the provided context:\n\n{context}"],
  ["user", "{question}"],
]);

const ragChain = RunnableSequence.from([
  {
    context: retriever.pipe(formatDocs),
    question: new RunnablePassthrough(),
  },
  ragPrompt,
  model,
  new StringOutputParser(),
]);

const answer = await ragChain.invoke("What is our PTO rollover policy?");
```

---

# Phase 7: Giving Models Hands (Tool Calling & bindTools)

When an LLM needs to calculate complex equations, fetch weather, or execute SQL queries, it uses **Tool Calling**:

```javascript
import { tool } from "@langchain/core/tools";
import { z } from "zod";

// 1. Define Tool with Zod:
const weatherTool = tool(
  async ({ city }) => `The weather in ${city} is 21°C and clear.`,
  {
    name: "get_weather",
    description: "Returns the current temperature for a city.",
    schema: z.object({
      city: z.string().describe("Name of the city"),
    }),
  }
);

// 2. Bind Tools to Chat Model:
const modelWithTools = chatModel.bindTools([weatherTool]);

// 3. Model outputs tool_calls instead of text:
const response = await modelWithTools.invoke([
  new HumanMessage("What is the weather in Tokyo right now?"),
]);

console.log(response.tool_calls);
// [{ name: "get_weather", args: { city: "Tokyo" }, id: "call_123" }]
```

---

# Python vs. JavaScript LangChain Rosetta Stone

| Concept | Python (`langchain`) | JavaScript (`@langchain/core`, `langchain`) |
| :--- | :--- | :--- |
| **Piping Chains** | `chain = prompt \| model \| parser` | `chain = prompt.pipe(model).pipe(parser)` |
| **Sequences** | `RunnableSequence(first, second)` | `RunnableSequence.from([first, second])` |
| **Passing Inputs** | `RunnablePassthrough()` | `new RunnablePassthrough()` |
| **String Parser** | `StrOutputParser()` | `new StringOutputParser()` |
| **JSON Parser** | `JsonOutputParser()` | `new JsonOutputParser()` |
| **Structured Output** | Pydantic (`BaseModel`) | Zod (`z.object({ ... })`) |
| **Prompt Template** | `ChatPromptTemplate.from_messages(...)` | `ChatPromptTemplate.fromMessages(...)` |
| **In-Memory Store** | `FAISS.from_documents(...)` | `MemoryVectorStore.fromDocuments(...)` |
| **Tool Definition** | `@tool` decorator | `tool(fn, { name, schema, description })` |
| **Bind Tools** | `model.bind_tools(tools)` | `model.bindTools(tools)` |

---

## How to Run All JavaScript Lessons

Navigate into `langChain_js` and install dependencies:

```bash
cd langChain_js
npm install

# Run any lesson:
npm run lesson:01   # Prompts & Models
npm run lesson:02   # LCEL Chains (.pipe() & Sequences)
npm run lesson:03   # Output Parsers (String, JSON, Zod)
npm run lesson:04   # Documents & Recursive Splitters
npm run lesson:05   # Vector Stores & Similarity Search
npm run lesson:06   # End-to-End RAG Chain with LCEL
npm run lesson:07   # Tool Calling & bindTools
```
