# LangChain Expression Language (LCEL) Chains: Reference Note

In LangChain, a **Chain** is a modular pipeline that links multiple components together. This document explains the core ideas, diagrams, real-world examples, and implementation techniques of different chain structures.

---

## 1. What are Chains? (Why, What, How)

### The Idea: The Assembly Line
Think of a chain like an **assembly line in a factory**:
*   Instead of one worker trying to do everything (building an entire car from scratch), the task is broken down into small, specialized steps.
*   Step 1 builds the frame.
*   Step 2 adds the engine.
*   Step 3 paints the body.
*   Each step passes its output down the conveyor belt to serve as the input for the next step.

### Why do we need Chains?
1.  **Modularity**: You can build, test, and debug small components (like prompt templates or custom parsers) individually.
2.  **Composability**: You can mix and match components to create completely new pipelines.
3.  **Readability**: Complex multi-step LLM operations become clean pipelines instead of nested spaghetti code.

### What is LCEL?
**LangChain Expression Language (LCEL)** is the declarative syntax used to compose chains. Every LCEL component implements the `Runnable` protocol, guaranteeing a standardized interface for execution (`.invoke()`, `.stream()`, `.batch()`, etc.).

### How does it work?
In Python, LCEL uses the **bitwise OR operator (`|`)** as a pipe operator to chain runnables together:

```python
# Conveyor Belt: User Input -> Prompt formats it -> LLM processes it -> Parser cleans it
chain = prompt | chat_model | output_parser
```

---

## 2. Key Features of LCEL

*   **Streaming Support**: Chaining with LCEL allows tokens to stream from the LLM to the client block-by-block immediately, reducing perceived latency.
*   **Async Support**: Any LCEL chain can be called using async methods (e.g. `.ainvoke()`, `.astream()`) for high-concurrency web servers.
*   **Parallel Execution**: If a step requires fetching multiple independent values, LCEL executes them in parallel (using `RunnableParallel` or dictionary syntax) to save time.
*   **Fallbacks**: Easily configure fallback models. If a primary API (like OpenAI) fails, the chain automatically routes requests to a secondary endpoint (like Hugging Face).

---

## 3. Types of Chains & Visualizations

Here is a breakdown of the 4 key chain architectures, complete with **real-world examples** and **visualizations**.

### A. Simple Chain
*   **Real-world Use**: A translator app. You input a sentence, it goes into a prompt, the model translates it, and the parser extracts the raw translated string.

#### Flow Diagram:
```
┌──────────────┐      ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│  User Input  ├─────►│   Prompt    ├─────►│  LLM Model  ├─────►│ Output      ├─────► Final
│  (Dict/Str)  │      │  Template   │      │   Wrapper   │      │  Parser     │       Text
└──────────────┘      └─────────────┘      └─────────────┘      └─────────────┘
```

#### LCEL Syntax:
```python
chain = prompt | chat_model | StrOutputParser()
```

---

### B. Sequential Chain
*   **Real-world Use**: Automatic blog writer. 
    *   **Chain 1**: Takes a topic and generates a comprehensive outline.
    *   **Chain 2**: Takes the outline from Chain 1 and writes the actual blog post.

#### Flow Diagram:
```
                  ┌──────────────────────────────────────────────┐
                  │                 Chain 1                      │
                  │  Input ──► Prompt 1 ──► LLM ──► Parser 1     │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼ (Intermediate Output / Context)
                  ┌──────────────────────┴───────────────────────┐
                  │                 Chain 2                      │
                  │  Context ──► Prompt 2 ──► LLM ──► Parser 2   │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                                    Final Output
```

#### LCEL Syntax:
```python
# Connect output of chain 1 directly to prompt of chain 2
chain = prompt1 | chat_model | parser | prompt2 | chat_model | parser
```

---

### C. Parallel Chain
*   **Real-world Use**: Product reviewer helper. You input a product name. In parallel, the system generates a list of **pros** and a list of **cons**. Finally, a synthesizer merges them into a balanced final review.

#### Flow Diagram:
```
                                 ┌──────────────┐      ┌─────────────┐
                           ┌────►│   Prompt A   ├─────►│   LLM (A)   ├────┐
                           │     │    (Pros)    │      │             │    │
     ┌──────────────┐      │     └──────────────┘      └─────────────┘    │      ┌──────────────┐
     │  User Input  ├──────┤                                              ├─────►│ Synthesizer  ├─────► Final
     │  (Product)   │      │     ┌──────────────┐      ┌─────────────┐    │      │   LLM /      │       Review
     └──────────────┘      └────►│   Prompt B   ├─────►│   LLM (B)   ├────┘      │ Prompt Chain │
                                 │    (Cons)    │      │             │           └──────────────┘
                                 └──────────────┘      └─────────────┘
```

#### LCEL Syntax:
```python
# Run A and B in parallel, then feed both to a final prompt
parallel_map = RunnableParallel(
    pros=pros_prompt | chat_model | parser,
    cons=cons_prompt | chat_model | parser
)
chain = parallel_map | synthesize_prompt | chat_model | parser
```

---

### D. Conditional Chain (Routing)
*   **Real-world Use**: Customer support triage. 
    *   If the user query is about "Billing", route to the **Billing LLM Chain**.
    *   If the query is about "Technical Issues", route to the **Technical LLM Chain**.
    *   Otherwise, route to the **General Support Chain**.

#### Flow Diagram:
```
                                                        ┌──────────────┐
                                                  ┌────►│ Billing Chain│
                                                  │     └──────────────┘
     ┌──────────────┐      ┌───────────────┐      │
     │  User Query  ├─────►│ Router Model  ├──────┼────►┌──────────────┐
     └──────────────┘      │ (Classification)     │     │ Tech Chain   │
                           └───────────────┘      │     └──────────────┘
                                                  │
                                                  └────►┌──────────────┐
                                                        │ General Chain│
                                                        └──────────────┘
```

#### LCEL Syntax:
```python
from langchain_core.runnables import RunnableBranch

# Define routing branch
branch = RunnableBranch(
    (lambda x: x["topic"] == "billing", billing_chain),
    (lambda x: x["topic"] == "tech", tech_chain),
    general_chain # Default fallback
)

chain = classification_chain | branch
```

---

## 4. Directory File Mapping

The examples in this directory illustrate these chain architectures step-by-step:
*   [simple_chain.py](file:///c:/Coding/langChain/chain/simple_chain.py): Basic pipeline demonstrating prompt formatting, LLM execution, string parsing, and chain graph visualization.
*   [sequencial_chain.py](file:///c:/Coding/langChain/chain/sequencial_chain.py): Combines two prompt-model sequences where the output text of the first becomes the input to the second.
*   [parallel_chain.py](file:///c:/Coding/langChain/chain/parallel_chain.py): Executes two chains in parallel (pros and cons analysis) and combines their outputs into a final summary.
*   [conditional_chain.py](file:///c:/Coding/langChain/chain/conditional_chain.py): Classifies input queries dynamically and routes them to specialized handler prompts.
