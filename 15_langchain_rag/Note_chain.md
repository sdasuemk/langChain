# YouTube RAG: LCEL Chaining Tutorial Note

This document explains the concept of **LangChain Expression Language (LCEL)** and details how to orchestrate our YouTube RAG pipeline using declarative pipe (`|`) operators..

---

## 1. What is LCEL?

**LangChain Expression Language (LCEL)** is a declarative way to compose different LangChain components (such as prompt templates, LLMs, document loaders, retrievers, and output parsers) into unified pipelines.

### The Pipe (`|`) Operator
LCEL leverages Python's bitwise OR operator `|` to pipe data from one component to another. The output of the left component automatically becomes the input of the right component:

```
[Input Data] ──► [Component A] ──(output of A)──► [Component B] ──► [Parsed Output]
```

In code, this is written as:
```python
chain = component_a | component_b
```

---

## 2. The LCEL RAG Chain Architecture

In [`youtube_rag_chain.py`](file:///c:/Coding/langChain/15_langchain_rag/youtube_rag_chain.py), we construct the final QA pipeline using a single LCEL chain definition:

```python
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# 1. Define formatting helper for retrieved context documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 2. Assemble RAG pipeline using explicit RunnableParallel and RunnableLambda
parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})
parser = StrOutputParser()
rag_chain = parallel_chain | prompt_template | chat_model | parser
```

### How Data Flows Through the Chain
When we execute the chain by calling `rag_chain.invoke("What are the two files that make up llama 2?")`, the data moves through the following stages:

1.  **Input Mapping (`parallel_chain` using `RunnableParallel`)**:
    - The raw input string question (`"What are..."`) is passed.
    - **`"question"`**: `RunnablePassthrough()` takes the input string and passes it directly through to the human question placeholder.
    - **`"context"`**: The input string is passed to `retriever`. The retriever retrieves matching documents from the FAISS database, and pipes (`|`) those documents to `RunnableLambda(format_docs)`, which executes our formatting function and combines them into a single string.
    - **Output of this stage**: A dictionary containing `{"context": "[retrieved context string]", "question": "What are..."}`.
2.  **Prompt Augmentation (`prompt_template`)**:
    - The dictionary from stage 1 is piped to `prompt_template`.
    - The template formats the `{context}` and `{question}` fields into System and Human messages.
    - **Output of this stage**: A `ChatPromptValue` object (representing formatted messages).
3.  **LLM Invocation (`chat_model`)**:
    - The formatted messages are piped to `chat_model` (`ChatHuggingFace` wrapper pointing to `DeepSeek-V4-Pro`).
    - The model computes the next tokens and outputs the answer.
    - **Output of this stage**: An `AIMessage` object containing the model's text response.
4.  **Output Parsing (`StrOutputParser`)**:
    - The `AIMessage` is piped to `StrOutputParser()`, which extracts the inner `.content` string.
    - **Output of this stage**: A clean Python string containing the final answer.

---

## 3. Comparing Step-by-Step vs LCEL Chain

| Phase | Step-by-Step (`youtube_rag.py`) | LCEL Chain (`youtube_rag_chain.py`) |
| :--- | :--- | :--- |
| **Orchestration** | Procedural (Explicit python variables & loops). | Declarative (Structured composition using `\|` operator). |
| **Retrieval** | Manual call: `results = retriever.invoke(query)`. | Automated: Integrated into input mapper. |
| **Augmentation** | Manual string building and prompt invocation. | Automated: Context is piped directly into prompt template. |
| **Generation** | Explicit invocation: `chat_model.invoke(prompt)`. | Automated: Prompts are passed to model and parsed. |
| **Code Verbosity** | Multi-step variable handling inside loop. | Single unified call: `rag_chain.invoke(query)`. |

---

## 4. Execution & Verification

To run the LCEL chain RAG pipeline:
```powershell
.venv\Scripts\python 15_langchain_rag/youtube_rag_chain.py "https://www.youtube.com/watch?v=zjkBMFhNj_g" "What are the two files that make up llama 2?"
```
This runs the entire chain pipeline declarations, computes the answers, and prints the result.
