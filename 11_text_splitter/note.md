# LangChain Text Splitters: Reference Note

Text splitting is a foundational preprocessing step in LLM application pipelines, particularly when implementing Retrieval-Augmented Generation (RAG) and Semantic Search.

---

## 1. The Core Idea: Why, What, and How

### Why do we split text?
1. **Context Window Limits**: Large Language Models have limits on how much text they can process in a single prompt (e.g., 4k, 8k, or 128k tokens). Passing an entire book or a massive PDF will exceed these limits.
2. **Retrieval Accuracy**: If a user asks a specific question, retrieving a short, highly relevant 200-word paragraph yields a cleaner prompt and higher quality answer than retrieving an entire 50-page document.
3. **Cost and Latency**: LLM API fees and response times scale with the number of input tokens. Minimizing prompt size saves resources.
4. **Noise Reduction**: Vector databases find similarities based on average vector directions. Oversized chunks dilute the meaning of specific sentences, leading to poorer search results.

### What is text splitting?
Text splitting is the process of breaking down a continuous document (like a PDF, markdown article, or source code file) into smaller, semantically coherent blocks (called **chunks**).

### How does it work?
Most splitters work by following this algorithm:
1. Split the text into small, individual units (e.g., characters or sentences) using separators.
2. Combine these units into a single chunk until the size of the chunk reaches `chunk_size`.
3. Once `chunk_size` is reached, start a new chunk, but overlap a portion of the text from the previous chunk (defined by `chunk_overlap`) to maintain context.

```
Original Text: [Sentence A] [Sentence B] [Sentence C] [Sentence D] [Sentence E]

With chunk_size = 3 sentences and chunk_overlap = 1 sentence:
Chunk 1: [Sentence A] [Sentence B] [Sentence C]
Chunk 2:              [Sentence C] [Sentence D] [Sentence E]
```

---

## 2. Package Installation

Before running the text splitters, install the necessary dependencies using `pip`. You can install them all at once:

```bash
# Core LangChain and text splitters
pip install langchain langchain-text-splitters langchain-community

# Document loaders (e.g., PDF reading)
pip install pypdf

# Environment variables loader
pip install python-dotenv

# Experimental splitters (e.g., Semantic Chunking) and embeddings
pip install langchain-experimental langchain-huggingface sentence-transformers
```

---

## 3. Key Parameters Explained

*   **`chunk_size`**: The target maximum size of each chunk (measured in characters or tokens, depending on the splitter).
*   **`chunk_overlap`**: The amount of text shared between consecutive chunks. Keeping an overlap (e.g., 10-20% of `chunk_size`) prevents splitting critical sentences or concepts right down the middle, preserving relational context.
*   **`separator` / `separators`**: The character(s) used to split the text. If a splitter cannot fit the text into the `chunk_size` limit using the preferred separator, it falls back to the next separator.

---

## 4. Types of Text Splitters

| Splitter Type | Class / Method | When to Use |
| :--- | :--- | :--- |
| **Character Splitter** | `CharacterTextSplitter` | Simple splitting by a single literal separator (e.g., double newlines `"\n\n"`). |
| **Recursive Character Splitter** | `RecursiveCharacterTextSplitter` | General-purpose splitting. Splits recursively on a list of characters `["\n\n", "\n", " ", ""]` to keep sentences/paragraphs intact. |
| **Markdown Header Splitter** | `MarkdownHeaderTextSplitter` | Markdown documents. Splits text at markdown headers (`#`, `##`, etc.) and attaches header context to the chunk's metadata. |
| **Code Splitters** | `RecursiveCharacterTextSplitter.from_language` | Source code files. Uses programming language syntax (Python, JS, C++, etc.) to keep classes and functions together. |
| **Semantic Splitter** | `SemanticChunker` | RAG pipelines with highly varying topics. Splits text when it detects a significant shift in meaning, computed using vector embeddings. |

---

## 5. Directory File Mapping

The examples in this directory illustrate these splitters step-by-step:
*   [length_based_text_splitting_1.py](file:///c:/Coding/langChain/11_text_splitter/length_based_text_splitting_1.py): Introduces character-based splitting parameters (`chunk_size`, `chunk_overlap`, and separator rules).
*   [document_loader_text_splitter_combined_2.py](file:///c:/Coding/langChain/11_text_splitter/document_loader_text_splitter_combined_2.py): Shows how to parse a PDF document and split it into chunks while preserving metadata.
*   [structure_based_text_splitter_3.py](file:///c:/Coding/langChain/11_text_splitter/structure_based_text_splitter_3.py): Explains recursive fallback separators using `RecursiveCharacterTextSplitter`.
*   [markdown_text_splitter_4.py](file:///c:/Coding/langChain/11_text_splitter/markdown_text_splitter_4.py): Demonstrates splitting markdown text based on header structure, enriching metadata.
*   [python_code_splitter_5.py](file:///c:/Coding/langChain/11_text_splitter/python_code_splitter_5.py): Demonstrates syntax-aware Python code splitting.
*   [js_code_splitter_6.py](file:///c:/Coding/langChain/11_text_splitter/js_code_splitter_6.py): Demonstrates Node.js (JavaScript) and React (JSX) code splitting.
*   [sementic_meaning_based_splitting_7.py](file:///c:/Coding/langChain/11_text_splitter/sementic_meaning_based_splitting_7.py): Showcases semantic splitting utilizing API-based endpoint embeddings.
