# Retrieval-Augmented Generation (RAG) & Document Loaders: Tutorial Note

This document explains the concept of **RAG (Retrieval-Augmented Generation)**, its architectural components, and the role of **Document Loaders** in modern LLM applications.

---

## 1. What is RAG?

### The Problem
Large Language Models (LLMs) are trained on massive public datasets, but they suffer from two major limitations:
1.  **Knowledge Cutoff**: They do not know about events that occurred after their training data was compiled.
2.  **No Access to Private Data**: They cannot access private company documents, custom databases, or user-specific files.
3.  **Hallucinations**: When asked about unknown or niche topics, LLMs may generate false information (hallucinations) with high confidence.

### The Solution: Retrieval-Augmented Generation (RAG)
RAG solves these problems by providing the LLM with an **external knowledge base**. Instead of relying solely on its internal training memory, the system searches (retrieves) relevant documents or data matching the user's query and inserts (augments) them into the prompt sent to the LLM. The LLM then answers the query based on the retrieved facts (generates).

*   **Why use RAG?** It makes LLM outputs accurate, verifiable (attributable to source files), cost-effective (no fine-tuning needed), and up-to-date.
*   **What is it?** A hybrid pattern merging information retrieval systems with text-generating LLMs.
*   **How does it work?** 
    1. A user asks a question.
    2. The system searches a data store for documents matching the question.
    3. The matching documents are combined with the user's question.
    4. The LLM reads the combined prompt and outputs an accurate answer.

---

## 2. RAG System Block Diagram

### Pipeline A: The Ingestion Pipeline (Preparation)
Before a user queries the system, documents must be loaded, processed, and stored in a vector database:

```
[Raw Files] (PDFs, Markdown, Webpages)
     │
     ▼
┌───────────────────────┐
│   Document Loaders    │  <── Extracts raw text and metadata from files
└───────────────────────┘
     │
     ▼
┌───────────────────────┐
│    Text Splitters     │  <── Chunks documents into smaller pieces (e.g. 500 characters)
└───────────────────────┘
     │
     ▼
┌───────────────────────┐
│   Embeddings Model    │  <── Converts text chunks into mathematical vectors (numbers)
└───────────────────────┘
     │
     ▼
┌───────────────────────┐
│    Vector Database    │  <── Stores vectors + original text metadata for search
└───────────────────────┘
```

### Pipeline B: The Retrieval and Generation Pipeline (Execution)
When a user asks a question, this real-time pipeline executes:

```
                  ┌─────────────────┐
                  │   User Query    │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ Embedding Model │ (Converts query to vector)
                  └────────┬────────┘
                           │
                           ▼
┌──────────────┐  Similarity Search
│  Vector DB   │ ◄──────────────────
└──────┬───────┘
       │
       ▼ (Top K Chunks)
┌─────────────────────────────────────────────────────────────┐
│                       Prompt Augmentation                   │
│                                                             │
│ "Use the following context to answer the user's question:    │
│ Context: [Retrieved Text Chunks...]                         │
│ Question: [User Query]"                                     │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │    LLM Model    │ (Reads prompt + context)
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │  Final Answer   │ (Accurate, hallucination-free response)
                  └─────────────────┘
```

---

## 3. Core Components of RAG

1.  **Document Loaders**: Connectors that extract text from various file formats and sources (PDFs, Webpages, APIs, Databases).
2.  **Text Splitters**: Split large documents into manageable text chunks.
3.  **Embeddings Models**: Algorithms that turn text into high-dimensional vectors representing semantic meaning.
4.  **Vector Stores (Vector DBs)**: Databases optimized for indexing and retrieving vector embeddings (e.g., Chroma, Pinecone, FAISS).
5.  **Retrievers**: Query interfaces that perform similarity searches on the Vector DB to return relevant chunks.
6.  **LLMs**: The brain that reads the retrieved context and answers the user's query.

---

## 4. Document Loaders

### The Idea
In LangChain, document loaders are classes that read source data and convert them into a standardized format: a list of **`Document`** objects.

A `Document` object always has two fields:
*   `page_content`: The extracted text string.
*   `metadata`: A Python dictionary containing info about the source (e.g., `{"source": "asset/llm-book.pdf", "page": 1}`).

### Why, What, and How
*   **Why**: Data exists in hundreds of different formats (CSV, JSON, HTML, PDF, Google Drive, Notion). Document Loaders abstract away this complexity so that downstream text splitters and embeddings models always receive the exact same standardized `Document` structure.
*   **What**: They are Python classes containing a `.load()` method that returns a list of `Document` objects.
*   **How**: They load the file, parse the content using helper libraries (like `pypdf` for PDFs, `beautifulsoup4` for web scraping, etc.), package the text, and populate the metadata.

### Code Examples of Common Loaders in LangChain

Here is how you load different files in Python:

#### 1. Loading Text Files (`TextLoader`)
```python
from langchain_community.document_loaders import TextLoader

# Initialize the loader with a text file path
loader = TextLoader("asset/readme.txt", encoding="utf-8")

# Load documents
docs = loader.load()

# Inspect results
print(docs[0].page_content)
print(docs[0].metadata)  # e.g., {'source': 'asset/readme.txt'}
```

#### 2. Loading PDF Files (`PyPDFLoader`)
```python
from langchain_community.document_loaders import PyPDFLoader

# Initialize the loader (automatically handles multi-page files)
loader = PyPDFLoader("asset/llm-book.pdf")

# Extract pages. Each page becomes a separate Document object
docs = loader.load()

print(f"Total Pages Loaded: {len(docs)}")
print(docs[0].page_content[:200])  # Print first 200 characters of page 1
print(docs[0].metadata)             # e.g., {'source': 'asset/llm-book.pdf', 'page': 0}
```

#### 3. Loading CSV Data (`CSVLoader`)
```python
from langchain_community.document_loaders.csv_loader import CSVLoader

# Initialize the loader. Each row in the CSV becomes a separate Document
loader = CSVLoader(file_path="asset/users.csv")
docs = loader.load()

print(docs[0].page_content)  # Automatically formats CSV columns as 'Header: Value' text lines
print(docs[0].metadata)       # e.g., {'source': 'asset/users.csv', 'row': 0}
```

#### 4. Loading Webpages (`BSHTMLLoader` or `WebBaseLoader`)
```python
from langchain_community.document_loaders import WebBaseLoader

# Fetch and scrape text from a URL
loader = WebBaseLoader("https://python.langchain.com/v0.2/docs/introduction/")
docs = loader.load()

print(docs[0].page_content[:300])
print(docs[0].metadata)  # Includes page title, source URL, description, etc.
```
