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

---

## 5. Advanced Loader Techniques

### Lazy Loading vs Batch Loading
Most document loaders support two ways of retrieving parsed data:
1.  **`.load()` (Batch)**: Reads the entire resource, parses it, builds a full Python list of `Document` objects, and returns it. This is straightforward but can saturate system RAM if loading a 1,000-page PDF or thousands of text files simultaneously.
2.  **`.lazy_load()` (Lazy)**: Returns a Python generator that yields `Document` objects page-by-page or line-by-line. This consumes minimal RAM because only one page/document is kept in active memory at a time.

```python
# Streaming pages of a PDF to process on the fly without RAM spikes
for page in loader.lazy_load():
     # Perform your custom processing, such as printing the page content
    process_page_text(page.page_content)  # any custom processing can be done here.
    print(page.page_content) # print page by page
```

### Loading Scanned PDFs (OCR)
If a PDF file is scanned (made of image files instead of digital text), standard text extractors will return empty content. 
To extract text, we configure `PyPDFLoader` to use Optical Character Recognition (OCR):
*   **Prerequisites**: `pip install rapidocr-onnxruntime pillow`
*   **Setup**: Use the `extract_images=True` option and supply the `RapidOCRBlobParser`:

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.parsers.images import RapidOCRBlobParser

loader = PyPDFLoader(
    file_path="scanned_document.pdf",
    extract_images=True,
    images_parser=RapidOCRBlobParser()
)
docs = loader.load() # Extracted text from scanned page images is loaded into page_content
```

#### Why not UnstructuredPDFLoader?
`UnstructuredPDFLoader` is a popular alternative that handles layout partitioning and automated OCR (using Tesseract). However:
*   **Local setup is complex**: It requires installing binary operating system tools (like Tesseract and Poppler utilities) and adding them to the PATH, which is error-prone on Windows.
*   **Hugging Face / RapidOCR alternative**: Using `PyPDFLoader` with `RapidOCRBlobParser` runs locally in Python without demanding complex system tool configuration.

### Creating a Custom Document Loader
If you need to ingest data from a unique database, custom API format, or proprietary file type, you can build a custom loader by subclassing `BaseLoader` and overriding `lazy_load()`.

```python
from typing import Iterator
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

class UserProfileLoader(BaseLoader):
    def __init__(self, profiles_list: list):
        self.profiles_list = profiles_list

    def lazy_load(self) -> Iterator[Document]:
        for profile in self.profiles_list:
            yield Document(
                page_content=f"User {profile['name']} works as a {profile['role']}.",
                metadata={"user_id": profile["id"], "source": "user_api"}
            )
```

---

## 6. Directory File Mapping

The examples in this directory illustrate these document loading concepts step-by-step:
*   [text_loader_1.py](file:///c:/Coding/langChain/10_document_loader/text_loader_1.py): A simple demonstration of loading raw text files using `TextLoader`.
*   [text_loader_summarizer_chain_2.py](file:///c:/Coding/langChain/10_document_loader/text_loader_summarizer_chain_2.py): Loads a poem using `TextLoader` and summarizes it using an LCEL pipeline and the Hugging Face hosted LLM API.
*   [scanned_pdf_chat_3.py](file:///c:/Coding/langChain/10_document_loader/scanned_pdf_chat_3.py): An interactive RAG chat session over scanned PDFs using OCR, an `InMemoryVectorStore`, and an LCEL RAG chain.
*   [simple_loader_example_4.py](file:///c:/Coding/langChain/10_document_loader/simple_loader_example_4.py): Shows side-by-side standard text loading (`TextLoader`) and PDF page loading (`PyPDFLoader`).
*   [directory_loader_5.py](file:///c:/Coding/langChain/10_document_loader/directory_loader_5.py): Demonstrates concurrent bulk file loading from a directory using `DirectoryLoader`.
*   [custom_loader_6.py](file:///c:/Coding/langChain/10_document_loader/custom_loader_6.py): Demonstrates how to write your own custom log file parser by inheriting from `BaseLoader`.
*   [pdf_lazy_loader_7.py](file:///c:/Coding/langChain/10_document_loader/pdf_lazy_loader_7.py): Demonstrates streaming large PDF documents page-by-page to save memory.

