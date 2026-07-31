# FAISS (Facebook AI Similarity Search) in LangChain

**FAISS** is a highly optimized library for efficient similarity search and clustering of dense vectors. Developed by Meta's AI research team, it runs entirely local (in-memory or serialized directly to disk files) and executes extremely fast CPU/GPU search indexes.

---

## 1. What is FAISS? (Why, What, How)

### Why FAISS?
1.  **Lightweight & local**: It runs as a compiled C++ library inside your Python process. No server administration or network overhead is required.
2.  **Blazing-fast Indexing**: Specifically optimized for fast index construction and nearest-neighbor search, supporting advanced clustering and compression (Quantization).
3.  **GPU Acceleration**: Supports seamless CUDA execution for indexing billions of vectors on compatible graphics hardware.

### What are its storage mechanisms?
Unlike server-based databases, FAISS is an **in-memory library**. It compiles index structures in RAM.
*   **Disk Persistence**: FAISS writes index models directly to standard binary files (`index.faiss` and `index.pkl`) on disk. You reload these binary files back to memory to restore the database index.

---

## 2. Core Setup & Package Installation
*   **CPU Version (Recommended for most cases)**:
    ```bash
    pip install langchain-community faiss-cpu
    ```
*   **GPU Version (Requires NVIDIA CUDA Toolkit)**:
    ```bash
    pip install faiss-gpu
    ```

---

## 3. Core Database Operations in LangChain

### A. Formation & Embedding Function
First, we configure the API-based embedding model and load the environment variables using `load_dotenv(find_dotenv())`.

```python
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpointEmbeddings
import os
import getpass
from dotenv import load_dotenv, find_dotenv

# Load .env
load_dotenv(find_dotenv())

# Prompt for Hugging Face token if missing
if not os.environ.get("HUGGINGFACEHUB_API_TOKEN"):
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = getpass.getpass("Hugging Face API Token: ")

# Initialize embeddings
embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=os.environ.get("HUGGINGFACEHUB_API_TOKEN")
)
```

### B. Adding Documents & Index Creation
```python
from langchain_core.documents import Document

# Define document objects
documents = [
    Document(page_content="Mumbai Indians (MI): Rohit Sharma is a legendary opening batsman.", metadata={"team": "MI", "id": "mi_001"}),
    Document(page_content="Chennai Super Kings (CSK): MS Dhoni is the legendary captain.", metadata={"team": "CSK", "id": "csk_001"})
]

# Instantiate FAISS and build the initial vector index in RAM
vector_store = FAISS.from_documents(documents, embeddings)
```

### C. Direct Serialization (Save & Load)
```python
# 1. Save index to local directory
vector_store.save_local("faiss_index_store")

# 2. Reload index from disk later
loaded_store = FAISS.load_local(
    folder_path="faiss_index_store",
    embeddings=embeddings,
    allow_dangerous_deserialization=True  # Required to load pickled objects safely
)
```

---

## 4. Querying & Search Types

### A. Standard Similarity Search
```python
results = vector_store.similarity_search("Who plays for Mumbai?", k=1)
print(results[0].page_content)
```

### B. Similarity Search with Score (Distance Evaluation)
For FAISS, the similarity score represents **L2 (Euclidean) distance** or **Dot Product/Cosine similarity** depending on the index type. For L2, **lower scores represent closer similarity**.
```python
results_with_scores = vector_store.similarity_search_with_score("MS Dhoni captain", k=1)
doc, score = results_with_scores[0]
print(f"L2 Distance: {score:.4f}")
```

---

## 5. Updating & Deleting Documents

### A. Updating Documents
FAISS indexes are technically static. To update records, LangChain's FAISS wrapper deletes the target elements and appends the new representations.
```python
updated_doc = Document(
    page_content="Mumbai Indians (MI): Suryakumar Yadav is the captain.",
    metadata={"team": "MI", "id": "mi_001"}
)

# FAISS does not have a native update_documents() method.
# We delete by ID first, then re-add the updated document with the same ID.
# Since delete() can raise NotImplementedError in older packages, we use a fallback to rebuild the index.
try:
    vector_store.delete(ids=["mi_001"])
except Exception:
    # Rebuild fallback: filter out deleted doc and recreate the store from the docstore
    remaining_docs = [doc for doc_id, doc in vector_store.docstore._dict.items() if doc_id not in ["mi_001"]]
    vector_store = FAISS.from_documents(remaining_docs, embeddings)

vector_store.add_documents(documents=[updated_doc], ids=["mi_001"])
```

### B. Deleting Documents
```python
# Deletes index entries by ID with a try-except rebuild fallback
try:
    vector_store.delete(ids=["mi_001"])
except Exception:
    remaining_docs = [doc for doc_id, doc in vector_store.docstore._dict.items() if doc_id not in ["mi_001"]]
    vector_store = FAISS.from_documents(remaining_docs, embeddings)
```
