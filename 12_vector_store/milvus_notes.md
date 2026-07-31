# Milvus in LangChain

**Milvus** is an open-source, highly-scalable vector database built for enterprise-grade AI search applications. It separates computation nodes from storage nodes, allowing you to orchestrate and scale individual services inside container clusters.

---

## 1. What is Milvus? (Why, What, How)

### Why Milvus?
1.  **Distributed Architecture**: Unmatched horizontal scaling capabilities designed to handle query/indexing workloads independently.
2.  **Milvus Lite**: Provides a lightweight SQLite-based local execution wrapper (running inside Python files) for development parity.
3.  **Advanced Vector Indexing**: Supports multiple index types (e.g. IVF_FLAT, HNSW, ANNOY) for tuning accuracy and speed trade-offs.

---

## 2. Core Setup & Packages
*   **Install**:
    ```bash
    pip install langchain-milvus pymilvus
    ```

---

## 3. Core Database Operations in LangChain

### A. Environment Configuration & Connection
We securely load credentials.

```python
from langchain_milvus import Milvus
from langchain_huggingface import HuggingFaceEndpointEmbeddings
import os
import getpass
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# Prompt for API keys if missing
if not os.environ.get("HUGGINGFACEHUB_API_TOKEN"):
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = getpass.getpass("Hugging Face API Token: ")

# Initialize embedding model
embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=os.environ.get("HUGGINGFACEHUB_API_TOKEN")
)
```

### B. Index Initialization
Depending on deployment, configure local Milvus Lite file or remote server:

```python
from langchain_core.documents import Document

documents = [
    Document(page_content="Mumbai Indians (MI): Jasprit Bumrah is a lead bowler.", metadata={"team": "MI"}),
    Document(page_content="Chennai Super Kings (CSK): MS Dhoni is the finisher.", metadata={"team": "CSK"})
]

# Mode 1: Milvus Lite (local file, no server installation needed)
vector_store = Milvus.from_documents(
    documents=documents,
    embeddings=embeddings,
    connection_args={"uri": "./milvus_db.db"},
    collection_name="ipl_rosters"
)

# Mode 2: Distributed Server connection (URI format http://localhost:19530)
# vector_store = Milvus.from_documents(
#     documents=documents,
#     embeddings=embeddings,
#     connection_args={"uri": "http://localhost:19530"},
#     collection_name="ipl_rosters"
# )
```

---

## 4. Querying & Search Types

### A. Similarity Search
```python
results = vector_store.similarity_search("Who plays for Chennai?", k=1)
print(results[0].page_content)
```

### B. Similarity Search with Score
```python
results_with_scores = vector_store.similarity_search_with_score("MS Dhoni captain", k=1)
doc, score = results_with_scores[0]
print(f"Similarity Score: {score:.4f}")
```

### C. Metadata Filtering
```python
results = vector_store.similarity_search(
    "Lead fast bowler",
    k=1,
    expr='team == "MI"'  # Milvus uses expression language string filters
)
```

---

## 5. CRUD updates & deletions

### A. Updating Documents
In Milvus, document elements are upserted. Supply the document list and matching IDs.
```python
updated_doc = Document(
    page_content="Chennai Super Kings (CSK): MS Dhoni is the legendary coach.",
    metadata={"team": "CSK"}
)

vector_store.add_documents(documents=[updated_doc], ids=["csk_001"])
```

### B. Deleting Documents
```python
# Deletes index entries by ID
vector_store.delete(ids=["csk_001"])
```
