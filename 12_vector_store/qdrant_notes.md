# Qdrant in LangChain

**Qdrant** is a high-performance vector similarity search engine and database written in Rust. It offers developers a unified API matching local in-memory testing, standalone Docker staging container, and production Qdrant cloud deployments.

---

## 1. What is Qdrant? (Why, What, How)

### Why Qdrant?
1.  **Written in Rust**: Designed for extreme speed, low footprint, and safe parallel execution.
2.  **Hybrid Filter Support**: Fully supports combining vector similarity calculation with exact payload filtering (types like ranges, match text, geographic parameters).
3.  **Local Dev Parity**: You can run it inside Python RAM (`:memory:`) without launching docker instances or database servers, and easily transition to docker.

---

## 2. Core Setup & Packages
*   **Install**:
    ```bash
    pip install langchain-qdrant qdrant-client
    ```
*   **Local Docker Setup**:
    ```bash
    docker run -p 6333:6333 qdrant/qdrant
    ```

---

## 3. Core Database Operations in LangChain

### A. Environment Configuration & Connection
We securely load configurations and credentials.

```python
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
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
Depending on deployment, configure local memory or remote client using direct instantiation to bypass deprecated from_documents arguments:

```python
from langchain_core.documents import Document
from qdrant_client.models import Distance, VectorParams

documents = [
    Document(page_content="Mumbai Indians (MI): Jasprit Bumrah is a world-class bowler.", metadata={"team": "MI"}),
    Document(page_content="Chennai Super Kings (CSK): MS Dhoni is the captain.", metadata={"team": "CSK"})
]

# Mode 1: In-Memory (Great for unit testing)
client = QdrantClient(location=":memory:")

# Ensure collection exists before vector store instantiation (avoids validation errors)
if not client.collection_exists("ipl_rosters"):
    client.create_collection(
        collection_name="ipl_rosters",
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )

vector_store = QdrantVectorStore(
    client=client,
    collection_name="ipl_rosters",
    embedding=embeddings
)
# Add documents to the store
vector_store.add_documents(documents=documents)

# Mode 2: Standalone Docker Server
# client = QdrantClient(url="http://localhost:6333")
# if not client.collection_exists("ipl_rosters"):
#     client.create_collection(
#         collection_name="ipl_rosters",
#         vectors_config=VectorParams(size=384, distance=Distance.COSINE)
#     )
# vector_store = QdrantVectorStore(
#     client=client,
#     collection_name="ipl_rosters",
#     embedding=embeddings
# )
# vector_store.add_documents(documents=documents)
```

---

## 4. Querying & Search Types

### A. Similarity Search
```python
results = vector_store.similarity_search("Who is the batsman?", k=1)
print(results[0].page_content)
```

### B. Similarity Search with Score
In Qdrant, distance/similarity scores represent dot product, cosine, or L2 metric according to index properties.
```python
results_with_scores = vector_store.similarity_search_with_score("MS Dhoni captain", k=1)
doc, score = results_with_scores[0]
print(f"Similarity Distance Score: {score:.4f}")
```

### C. Metadata Filtering
Qdrant uses advanced payload filtering constraints using `qdrant_client.models.Filter`:
```python
from qdrant_client.models import Filter, FieldCondition, MatchValue

results = vector_store.similarity_search(
    "Fast bowler",
    k=1,
    filter=Filter(
        must=[
            FieldCondition(
                key="metadata.team",
                match=MatchValue(value="MI")
            )
        ]
    )
)
```

---

## 5. CRUD updates & deletions

### A. Updating Documents
QdrantVectorStore does not have a native `update_documents()` method. However, you can update documents directly using `add_documents()` by passing the target document IDs. If the ID already exists, Qdrant will overwrite (upsert) the existing document:

```python
import uuid

# Generate a deterministic UUID for the ID "mi_001"
mi_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, "mi_001"))

updated_doc = Document(
    page_content="Mumbai Indians (MI): Suryakumar Yadav is the captain. Jasprit Bumrah leads fast bowling.",
    metadata={"team": "MI", "city": "Mumbai"}
)

# This performs an upsert, updating the document with the matching ID
vector_store.add_documents(ids=[mi_uuid], documents=[updated_doc])
```

Alternatively, you can also delete the document by its UUID first, and then add the updated document:
```python
# Delete the old document and add the updated one
vector_store.delete(ids=[mi_uuid])
vector_store.add_documents(documents=[updated_doc])
```

### B. Deleting Documents
```python
# Deletes index entries from the target collection using its UUID representation
vector_store.delete(ids=[mi_uuid])
```
