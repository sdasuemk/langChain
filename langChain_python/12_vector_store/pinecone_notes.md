# Pinecone in LangChain

**Pinecone** is a fully managed, cloud-native vector database (SaaS) built to support large-scale semantic search, recommendation engines, and RAG pipelines. Since it operates in the cloud, you do not need to run local containers or manage index files.

---

## 1. What is Pinecone? (Why, What, How)

### Why Pinecone?
1.  **Fully Managed**: Zero infrastructure to set up, monitor, or maintain. Serverless and pod-based options deploy instantly.
2.  **Highly Scalable**: Smoothly handles millions to billions of high-dimensional vectors with sub-second response times.
3.  **Real-Time Updates**: Instantly reflects additions, edits, and deletions without manually indexing.
4.  **Metadata Filtering**: Highly optimized boolean filters on document attributes during query time.

---

## 2. Core Setup & Packages
*   **Install**:
    ```bash
    pip install langchain-pinecone
    ```
*   **API Registration**: You must sign up on Pinecone's dashboard to retrieve your **Pinecone API Key** and create an index with appropriate dimensions (e.g. 384 dimensions for the `all-MiniLM-L6-v2` model).

---

## 3. Core Database Operations in LangChain

### A. Environment Configuration & Connection
We securely fetch variables using `load_dotenv(find_dotenv())` and verify `PINECONE_API_KEY` is present.

```python
import os
import getpass
from dotenv import load_dotenv, find_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEndpointEmbeddings

load_dotenv(find_dotenv())

# Prompt for API keys if missing from env
if not os.environ.get("PINECONE_API_KEY"):
    os.environ["PINECONE_API_KEY"] = getpass.getpass("Pinecone API Key: ")
if not os.environ.get("HUGGINGFACEHUB_API_TOKEN"):
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = getpass.getpass("Hugging Face API Token: ")

# Initialize embedding model
embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=os.environ.get("HUGGINGFACEHUB_API_TOKEN")
)
```

### B. Index Initialization & Document Uploading
```python
from langchain_core.documents import Document

# Define Pinecone index name (must exist on the Pinecone console)
index_name = "ipl-rosters-index"

documents = [
    Document(page_content="Mumbai Indians (MI): Jasprit Bumrah is a lead bowler.", metadata={"team": "MI"}),
    Document(page_content="Chennai Super Kings (CSK): MS Dhoni is the finisher.", metadata={"team": "CSK"})
]

# Create store and upload vectors to the cloud index
vector_store = PineconeVectorStore.from_documents(
    documents=documents,
    embedding=embeddings,
    index_name=index_name
)
```

---

## 4. Querying & Search Types

### A. Similarity Search
```python
results = vector_store.similarity_search("Who plays for Chennai?", k=1)
print(results[0].page_content)
```

### B. Similarity Search with Score
In Pinecone, scores represent Cosine similarity (values between -1 and 1). **Higher scores represent closer similarity**.
```python
results_with_scores = vector_store.similarity_search_with_score("MS Dhoni captain", k=1)
doc, score = results_with_scores[0]
print(f"Cosine Similarity Score: {score:.4f}")
```

### C. Metadata Filtering
```python
results = vector_store.similarity_search(
    "Dangerous batsman",
    k=1,
    filter={"team": "CSK"}  # restrics target documents matching metadata key
)
```

---

## 5. CRUD updates & deletions

### A. Updating Documents
In Pinecone, document ids can be overwritten directly.
```python
updated_doc = Document(
    page_content="Chennai Super Kings (CSK): MS Dhoni is the legendary coach.",
    metadata={"team": "CSK"}
)

# Overwrites target vectors using its unique ID
vector_store.add_documents(documents=[updated_doc], ids=["csk_001"])
```

### B. Deleting Documents
```python
# Deletes index entries from the cloud database
vector_store.delete(ids=["csk_001"])
```
