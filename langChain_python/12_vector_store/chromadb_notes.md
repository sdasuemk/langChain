# ChromaDB in LangChain: Detailed Reference Note

**ChromaDB** (or simply **Chroma**) is an open-source, AI-native vector database designed to save, search, and manage embeddings. It is the most popular choice for local RAG development due to its lightweight setup and clean integration with LangChain.

---

## 1. What is ChromaDB? (Why, What, How)

### Why Chroma?
1.  **Lightweight & Embeddable**: Chroma can run directly inside your Python application process (in-memory or saved to local disk). You do not need to boot a separate database server.
2.  **Fast Prototyping**: It requires zero cloud setup, APIs, or registration.
3.  **Rich Metadata Filtering**: Enables query isolation by tagging documents (e.g., searching only documents belonging to a specific user, department, or team).

### What are its storage modes?
*   **In-Memory (Ephemeral)**: Data is stored in your computer's RAM. It is extremely fast but vanishes when the Python script stops. Ideal for test suites.
*   **Local Persistent (Disk Storage)**: Saves data locally to SQLite and parquet files in a directory on your machine. Data persists across restarts.
*   **Client-Server (Distributed Mode)**: Chroma runs as a standalone docker container or cloud server, and your Python code communicates with it via HTTP requests.

---

## 2. ChromaDB Operations in LangChain: An IPL Players Case Study

To explain Chroma's capabilities, let's use a real-world scenario: **Storing rosters of 10 IPL (Indian Premier League) Teams**.

### The Setup
We have 10 documents, each containing roster information for a specific IPL team.
*   **Document 1**: *"Mumbai Indians (MI): Rohit Sharma is a legendary batsman. Jasprit Bumrah is a world-class fast bowler."* (Metadata: `{"team": "MI", "city": "Mumbai"}`)
*   **Document 2**: *"Chennai Super Kings (CSK): MS Dhoni is the legendary captain and wicketkeeper. Ravindra Jadeja is a top all-rounder."* (Metadata: `{"team": "CSK", "city": "Chennai"}`)
*   *...and so on for 10 teams.*

---

## 3. Core Database Operations in LangChain

### A. Formation & Embedding Function
First, we configure the **API-based** embedding model. 

*   **API-Based (Recommended)**: Uses `HuggingFaceEndpointEmbeddings`. It performs remote API calls to Hugging Face Inference endpoints, requiring **no local memory or CPU/GPU processing**. You must configure the `HUGGINGFACEHUB_API_TOKEN` in your environment.
*   **Local-Based (Alternative)**: Uses `HuggingFaceEmbeddings`. It downloads the model weights (e.g. 500MB) locally and executes computations on your machine's processor.

```python
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEndpointEmbeddings
import os
import getpass

# Ensure API token is present in the environment to avoid 401 Unauthorized errors
if not os.environ.get("HUGGINGFACEHUB_API_TOKEN"):
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = getpass.getpass("Enter your Hugging Face API Token: ")

# 1. Initialize the API-based embedding model
embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=os.environ.get("HUGGINGFACEHUB_API_TOKEN")
)

# --- LOCAL ALTERNATIVE (Uncomment to run locally without API keys) ---
# from langchain_huggingface import HuggingFaceEmbeddings
# embeddings = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2"
# )
# ---------------------------------------------------------------------

# 2. Form the vector store in-memory (or supply persist_directory for disk storage)
vector_store = Chroma(
    collection_name="ipl_rosters",
    embedding_function=embeddings
)
```

### B. Adding Documents
We convert our raw data strings into `Document` objects and store them. LangChain computes their embeddings automatically.

```python
from langchain_core.documents import Document

# Build document objects with metadata
documents = [
    Document(
        page_content="Mumbai Indians (MI): Rohit Sharma is a legendary batsman. Jasprit Bumrah is a world-class fast bowler.",
        metadata={"team": "MI", "city": "Mumbai"},
        id="mi_001" # Supplying explicit unique IDs makes Updates and Deletes easy!
    ),
    Document(
        page_content="Chennai Super Kings (CSK): MS Dhoni is the legendary captain and wicketkeeper. Ravindra Jadeja is a top all-rounder.",
        metadata={"team": "CSK", "city": "Chennai"},
        id="csk_001"
    )
]

# Add documents to Chroma
vector_store.add_documents(documents)
```

---

## 4. Querying & Similarity Search Types

### A. Standard Similarity Search
Returns a list of matching `Document` objects based on vector distance.

```python
# Query: "Who is the fast bowler?"
results = vector_store.similarity_search("Who is the fast bowler?", k=1)
print(results[0].page_content)
# Output: "Mumbai Indians (MI): Rohit Sharma is a legendary batsman. Jasprit Bumrah..."
```

### B. Similarity Search with Score
Returns a list of tuples: `(Document, Score)`. 
*   **Note on Scores**: For Chroma, the score represents the **distance metric** (typically L2 squared distance). 
*   **Interpretation**: **Lower scores represent closer matches (higher similarity)**. A score of `0.0` would mean an exact duplicate vector.

```python
results_with_scores = vector_store.similarity_search_with_score("MS Dhoni captain", k=1)
doc, score = results_with_scores[0]
print(f"Match: {doc.page_content}")
print(f"Distance Score (L2): {score:.4f}")
```

### C. Metadata Filtering
Narrow down your search area to save computation and ensure correct contexts.

```python
# Query: "Who is the legendary batsman?" but search ONLY CSK's roster
results = vector_store.similarity_search(
    "Who is the legendary batsman?",
    k=1,
    filter={"team": "CSK"}
)
# This will output Chennai Super Kings' document instead of Mumbai Indians' document
# because the filter restricts the scan to 'CSK' metadata.
```

---

## 5. View Details in Vector Database (Inspection)

To see the internal collection metrics and verify the contents of the database, you can interact with Chroma's native collection object.

```python
# Get raw access to Chroma's native collection object
collection = vector_store._collection

# 1. Count items in collection
print(f"Total Vectors stored: {collection.count()}")

# 2. Peek at the first item's raw vectors, IDs, and metadata
peek_data = collection.peek(limit=1)
print("Stored ID:", peek_data["ids"])
print("Stored Metadata:", peek_data["metadatas"])
```

---

## 6. Updating & Deleting Documents

### A. Updating
If Jasprit Bumrah is traded to another team, or we need to update the text content, we overwrite the record using its **Unique ID**.

```python
updated_doc = Document(
    page_content="Mumbai Indians (MI): Suryakumar Yadav is the captain. Jasprit Bumrah remains the lead fast bowler.",
    metadata={"team": "MI", "city": "Mumbai"},
    id="mi_001" # Matching the original ID overwrites the old vector and text!
)

# Overwrites the database record with the new vector representation (requires both ids and documents lists)
vector_store.update_documents(ids=["mi_001"], documents=[updated_doc])
```

### B. Deleting
If an IPL team drops out of the league, we can delete their roster using their IDs.

```python
# Deletes the Mumbai Indians document from the collection
vector_store.delete(ids=["mi_001"])
```

### C. Retrieving Documents Directly (get())
Sometimes you want to retrieve documents directly by their unique ID or metadata query without running a slow similarity search vector calculation. LangChain's Chroma wrapper exposes the `.get()` method for this:

```python
# 1. Retrieve a document by its unique ID
data = vector_store.get(ids=["csk_001"])
print("Documents fetched:", data["documents"])
print("Metadatas fetched:", data["metadatas"])

# 2. Retrieve documents matching metadata filters
filtered_data = vector_store.get(where={"team": "CSK"})

# 3. Retrieve specific fields (e.g. including the raw vectors/embeddings)
# By default, embeddings are excluded to save bandwidth. You can explicitly include them:
complete_data = vector_store.get(
    ids=["csk_001"], 
    include=["documents", "metadatas", "embeddings"]
)
print("Stored Vector Embeddings:", complete_data["embeddings"])
```


---

## 7. Where is the Vector Store Saved and How Do I Inspect It?

Depending on how you initialize Chroma, the database is stored in different places:

### A. In-Memory Mode (RAM)
If you initialize Chroma without a `persist_directory`:
*   **Where it is saved**: The database exists entirely in the server's **RAM (volatile memory)**.
*   **How to see it**: Since it is not written to a file, you cannot view it in an explorer. You can only inspect it by running code in your Python script:
    ```python
    # Count the items loaded in RAM
    print(vector_store._collection.count())
    # Retrieve first item
    print(vector_store._collection.peek(limit=1))
    ```
*   **Lifespan**: It is destroyed immediately when your Python process terminates.

### B. Local Persistent Mode (Disk Storage)
If you initialize Chroma with a `persist_directory`:
```python
vector_store = Chroma(
    collection_name="my_collection",
    embedding_function=embeddings,
    persist_directory="c:/Coding/langChain/12_vector_store/chroma_db"
)
```
*   **Where it is saved**: It is written directly to files on your hard drive in the folder specified (e.g. `chroma_db/`).
*   **Directory Structure**: If you open the folder, you will see:
    ```text
    chroma_db/
    ├── chroma.sqlite3        <-- Metadata, collections, and document indexing parameters
    └── a3b8418f-d122/...     <-- Folder containing binary parquet files storing raw vectors
    ```
*   **How to see/inspect it**:
    1.  **SQLite Tooling**: Since the database metadata is stored in standard SQLite, you can open the `chroma.sqlite3` file using any SQLite visualizer (like **DB Browser for SQLite**, **DBeaver**, or the **VS Code SQLite Viewer Extension**).
    2.  **Inspect Tables**: Opening the SQLite file reveals internal schema tables like:
        *   `collections`: Holds the collections configurations.
        *   `embeddings`: Tracks document IDs, original text mapping, and metadata filters.

