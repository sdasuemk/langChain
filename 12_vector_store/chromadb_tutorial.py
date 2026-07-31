# -*- coding: utf-8 -*-
# Generated from chromadb_tutorial.ipynb

# %% [markdown]
# # ChromaDB Vector Store Tutorial in LangChain
# This Google Colab compatible notebook demonstrates how to build, query, and manage a **ChromaDB Vector Store** using LangChain.
# 
# ### Case Study: IPL Team Rosters
# We will load roster documents for **10 IPL (Indian Premier League) Teams** and perform various operations:
# 1. **Embedding & Store Creation** (using Hugging Face embeddings and in-memory Chroma)
# 2. **Adding Documents** (with custom metadata and unique IDs)
# 3. **Similarity Search** (Standard vs. With-Score distance evaluations)
# 4. **Metadata Filtering** (restricting search queries by team/city parameters)
# 5. **Inspecting DB Internals** (counts and collection peering)
# 6. **Direct Document Retrieval** (using `.get()` method)
# 7. **CRUD updates & deletions**.

# %% [markdown]
# ### Step 1: Install Dependencies
# Run this cell to install the latest integration packages for Chroma and Hugging Face.

# %%
# Install dependencies (Colab specific)
# !pip install -q langchain-chroma langchain-huggingface python-dotenv


# %% [markdown]
# ### Step 2: Load Environment Variables
# To use Hugging Face API hosted models, make sure your `.env` contains `HUGGINGFACEHUB_API_TOKEN` or set it in your environment.
# 
# *Note: We use `find_dotenv()` to locate the `.env` file at the root folder when executing code inside subfolders. If the key is missing (for example, when running inside Google Colab), this cell will securely prompt you to input it.*

# %%
import os
from dotenv import load_dotenv, find_dotenv
import getpass

# Search parent folders to find the root .env file and load it
load_dotenv(find_dotenv())

# Secure fallback check to prompt for the API key if missing from environment variables
if not os.environ.get("HUGGINGFACEHUB_API_TOKEN"):
    print("HUGGINGFACEHUB_API_TOKEN not found in environment.")
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = getpass.getpass("Please paste your Hugging Face API token: ")
else:
    print("HUGGINGFACEHUB_API_TOKEN successfully loaded.")


# %% [markdown]
# ### Step 3: Define the IPL Roster Documents
# We construct 10 separate documents, one for each IPL franchise team. We assign metadata dictionaries (`team` and `city`) and a unique document ID to simplify target editing later.

# %%
from langchain_core.documents import Document

# Defined roster details for 10 IPL Teams
raw_rosters = [
    {
        "id": "mi_001",
        "content": "Mumbai Indians (MI): Rohit Sharma is a legendary opening batsman. Jasprit Bumrah is a world-class fast bowler.",
        "metadata": {"team": "MI", "city": "Mumbai"}
    },
    {
        "id": "csk_001",
        "content": "Chennai Super Kings (CSK): MS Dhoni is the legendary captain and finisher. Ravindra Jadeja is a world-class spin all-rounder.",
        "metadata": {"team": "CSK", "city": "Chennai"}
    },
    {
        "id": "rcb_001",
        "content": "Royal Challengers Bengaluru (RCB): Virat Kohli is an elite run-machine. Glenn Maxwell is an explosive batsman and spinner.",
        "metadata": {"team": "RCB", "city": "Bengaluru"}
    },
    {
        "id": "kkr_001",
        "content": "Kolkata Knight Riders (KKR): Shreyas Iyer is a solid middle-order batsman. Sunil Narine is a mysterious spinner and opener.",
        "metadata": {"team": "KKR", "city": "Kolkata"}
    },
    {
        "id": "rr_001",
        "content": "Rajasthan Royals (RR): Sanju Samson is a fluent wicketkeeper-batsman. Jos Buttler is an explosive English batsman.",
        "metadata": {"team": "RR", "city": "Jaipur"}
    },
    {
        "id": "gt_001",
        "content": "Gujarat Titans (GT): Shubman Gill is a classy young batsman. Rashid Khan is a legendary Afghanistan leg-spinner.",
        "metadata": {"team": "GT", "city": "Ahmedabad"}
    },
    {
        "id": "lsg_001",
        "content": "Lucknow Super Giants (LSG): KL Rahul is a technically sound batsman. Nicholas Pooran is a destructive middle-order finisher.",
        "metadata": {"team": "LSG", "city": "Lucknow"}
    },
    {
        "id": "pbks_001",
        "content": "Punjab Kings (PBKS): Shikhar Dhawan is a veteran opener. Sam Curran is an English bowling all-rounder.",
        "metadata": {"team": "PBKS", "city": "Mohali"}
    },
    {
        "id": "srh_001",
        "content": "Sunrisers Hyderabad (SRH): Pat Cummins is the fast-bowling captain. Travis Head is a dangerous Australian opener.",
        "metadata": {"team": "SRH", "city": "Hyderabad"}
    },
    {
        "id": "dc_001",
        "content": "Delhi Capitals (DC): Rishabh Pant is a dynamic wicketkeeper-batsman. Axar Patel is a reliable left-arm spin all-rounder.",
        "metadata": {"team": "DC", "city": "Delhi"}
    }
]

# Create Document objects
documents = [
    Document(page_content=item["content"], metadata=item["metadata"], id=item["id"])
    for item in raw_rosters
]
print(f"Created {len(documents)} document objects.")


# %% [markdown]
# ### Step 4: Initialize Chroma and Add Documents
# We will initialize the **API-based** embedding wrapper (`HuggingFaceEndpointEmbeddings`). This makes HTTP calls to Hugging Face Inference API to compute embeddings in the cloud without requiring local RAM or GPU compute.
# 
# *(Alternatively, commented out below is `HuggingFaceEmbeddings` which downloads model weights and executes locally on your CPU/GPU).* 

# %%
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEndpointEmbeddings

print("Initializing API-based Hugging Face Embeddings...")
# This client sends queries to Hugging Face's remote Inference endpoints (API-based)
embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2"
)

# --- LOCAL CPU/GPU ALTERNATIVE (No API keys needed but downloads model weights locally) ---
# from langchain_huggingface import HuggingFaceEmbeddings
# embeddings = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2"
# )
# -----------------------------------------------------------------------------------------

print("Creating Chroma Vector Store in-memory...")
vector_store = Chroma(
    collection_name="ipl_teams",
    embedding_function=embeddings
)

print("Adding IPL documents to the vector store...")
vector_store.add_documents(documents)
print("Documents successfully embedded and stored.")


# %% [markdown]
# ### Step 5: Similarity Search
# Let's query the database to find relevant documents. Notice how semantic search successfully matches players even without explicit team name clues in the query.

# %%
# Querying for Virat Kohli - should return RCB
query_1 = "Who is the run-machine from Bengaluru?"
results_1 = vector_store.similarity_search(query_1, k=1)

print(f"Query: {query_1}")
print(f"Best Match:\n{results_1[0].page_content}")


# %% [markdown]
# ### Step 6: Similarity Search with Score
# Retrieving scores allows us to see how confident the database is about the match. In Chroma, the score represents L2 distance. **Lower distance scores mean closer similarity**.

# %%
query_2 = "Wicketkeeper captain MS Dhoni"
results_with_scores = vector_store.similarity_search_with_score(query_2, k=2)

print(f"Query: {query_2}\n")
for doc, score in results_with_scores:
    print(f"Distance Score (L2): {score:.4f}")
    print(f"Team: {doc.metadata['team']}")
    print(f"Content: {doc.page_content}")
    print("-" * 50)


# %% [markdown]
# ### Step 7: Metadata Filtering
# If we search for "Australian opener", we might get sunrisers Hyderabad (`SRH` - Travis Head). But if we want to restrict search criteria to a specific city or team, we pass a `filter` dictionary.

# %%
# Search for opener, but restrict target metadata to Chennai city
query_3 = "Dangerous opening batsman"
filtered_results = vector_store.similarity_search(
    query_3,
    k=1,
    filter={"city": "Chennai"}
)

print(f"Query with Filter (city: Chennai): {query_3}")
print(f"Match: {filtered_results[0].page_content}")


# %% [markdown]
# ### Step 8: View Database Internals (Inspection)
# 
# #### **Where is this Vector Store saved?**
# 1. **RAM (In-Memory)**: Because we did not supply a `persist_directory` parameter to `Chroma()`, the database exists **only in your computer's RAM**. This means it is ephemeral, cannot be seen in your file explorer, and vanishes when the Python process finishes.
# 2. **On Disk (Persistent)**: To write the database to a directory on your hard disk, configure a storage path like:
#    ```python
#    vector_store = Chroma(collection_name="ipl_teams", embedding_function=embeddings, persist_directory="./chroma_db")
#    ```
#    This creates a directory called `chroma_db/` containing a `chroma.sqlite3` file and folders of parquet files. You can inspect `chroma.sqlite3` using SQLite visualizer applications (e.g. DB Browser for SQLite or VS Code SQLite Viewers).
# 
# #### **How to peek inside RAM stores?**
# Since we cannot look at files in RAM, we write Python queries to peek at IDs and count vectors.

# %%
collection = vector_store._collection

print(f"Active Vectors Count inside RAM: {collection.count()}")

# Peek at the first record's ID and Metadata
peek_data = collection.peek(limit=1)
print("Peek ID:", peek_data["ids"])
print("Peek Metadata:", peek_data["metadatas"])


# %% [markdown]
# ### Step 8.5: Direct Document Retrieval using `.get()`
# Instead of calculating mathematical similarities (which requires a query embedding and distances calculations), you can retrieve documents directly using the `.get()` method by passing ID keys or metadata parameter filters.

# %%
# 1. Retrieve a document directly using its ID
direct_doc = vector_store.get(ids=["csk_001"])
print("Direct ID search matching text:", direct_doc["documents"])

# 2. Retrieve documents using metadata parameter constraints
metadata_fetch = vector_store.get(where={"team": "RCB"})
print("Metadata filtered matching text:", metadata_fetch["documents"])

# 3. Retrieve documents including raw vector representations (embeddings)
full_data = vector_store.get(ids=["mi_001"], include=["documents", "metadatas", "embeddings"])
print("Vector dimensions stored:", len(full_data["embeddings"][0]))
print("Raw vector sample:", full_data["embeddings"][0][:5])


# %% [markdown]
# ### Step 9: Updating Documents
# Let's update Delhi Capitals (`DC` / `dc_001`) with a new roster text. The `update_documents` method in LangChain requires both the `ids` list and the `documents` list as arguments.

# %%
updated_doc = Document(
    page_content="Delhi Capitals (DC): Rishabh Pant is the dynamic captain. Jake Fraser-McGurk is the new explosive opener.",
    metadata={"team": "DC", "city": "Delhi"},
    id="dc_001"
)

print("Updating document dc_001...")
vector_store.update_documents(ids=["dc_001"], documents=[updated_doc])

# Querying the update to check if the opener changed
search_after_update = vector_store.similarity_search("explosive opener for Delhi", k=1)
print(f"\nSearch result after update:\n{search_after_update[0].page_content}")


# %% [markdown]
# ### Step 10: Deleting Documents
# Finally, we can delete documents from our vector database using their Unique IDs. We will delete the Mumbai Indians (`mi_001`) roster and check the database count.

# %%
print(f"Count before deletion: {collection.count()}")

print("Deleting Mumbai Indians (mi_001)...")
vector_store.delete(ids=["mi_001"])

print(f"Count after deletion: {collection.count()}")

