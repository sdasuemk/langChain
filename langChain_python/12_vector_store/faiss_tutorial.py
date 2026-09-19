# -*- coding: utf-8 -*-
# Generated from faiss_tutorial.ipynb

# %% [markdown]
# # FAISS (Facebook AI Similarity Search) Tutorial in LangChain
# This Google Colab compatible notebook demonstrates how to construct, serialize, query, and manage a **FAISS Vector Store** using LangChain.
# 
# ### Case Study: IPL Team Rosters
# We will load roster documents for **10 IPL (Indian Premier League) Teams** and perform various operations:
# 1. **Embedding & Store Creation** (using Hugging Face embeddings and local FAISS in-memory index)
# 2. **Serialization** (saving the index binary directly to disk and reloading it)
# 3. **Similarity Search** (Standard vs. L2 distance evaluations)
# 4. **Direct Document Retrieval** (using `.get()` constraints)
# 5. **CRUD updates & deletions**.

# %% [markdown]
# ### Step 1: Install Dependencies
# Run this cell to install the latest integration packages for FAISS and Hugging Face.

# %%
# Install dependencies (Colab specific)
# !pip install -q langchain-community faiss-cpu langchain-huggingface python-dotenv


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
# ### Step 4: Initialize FAISS and Add Documents
# We will initialize the **API-based** embedding wrapper (`HuggingFaceEndpointEmbeddings`). Since FAISS runs entirely in-memory inside the Python runtime process, this creates a vector search index structure in RAM.

# %%
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpointEmbeddings

print("Initializing API-based Hugging Face Embeddings...")
embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=os.environ.get("HUGGINGFACEHUB_API_TOKEN")
)

print("Creating FAISS vector index in-memory...")
vector_store = FAISS.from_documents(documents, embeddings)
print("Documents successfully embedded and indexed in FAISS memory.")


# %% [markdown]
# ### Step 5: Save & Load FAISS Index (Serialization)
# Because FAISS operates in RAM, it disappears when the notebook session closes. We serialize and write the index files (`index.faiss` and `index.pkl`) directly to disk.

# %%
# Save FAISS to local files
persist_dir = "faiss_index_store"
print(f"Saving FAISS index binaries to folder: '{persist_dir}'")
vector_store.save_local(persist_dir)

# Reload the store from disk files
print("Reloading index binary from disk...")
loaded_store = FAISS.load_local(
    persist_dir,
    embeddings,
    allow_dangerous_deserialization=True  # Required to safely unpack pickle objects
)
print("FAISS store successfully reloaded.")


# %% [markdown]
# ### Step 6: Similarity Search
# Let's query our loaded FAISS store to find relevant documents.

# %%
# Querying for Virat Kohli - should return RCB
query_1 = "Who is the run-machine from Bengaluru?"
results_1 = loaded_store.similarity_search(query_1, k=1)

print(f"Query: {query_1}")
print(f"Best Match:\n{results_1[0].page_content}")


# %% [markdown]
# ### Step 7: Similarity Search with Score
# Retrieve distance scores. In FAISS, the default metric represents **Euclidean (L2) Distance** or Cosine Distance. **Lower values mean closer vectors (more similar)**.

# %%
query_2 = "Wicketkeeper captain MS Dhoni"
results_with_scores = loaded_store.similarity_search_with_score(query_2, k=2)

print(f"Query: {query_2}\n")
for doc, score in results_with_scores:
    print(f"L2 Distance Score: {score:.4f}")
    print(f"Content: {doc.page_content}")
    print("-" * 50)


# %% [markdown]
# ### Step 8: Updating Documents
# FAISS does not have a native `update_documents()` method. To overwrite a document in FAISS, we delete the document using its ID first, then add the new document under the same ID. Because delete can raise error on older package versions, we implement a rebuild fallback.

# %%
updated_doc = Document(
    page_content="Delhi Capitals (DC): Rishabh Pant is the captain. Jake Fraser-McGurk is the new explosive opener.",
    metadata={"team": "DC", "city": "Delhi"},
    id="dc_001"
)

print("Updating document dc_001...")
try:
    loaded_store.delete(ids=["dc_001"])
except Exception:
    # Rebuild fallback: filter out deleted doc and recreate the store from the docstore
    remaining_docs = [doc for doc_id, doc in loaded_store.docstore._dict.items() if doc_id not in ["dc_001"]]
    loaded_store = FAISS.from_documents(remaining_docs, embeddings)

loaded_store.add_documents(documents=[updated_doc], ids=["dc_001"])

# Search after update
search_after_update = loaded_store.similarity_search("explosive opener for Delhi", k=1)
print(f"\nSearch result after update:\n{search_after_update[0].page_content}")


# %% [markdown]
# ### Step 9: Deleting Documents
# We can delete vectors from our FAISS index using their Unique IDs. Since `delete` can raise `NotImplementedError` in older langchain-community versions, we wrapper-wrap it in a try-except rebuild fallback.

# %%
print("Deleting Mumbai Indians index element (mi_001)...")
try:
    loaded_store.delete(ids=["mi_001"])
except Exception:
    # Fallback rebuild when delete is not natively supported by the subclass wrapper
    remaining_docs = [doc for doc_id, doc in loaded_store.docstore._dict.items() if doc_id not in ["mi_001"]]
    loaded_store = FAISS.from_documents(remaining_docs, embeddings)
print("Deletion complete.")

