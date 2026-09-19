# -*- coding: utf-8 -*-
# Generated from pinecone_tutorial.ipynb

# %% [markdown]
# # Pinecone Cloud Vector Database Tutorial in LangChain
# This Google Colab compatible notebook demonstrates how to construct, query, filter, and manage a **Pinecone Cloud Vector Database** using LangChain.
# 
# ### Case Study: IPL Team Rosters
# We will load roster documents for **10 IPL (Indian Premier League) Teams** and perform various operations:
# 1. **Embedding & Store Creation** (using Hugging Face embeddings and cloud Pinecone index)
# 2. **Similarity Search** (Standard vs. Cosine distance evaluations)
# 3. **Metadata Filtering** (restricting search queries by team/city parameters)
# 4. **CRUD updates & deletions**.

# %% [markdown]
# ### Step 1: Install Dependencies
# Run this cell to install the latest integration packages for Pinecone and Hugging Face.

# %%
# Install dependencies (Colab specific)
# !pip install -q langchain-pinecone langchain-huggingface python-dotenv


# %% [markdown]
# ### Step 2: Load Credentials & API Keys
# To use Pinecone and Hugging Face, we require `PINECONE_API_KEY` and `HUGGINGFACEHUB_API_TOKEN` set in our environment.
# 
# *Note: We use `find_dotenv()` to load locally. If keys are missing (such as in Colab), this cell securely prompts you to paste them.*

# %%
import os
from dotenv import load_dotenv, find_dotenv
import getpass

# Load .env file
load_dotenv(find_dotenv())

# Verify Pinecone API Key
if not os.environ.get("PINECONE_API_KEY"):
    print("PINECONE_API_KEY not found.")
    os.environ["PINECONE_API_KEY"] = getpass.getpass("Please paste your Pinecone API token: ")

# Verify Hugging Face API Token
if not os.environ.get("HUGGINGFACEHUB_API_TOKEN"):
    print("HUGGINGFACEHUB_API_TOKEN not found.")
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = getpass.getpass("Please paste your Hugging Face API token: ")
else:
    print("API keys successfully configured.")


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

documents = [
    Document(page_content=item["content"], metadata=item["metadata"], id=item["id"])
    for item in raw_rosters
]
print(f"Compiled {len(documents)} document objects.")


# %% [markdown]
# ### Step 4: Initialize Pinecone and Upload Documents
# We will configure the embeddings client and push documents up to the Pinecone index. 
# 
# *Make sure you have created an index in Pinecone matching your model's dimensions (e.g. 384 dimensions for all-MiniLM-L6-v2) before running.*

# %%
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEndpointEmbeddings

print("Initializing Hugging Face Embeddings...")
embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=os.environ.get("HUGGINGFACEHUB_API_TOKEN")
)

# Please supply your active Pinecone index name
index_name = "ipl-rosters-index"

print(f"Connecting to Pinecone index '{index_name}' and indexing documents...")
# Under the hood, this embeds the text and upserts raw vectors to the cloud
"""
vector_store = PineconeVectorStore.from_documents(
    documents=documents,
    embedding=embeddings,
    index_name=index_name
)
print("Upload complete.")
"""


# %% [markdown]
# ### Step 5: Similarity Search
# Query the database to find relevant documents. For demonstration, we assume your index is populated.

# %%
query = "Who is the run-machine from Bengaluru?"
print(f"Query: {query}")
"""
results = vector_store.similarity_search(query, k=1)
print(f"Best Match:\n{results[0].page_content}")
"""


# %% [markdown]
# ### Step 6: Similarity Search with Score
# In Pinecone, scores represent Cosine similarity. **Higher values (closer to 1.0) mean closer matches**.

# %%
query_2 = "Wicketkeeper captain MS Dhoni"
print(f"Query: {query_2}\n")
"""
results_with_scores = vector_store.similarity_search_with_score(query_2, k=2)
for doc, score in results_with_scores:
    print(f"Cosine Similarity Score: {score:.4f}")
    print(f"Content: {doc.page_content}")
    print("-" * 50)
"""


# %% [markdown]
# ### Step 7: Metadata Filtering
# Isolate query targets using Pinecone's server-side metadata constraints.

# %%
query_3 = "Dangerous opening batsman"
print(f"Querying with Filter (city: Chennai): {query_3}")
"""
filtered_results = vector_store.similarity_search(
    query_3,
    k=1,
    filter={"city": "Chennai"}
)
print(f"Match: {filtered_results[0].page_content}")
"""


# %% [markdown]
# ### Step 8: Updating and Deleting Documents
# Use unique IDs to edit or drop index entries on the Pinecone service.

# %%
updated_doc = Document(
    page_content="Delhi Capitals (DC): Rishabh Pant is the dynamic captain. Jake Fraser-McGurk is the new explosive opener.",
    metadata={"team": "DC", "city": "Delhi"}
)

# 1. Update/Add document with targeted ID
print("Updating document dc_001...")
"""
vector_store.add_documents(documents=[updated_doc], ids=["dc_001"])
"""

# 2. Delete index item
print("Deleting Mumbai Indians index item (mi_001)...")
"""
vector_store.delete(ids=["mi_001"])
"""

