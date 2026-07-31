# -*- coding: utf-8 -*-
# Generated from milvus_tutorial.ipynb

# %% [markdown]
# # Milvus Enterprise Vector Database Tutorial in LangChain
# This Google Colab compatible notebook demonstrates how to construct, query, filter, and manage a **Milvus Vector Database** using LangChain and Milvus Lite.
# 
# ### Case Study: IPL Team Rosters
# We will load roster documents for **10 IPL (Indian Premier League) Teams** and perform various operations:
# 1. **Embedding & Store Creation** (using Hugging Face embeddings and local sqlite-based Milvus Lite file)
# 2. **Similarity Search** (Standard vs. Score evaluations)
# 3. **Metadata Filtering** (restricting search queries by team/city parameters using expression strings)
# 4. **CRUD updates & deletions**.

# %% [markdown]
# ### Step 1: Install Dependencies
# Run this cell to install the latest integration packages for Milvus and Hugging Face.

# %%
# Install dependencies (Colab specific)
# !pip install -q langchain-milvus pymilvus langchain-huggingface python-dotenv


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
# ### Step 4: Initialize Milvus Lite and Add Documents
# We will initialize the **API-based** embedding wrapper (`HuggingFaceEndpointEmbeddings`). We instantiate Milvus Lite using local database path `./milvus_demo.db` so it executes entirely inside python process files.

# %%
from langchain_milvus import Milvus
from langchain_huggingface import HuggingFaceEndpointEmbeddings

print("Initializing API-based Hugging Face Embeddings...")
embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=os.environ.get("HUGGINGFACEHUB_API_TOKEN")
)

print("Creating Milvus Lite Local Store...")
vector_store = Milvus.from_documents(
    documents=documents,
    embedding=embeddings,
    connection_args={"uri": "./milvus_demo.db"},
    collection_name="ipl_teams"
)
print("Milvus Lite successfully embedded and indexed vectors.")


# %% [markdown]
# ### Step 5: Similarity Search
# Let's query the database to find relevant documents.

# %%
query_1 = "Who is the run-machine from Bengaluru?"
results_1 = vector_store.similarity_search(query_1, k=1)

print(f"Query: {query_1}")
print(f"Best Match:\n{results_1[0].page_content}")


# %% [markdown]
# ### Step 6: Similarity Search with Score
# Retrieving scores allows us to see how confident the database is about the match.

# %%
query_2 = "Wicketkeeper captain MS Dhoni"
results_with_scores = vector_store.similarity_search_with_score(query_2, k=2)

print(f"Query: {query_2}\n")
for doc, score in results_with_scores:
    print(f"Similarity Score: {score:.4f}")
    print(f"Team: {doc.metadata['team']}")
    print(f"Content: {doc.page_content}")
    print("-" * 50)


# %% [markdown]
# ### Step 7: Metadata Filtering
# Milvus supports metadata isolating filters using its Boolean Expression Language syntax string.

# %%
query_3 = "Dangerous opening batsman"
filtered_results = vector_store.similarity_search(
    query_3,
    k=1,
    expr='city == "Chennai"'
)

print(f"Query with Filter (city: Chennai): {query_3}")
print(f"Match: {filtered_results[0].page_content}")


# %% [markdown]
# ### Step 8: Updating Documents
# To edit a document, pass the updated `Document` list with their original IDs.

# %%
updated_doc = Document(
    page_content="Delhi Capitals (DC): Rishabh Pant is the dynamic captain. Jake Fraser-McGurk is the new explosive opener.",
    metadata={"team": "DC", "city": "Delhi"}
)

print("Updating document dc_001...")
vector_store.add_documents(documents=[updated_doc], ids=["dc_001"])

# Querying the update to check if the opener changed
search_after_update = vector_store.similarity_search("explosive opener for Delhi", k=1)
print(f"\nSearch result after update:\n{search_after_update[0].page_content}")


# %% [markdown]
# ### Step 9: Deleting Documents
# Finally, delete documents from our vector database using their Unique IDs.

# %%
print("Deleting Mumbai Indians (mi_001)...")
vector_store.delete(ids=["mi_001"])
print("Deletion complete.")

