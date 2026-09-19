"""
LangChain Semantic Meaning-Based Splitting Tutorial
====================================================
This file demonstrates how to use the SemanticChunker to split text 
based on shifts in semantic meaning/context, rather than arbitrary character 
lengths or standard separators.

This tutorial covers two approaches for loading embeddings:
1. API-Based Embeddings (used live): Sends texts to a hosted endpoint.
2. Local-Based Embeddings (documented in comments): Runs the model locally.

Prerequisites:
Make sure you have the experimental and embedding packages installed:
$ pip install langchain-experimental langchain-huggingface

Make sure you have your Hugging Face API key configured in a `.env` file as:
HUGGINGFACEHUB_API_TOKEN=your_token_here

Key Concepts:
- SemanticChunker: A splitter that groups sentences together if they are semantically 
  similar, and splits them when a transition in topic (semantic shift) is detected.
- API-Based Embeddings: Uses HuggingFaceEndpointEmbeddings to compute text embeddings 
  via Hugging Face hosted inference endpoints, eliminating local model downloads.
- Breakpoint Threshold Types:
  - 'percentile' (default): Splits at distances greater than the specified percentile.
  - 'standard_deviation': Splits at distances greater than a standard deviation threshold.
  - 'interquartile': Splits based on the interquartile range of similarities.
"""

import os
from dotenv import load_dotenv
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
# For local embeddings (Option B), you would import:
# from langchain_huggingface import HuggingFaceEmbeddings

# ---------------------------------------------------------
# STEP 1: Environment Setup
# ---------------------------------------------------------
# Load environment variables. This loads the `HUGGINGFACEHUB_API_TOKEN` 
# required to authenticate with the Hugging Face Inference API.
load_dotenv()

# ---------------------------------------------------------
# STEP 2: Initialize the Embeddings Model (API-Based vs Local)
# ---------------------------------------------------------

# Option A: API-Based (Used Live)
# We use HuggingFaceEndpointEmbeddings to send embedding requests to Hugging Face APIs.
# By specifying the hosted model repository, this executes the model online and returns 
# sentence vectors without downloading or running the model locally. It uses the API key 
# loaded in STEP 1.
print("Initializing HuggingFaceEndpointEmbeddings via Hosted API...")
embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2", # Hosted model ID on Hugging Face
    task="feature-extraction"                       # Task type required for embeddings
)

# Option B: Local-Based (Commented Out for Future Reference)
# If you prefer to run the model entirely on your local machine instead of making API calls:
# 1. Install sentence-transformers: pip install sentence-transformers
# 2. Uncomment the following lines:
#
# print("Initializing HuggingFaceEmbeddings model locally...")
# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# ---------------------------------------------------------
# STEP 3: Define the Raw Input Text
# ---------------------------------------------------------
# We create a sample text containing three completely different topics:
# 1. Puppies and dogs
# 2. Database indexing and querying
# 3. Cooking spaghetti
#
# A normal character splitter might split in the middle of these topic sections.
# The SemanticChunker will detect the transitions in topic and split accordingly.
text = """
Puppies are extremely cute animals that bring joy to many families. 
They need regular exercise, a balanced diet, and plenty of love. 
Training a dog requires patience and consistent positive reinforcement.

Relational databases store data in tables containing rows and columns.
To speed up data retrieval, developers often create indexes on frequently searched columns.
Using database indexes speeds up queries but can slow down write operations.

To cook a perfect plate of spaghetti, start by boiling a large pot of salted water.
Add the pasta and cook it until it is al dente, stirring occasionally.
Toss the hot pasta with a fresh marinara sauce and garnish with fresh basil and parmesan cheese.
"""

# ---------------------------------------------------------
# STEP 4: Initialize the SemanticChunker
# ---------------------------------------------------------
# The SemanticChunker splits text by finding the difference in embedding vectors 
# between consecutive sentences.
# We set `breakpoint_threshold_type="percentile"` to split at points where the 
# distance is in the top percentile of all distances.
text_splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile"  # Other options: 'standard_deviation', 'interquartile'
)

# ---------------------------------------------------------
# STEP 5: Split the Text into Chunks
# ---------------------------------------------------------
# Split the document text. The output is a list of Document objects.
print("Splitting text based on semantic similarity...")
docs = text_splitter.create_documents([text])

# ---------------------------------------------------------
# STEP 6: Inspect the Generated Chunks
# ---------------------------------------------------------
# Notice how the sentences about dogs, databases, and spaghetti are grouped 
# into their own separate chunks.
for i, doc in enumerate(docs):
    print(f"\n--- Chunk {i+1} ---")
    print(doc.page_content.strip())
    print("-" * 50)
