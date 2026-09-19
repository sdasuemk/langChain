"""
LangChain DirectoryLoader Tutorial
==================================
This file demonstrates how to use the DirectoryLoader to load multiple 
documents from a directory concurrently using glob patterns.

Key Concepts:
- DirectoryLoader: Reads all files matching a glob pattern inside a directory.
- loader_cls: Specifies the child loader class used to parse individual files 
  (e.g., TextLoader, PyPDFLoader, CSVLoader).
- glob: Glob patterns for selecting target file types (e.g., "*.txt", "**/*.py").
"""

from langchain_community.document_loaders import DirectoryLoader, TextLoader

# ---------------------------------------------------------
# STEP 1: Initialize the DirectoryLoader
# ---------------------------------------------------------
# We target the 'asset' directory and load all text files (*.txt).
# We explicitly configure it to use 'TextLoader' as the parser class.
print("Initializing DirectoryLoader for text files in 'asset' directory...")
loader = DirectoryLoader(
    path="asset",
    glob="*.txt",
    loader_cls=TextLoader,
    show_progress=True,
    use_multithreading=True
)

# ---------------------------------------------------------
# STEP 2: Load the Documents
# ---------------------------------------------------------
# `.load()` returns a list of Document objects from all matched files.
docs = loader.load()

# ---------------------------------------------------------
# STEP 3: Inspect the Loaded Documents
# ---------------------------------------------------------
print(f"\nSuccessfully loaded {len(docs)} document(s) from directory.\n")

for i, doc in enumerate(docs):
    print(f"--- Document {i+1} ---")
    print(f"Source: {doc.metadata.get('source')}")
    print(f"Content Snippet:\n{doc.page_content[:150]}...")
    print("-" * 50)
