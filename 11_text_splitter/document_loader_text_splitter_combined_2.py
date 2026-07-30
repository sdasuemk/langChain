"""
LangChain Document Loader & Text Splitter Integration Tutorial
=============================================================
This file demonstrates how to load a PDF document using PyPDFLoader, 
inspect its content and metadata, and split it into smaller document 
chunks using CharacterTextSplitter.

Key Concepts:
- PyPDFLoader: Loads a local PDF file and represents its pages as a list of Document objects.
- Document: A LangChain object containing `page_content` (the text) and `metadata` (dictionary with sources, pages, etc.).
- split_documents: Splits a list of Document objects into smaller chunks while preserving their source metadata.
"""

from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# ---------------------------------------------------------
# STEP 1: Load the PDF Document
# ---------------------------------------------------------
# We initialize PyPDFLoader with the path to a PDF file.
# Calling `.load()` extracts all pages from the PDF.
# Each page is converted into a separate LangChain Document object.
loader = PyPDFLoader("asset/llm-book.pdf")
docs = loader.load()

# ---------------------------------------------------------
# STEP 2: Inspect the Loaded Document
# ---------------------------------------------------------
# Let's inspect the first loaded page to understand the Document structure.
# A Document has two main attributes:
# - page_content: The text content of the page.
# - metadata: Metadata about the page (e.g., source file name, page number).
print("--- Loaded Document (First Page) ---")
print(docs[0])

# ---------------------------------------------------------
# STEP 3: Initialize the CharacterTextSplitter
# ---------------------------------------------------------
# We configure the text splitter. Note that we set separator="" here.
# With an empty separator, the splitter will force a split once the 
# chunk_size threshold (100 characters) is reached.
text_splitter = CharacterTextSplitter(
    separator="",              # No specific separator; split strictly on length boundary
    chunk_size=100,            # Target character length for each document chunk
    chunk_overlap=20,          # Overlapping characters to keep continuity
    length_function=len,       # Function used to calculate size (default python len())
    is_separator_regex=False,  # Treat separator as raw string
)

# ---------------------------------------------------------
# STEP 4: Split the Document Objects
# ---------------------------------------------------------
# Instead of splitting a raw string, we use `.split_documents()` 
# to split our list of Document objects.
# This automatically carries over metadata (e.g. source, page index) to each split chunk.
chunks = text_splitter.split_documents(docs)

# ---------------------------------------------------------
# STEP 5: Inspect the Generated Document Chunks
# ---------------------------------------------------------
# Print the chunks with their index and see how the original content was chunked.
for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i+1} ---")
    print(chunk)

print("\n--- Summary ---")
print("Total document chunks generated:", len(chunks))