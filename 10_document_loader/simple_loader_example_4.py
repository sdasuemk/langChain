"""
LangChain Simple Document Loaders Example
=========================================
This file demonstrates the simplest way to use LangChain's Document Loaders 
to read different file formats (Txt and PDF) and convert them into 
standardized LangChain Document objects.

Key Concepts:
- Document: The standard data model in LangChain consisting of 'page_content' (text) 
  and 'metadata' (dictionary of source information).
- TextLoader: Loads raw text from a text file (.txt, .md, etc.).
- PyPDFLoader: Loads page-by-page text from a PDF file (.pdf).
"""

from langchain_community.document_loaders import TextLoader, PyPDFLoader

# ---------------------------------------------------------
# EXAMPLE 1: Loading a Text File (.txt)
# ---------------------------------------------------------
print("--- Loading Text File ---")
# Initialize the loader with the text file path and encoding
text_loader = TextLoader("asset/langchain_poem.txt", encoding="utf-8")

# Load the file into a list of Document objects
text_docs = text_loader.load()

# Inspect the loaded text document
print(f"Number of documents loaded: {len(text_docs)}")
print(f"Metadata: {text_docs[0].metadata}")
print(f"Sample Content:\n{text_docs[0].page_content[:150]}...")
print("=" * 60)

# ---------------------------------------------------------
# EXAMPLE 2: Loading a PDF File (.pdf)
# ---------------------------------------------------------
print("\n--- Loading PDF File ---")
# Initialize the loader with the PDF path. It loads page by page automatically.
pdf_loader = PyPDFLoader("asset/llm-book.pdf")

# Load the PDF pages into a list of Document objects
pdf_pages = pdf_loader.load()

# Inspect the loaded pages
print(f"Number of pages loaded: {len(pdf_pages)}")
print(f"Metadata of page 1: {pdf_pages[0].metadata}")
print(f"Sample Content from page 1:\n{pdf_pages[0].page_content[:150]}...")
