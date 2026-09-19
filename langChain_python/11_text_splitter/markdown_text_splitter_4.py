"""
LangChain MarkdownHeaderTextSplitter Tutorial
=============================================
This file demonstrates how to use the MarkdownHeaderTextSplitter to split 
markdown text based on specified headers. It preserves the logical structure 
of the document by adding header information directly to the metadata of each chunk.

Key Concepts:
- MarkdownHeaderTextSplitter: A structure-aware text splitter that splits text 
  along markdown headers (e.g., #, ##, ###) rather than character length.
- Metadata Enrichment: Header values are injected as metadata keys in the output 
  documents, allowing vector stores to easily filter or weight chunks based on section.
"""

from langchain_text_splitters import MarkdownHeaderTextSplitter

# ---------------------------------------------------------
# STEP 1: Define the Markdown Document Text
# ---------------------------------------------------------
# We define a sample markdown document with nested headers, list items, and text.
markdown_document = """# Intro

Welcome to the LangChain tutorial. This is the introduction section.

## Getting Started

To install LangChain, run:
```bash
pip install langchain
```

### Advanced Usage

For advanced workflows, you can chain multiple prompts and parsers.

# Conclusion

We hope this tutorial helps you split your documents efficiently!
"""

# ---------------------------------------------------------
# STEP 2: Define the Headers to Split On
# ---------------------------------------------------------
# We specify which Markdown headers to split on and what metadata keys to map them to.
# For example, text under "# Intro" will get the metadata {"Header 1": "Intro"}.
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

# ---------------------------------------------------------
# STEP 3: Initialize the MarkdownHeaderTextSplitter
# ---------------------------------------------------------
# We initialize the splitter using the configured headers.
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

# ---------------------------------------------------------
# STEP 4: Split the Markdown Text
# ---------------------------------------------------------
# Use `.split_text()` to split the markdown document.
# This returns a list of Document objects with their respective text and header metadata.
docs = markdown_splitter.split_text(markdown_document)

# ---------------------------------------------------------
# STEP 5: Inspect the Generated Chunks
# ---------------------------------------------------------
# Print each chunk content and its metadata.
# Notice how the parent headers are tracked inside the metadata dictionary.
for i, doc in enumerate(docs):
    print(f"--- Document Chunk {i+1} ---")
    print(f"Content:\n{doc.page_content.strip()}")
    print(f"Metadata: {doc.metadata}")
    print("-" * 40)
