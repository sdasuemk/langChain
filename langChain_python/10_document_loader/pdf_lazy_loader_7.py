"""
LangChain PDF Lazy Loading Tutorial
===================================
This file demonstrates how to use the lazy loading capability of `PyPDFLoader` 
to stream pages of a PDF one by one instead of loading the entire document 
into memory at once.

Key Concepts:
- lazy_load(): A generator method available in LangChain loaders that yields 
  Document objects (pages) sequentially.
- Memory Efficiency: Crucial for processing large documents (e.g., hundreds of 
  pages) or during concurrent operations, preventing memory spikes.
"""

from langchain_community.document_loaders import PyPDFLoader

# ---------------------------------------------------------
# STEP 1: Initialize PyPDFLoader
# ---------------------------------------------------------
# We point the loader to our local PDF. It will load pages lazily.
pdf_path = "asset/llm-book.pdf"
print(f"Initializing PyPDFLoader for '{pdf_path}'...")
loader = PyPDFLoader(file_path=pdf_path)

# ---------------------------------------------------------
# STEP 2: Iterate Pages Lazily
# ---------------------------------------------------------
# Instead of calling loader.load() which returns a full Python list of all pages,
# we use loader.lazy_load() to get an iterator.
# This yields pages one by one as we progress in the loop.
print("\nStreaming PDF pages lazily...")
page_count = 0

for page in loader.lazy_load():
    page_count += 1
    # Print progress information for each page
    # Note that metadata tracks the index of the current page (starting at 0)
    print(f"----> Processed Page {page_count} (Metadata: {page.metadata})")
    
    # We can inspect or process page content on the fly without keeping all page texts in RAM
    text_snippet = page.page_content[:100].replace("\n", " ").strip()
    print(f"     Content snippet: \"{text_snippet}...\"")
    
    # Stop early for demonstration purposes (e.g. first 3 pages)
    if page_count >= 3:
        print("\nStopping early. (Only loaded first 3 pages into memory)")
        break

print(f"\nCompleted lazy loading demonstration. Processed {page_count} page(s).")
