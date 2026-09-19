"""
LangChain Custom Document Loader Tutorial
========================================
This file demonstrates how to create a custom Document Loader by subclassing 
LangChain's `BaseLoader` and implementing `lazy_load()` for memory-efficient 
streaming of custom data.

Key Concepts:
- BaseLoader: The foundation class for all loaders in LangChain.
- lazy_load(): A generator method that yields `Document` objects one by one.
- Custom Parsing: Demonstrates parsing a custom text log file and injecting 
  structured fields (like log level, timestamp) into document metadata.
"""

from typing import Iterator
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

# Let's define a mock logs dataset (simulating a log file content)
MOCK_LOGS = """
[2026-07-30T10:00:00] [INFO] System initialized successfully.
[2026-07-30T10:05:22] [WARNING] Disk space usage exceeds 85%.
[2026-07-30T10:12:01] [ERROR] Database connection failed. Refusing connection on port 5432.
[2026-07-30T10:15:30] [INFO] User login successful for admin@domain.com.
"""

class CustomLogLoader(BaseLoader):
    """A custom loader that parses system logs.
    
    Extracts timestamps and log levels into the Document metadata dictionary 
    for detailed filtering during RAG retrieval.
    """

    def __init__(self, log_content: str):
        self.log_content = log_content

    def lazy_load(self) -> Iterator[Document]:
        """A lazy generator that parses logs line by line and yields Document objects."""
        # Split logs by newline
        lines = self.log_content.strip().split("\n")
        
        for line in lines:
            if not line.strip():
                continue
            
            # Simple log parser: [timestamp] [level] message
            try:
                # Find brackets
                parts = line.split("] ")
                timestamp = parts[0].replace("[", "")
                level_part = parts[1].split("] ")
                level = level_part[0].replace("[", "")
                message = level_part[1]
                
                # Yield a Document object
                yield Document(
                    page_content=message,
                    metadata={
                        "timestamp": timestamp,
                        "log_level": level,
                        "source": "system_logs"
                    }
                )
            except Exception:
                # If a line doesn't match standard log formatting, load it as raw text
                yield Document(
                    page_content=line,
                    metadata={"source": "system_logs_raw"}
                )

# ---------------------------------------------------------
# STEP 1: Initialize the Custom Loader
# ---------------------------------------------------------
print("Initializing CustomLogLoader with mock log content...")
loader = CustomLogLoader(MOCK_LOGS)

# ---------------------------------------------------------
# STEP 2: Load Documents using lazy_load() (Memory Efficient)
# ---------------------------------------------------------
# `lazy_load()` yields documents one-by-one. This prevents loading large files
# entirely into memory.
print("\n--- Lazy Loading Logs (Iterating Generator) ---")
for doc in loader.lazy_load():
    print(f"[{doc.metadata['log_level']}] Content: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")
    print("-" * 50)

# ---------------------------------------------------------
# STEP 3: Load Documents using standard load()
# ---------------------------------------------------------
# BaseLoader automatically provides `.load()` by consuming the generator
# and returning a complete list.
print("\n--- Standard Loading (load() list) ---")
all_docs = loader.load()
print(f"Total documents loaded: {len(all_docs)}")
