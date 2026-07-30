"""
LangChain RecursiveCharacterTextSplitter Tutorial
==================================================
This file demonstrates how to use the RecursiveCharacterTextSplitter to split 
a raw string into smaller chunks recursively using a list of separators.

Key Concepts:
- RecursiveCharacterTextSplitter: Splits text by trying to split on a list of 
  separators in order, aiming to keep paragraphs, sentences, and words together 
  as much as possible.
- Default Separators: ["\n\n", "\n", " ", ""]. It starts by trying to split on 
  double newlines, then single newlines, then spaces, and finally character by character.
- Chunk Size & Overlap: Identical in function to CharacterTextSplitter, but applied 
  flexibly down the separator hierarchy.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------
# STEP 1: Define the Raw Input Text
# ---------------------------------------------------------
# Here we define a multi-paragraph raw string.
# We will split this text into smaller parts.
text = """Paragraph 1:
This is the first paragraph. It contains some detailed information that needs to be split properly.

Paragraph 2:
This is the second paragraph. It also contains text that will be chunked based on size and separator.

Paragraph 3:
This is the third paragraph.
"""

# ---------------------------------------------------------
# STEP 2: Initialize the RecursiveCharacterTextSplitter
# ---------------------------------------------------------
# Unlike CharacterTextSplitter, which only splits on a single separator and will fail
# to split if the text is longer than chunk_size without a separator, 
# RecursiveCharacterTextSplitter uses a default separator list: ["\n\n", "\n", " ", ""].
#
# Rules of RecursiveCharacterTextSplitter:
# 1. First, it tries to split on double newlines ("\n\n") to keep paragraphs together.
# 2. If a paragraph is still larger than the target chunk_size (100 characters), it will 
#    recursively split that chunk using single newlines ("\n") to keep sentences together.
# 3. If it's still too large, it splits using spaces (" ") to keep words together.
# 4. As a last resort, it splits character by character ("").
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,            # Target character length for each text chunk
    chunk_overlap=20,          # Overlapping characters between consecutive chunks to retain context
    length_function=len,       # Function used to calculate chunk size (in characters)
    is_separator_regex=False,  # Treat separators as literal strings, not regular expressions
)

# ---------------------------------------------------------
# STEP 3: Split the Raw Text into Chunks
# ---------------------------------------------------------
# Use `.split_text()` to split the raw python string into a list of strings (chunks).
chunks = text_splitter.split_text(text)

# ---------------------------------------------------------
# STEP 4: Inspect the Generated Chunks
# ---------------------------------------------------------
# Print the chunks with their index to see the split outcomes.
for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i+1} ---")
    print(chunk)

