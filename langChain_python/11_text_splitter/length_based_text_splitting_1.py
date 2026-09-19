"""
LangChain CharacterTextSplitter Tutorial
=========================================
This file demonstrates how to use the CharacterTextSplitter to split 
a raw string into smaller chunks based on a specific separator, 
chunk size, and chunk overlap.

Key Concepts:
- CharacterTextSplitter: Splits text by searching for a specific separator.
- Chunk Size: The maximum target size of each split chunk.
- Chunk Overlap: The number of characters that can overlap between consecutive chunks to preserve context.
"""

from langchain_text_splitters import CharacterTextSplitter

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
# STEP 2: Initialize the CharacterTextSplitter
# ---------------------------------------------------------
# CharacterTextSplitter splits documents based on a separator (default is "\n\n").
#
# Rules of CharacterTextSplitter:
# 1. It will not break apart a piece of text if it doesn't find your designated separator 
#    within the target chunk_size.
# 2. Since Paragraph 1 has no "\n\n" inside it, the splitter cannot split it any further 
#    without breaking its rules. Thus, it keeps the entire 114-character paragraph intact 
#    rather than cutting off in the middle of words.
# 3. If you set separator="", it will split after the chunk_size threshold (e.g., 100 characters) 
#    is reached, potentially breaking in the middle of words or sentences.
text_splitter = CharacterTextSplitter(
    separator="\n\n",          # Splits paragraphs using the double newline character
    chunk_size=100,            # Target character length for each text chunk
    chunk_overlap=20,          # Overlapping characters between consecutive chunks to retain context
    length_function=len,       # Function used to calculate chunk size (in characters)
    is_separator_regex=False,  # Treat the separator as a literal string, not a regular expression
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