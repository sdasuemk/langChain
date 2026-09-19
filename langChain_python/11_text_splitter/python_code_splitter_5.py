"""
LangChain Python Code Text Splitter Tutorial
===========================================
This file demonstrates how to use the RecursiveCharacterTextSplitter 
to split Python source code based on Python-specific syntax separators.

Key Concepts:
- Language.PYTHON: Configures the recursive character text splitter to use 
  separators tailored for Python structure (e.g., class, def, indentation blocks).
- Syntax-Aware Splitting: Keeps Python classes, functions, and logical code blocks 
  together as much as possible, preventing splits in the middle of statements.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

# ---------------------------------------------------------
# STEP 1: Define the Python Source Code
# ---------------------------------------------------------
# We define a sample Python file string containing a class, functions, and comments.
python_code = """
class Calculator:
    def __init__(self, owner: str):
        self.owner = owner

    def add(self, a: int, b: int) -> int:
        \"\"\"Add two numbers.\"\"\"
        return a + b

    def subtract(self, a: int, b: int) -> int:
        \"\"\"Subtract two numbers.\"\"\"
        return a - b

def main():
    calc = Calculator("Alice")
    result = calc.add(5, 7)
    print(f"Result: {result}")

if __name__ == "__main__":
    main()
"""

# ---------------------------------------------------------
# STEP 2: Initialize the Splitter for Python Code
# ---------------------------------------------------------
# We create a splitter using `.from_language()`, passing `Language.PYTHON`.
# This automatically configures the separators for Python:
# ['\\ndef ', '\\nclass ', '\\n\\tdef ', '\\n\\tclass ', '\\n', ' ', '']
python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=150,            # Small chunk size to demonstrate how it splits functions/classes
    chunk_overlap=0,           # No overlap for clear boundary demonstration
)

# ---------------------------------------------------------
# STEP 3: Split the Code into Chunks
# ---------------------------------------------------------
# Split the Python code string into text chunks using `.split_text()`.
chunks = python_splitter.split_text(python_code)

# ---------------------------------------------------------
# STEP 4: Inspect the Separators Used
# ---------------------------------------------------------
print("--- Python Splitter Separators ---")
print(RecursiveCharacterTextSplitter.get_separators_for_language(Language.PYTHON))
print()

# ---------------------------------------------------------
# STEP 5: Inspect the Generated Chunks
# ---------------------------------------------------------
# Print each chunk content to see where splits were made.
for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i+1} ---")
    print(chunk.strip())
    print("-" * 40)
