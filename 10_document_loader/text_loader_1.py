from langchain_community.document_loaders import TextLoader

# Initialize the loader with a text file path
loader = TextLoader("asset/langchain_poem.txt", encoding="utf-8") # e.g., {'source': 'asset/readme.txt'}

# Load documents
docs = loader.load()

# Inspect results
print(docs[0].page_content)
print(docs[0].metadata)  