# length based text splitting

from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("asset/llm-book.pdf")

docs = loader.load()

# print("Documents", docs)
print("Document -1", docs[0]) # page_content, metadata

# Initialize the splitter
text_splitter = CharacterTextSplitter(
    separator="", 
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)

# Split a documents object into text chunks
chunks = text_splitter.split_documents(docs)


for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i+1} ---")
    print(chunk)

print("Chunk length", len(chunks))