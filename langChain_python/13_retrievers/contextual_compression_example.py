"""
Contextual Compression Retriever Example in LangChain

This script demonstrates how to set up and run a Contextual Compression Retriever.
To ensure this runs immediately without requiring an OpenAI or Anthropic API key,
we define a custom BaseDocumentCompressor that extracts relevant sentences.

Requirements:
    pip install langchain-core langchain-community
"""

from typing import List, Sequence
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_classic.retrievers import ContextualCompressionRetriever


# =========================================================================
# 1. Custom Embedding Simulation
# =========================================================================
class SimpleEmbeddings(Embeddings):
    """A basic keyword-matching embedding mapping for demonstration."""
    def __init__(self):
        self.vectors = {
            "FastAPI is a modern web framework. It was created by Sebastián Ramírez. To deploy FastAPI on AWS, use App Runner which handles auto-scaling.": [1.0, 0.0],
            "Python is a versatile programming language. It supports multiple paradigms. To read CSV files, you can use the built-in csv module.": [0.0, 1.0],
        }
        self.default_vector = [0.5, 0.5]

    def embed_documents(self, texts):
        return [self.vectors.get(text, self.default_vector) for text in texts]

    def embed_query(self, text):
        if "fastapi" in text.lower() or "deploy" in text.lower():
            return [1.0, 0.0]
        if "csv" in text.lower() or "python" in text.lower():
            return [0.0, 1.0]
        return [0.5, 0.5]


# =========================================================================
# 2. Custom Document Compressor
# =========================================================================
class CustomSentenceExtractor(BaseDocumentCompressor):
    """
    A simple rule-based compressor that splits documents into sentences
    and retains only those containing keywords from the query.
    """

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks = None
    ) -> Sequence[Document]:
        compressed_docs = []
        query_words = [w.lower() for w in query.split() if len(w) > 2]  # ignore tiny words

        for doc in documents:
            # Split the document into sentences
            sentences = doc.page_content.split(". ")
            relevant_sentences = []

            for sentence in sentences:
                # If any query keyword matches a word in the sentence, keep it
                if any(word in sentence.lower() for word in query_words):
                    relevant_sentences.append(sentence.strip())

            if relevant_sentences:
                # Reconstruct document with only relevant sentences
                compressed_content = ". ".join(relevant_sentences)
                # Ensure the text ends with a period
                if not compressed_content.endswith("."):
                    compressed_content += "."
                
                compressed_docs.append(
                    Document(page_content=compressed_content, metadata=doc.metadata)
                )

        return compressed_docs


# =========================================================================
# 3. Main Execution
# =========================================================================
def main():
    print("=== Contextual Compression Retriever Demo ===")

    # 1. Initialize vector store
    embeddings = SimpleEmbeddings()
    vectorstore = InMemoryVectorStore(embeddings)

    # 2. Add sample documents (consisting of multiple sentences, some irrelevant)
    documents = [
        Document(
            page_content=(
                "FastAPI is a modern web framework. "
                "It was created by Sebastián Ramírez. "
                "To deploy FastAPI on AWS, use App Runner which handles auto-scaling."
            ),
            metadata={"source": "fastapi-docs"}
        ),
        Document(
            page_content=(
                "Python is a versatile programming language. "
                "It supports multiple paradigms. "
                "To read CSV files, you can use the built-in csv module."
            ),
            metadata={"source": "python-docs"}
        )
    ]
    vectorstore.add_documents(documents)
    print("Vector Store initialized with documents containing filler text.\n")

    query = "How to deploy FastAPI on AWS"
    print(f"Query: '{query}'")

    # -----------------------------------------------------------------
    # Approach A: Standard Retriever (Returns the entire document)
    # -----------------------------------------------------------------
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
    
    print("\n--- [A] Standard Retrieval (Entire Document) ---")
    standard_docs = base_retriever.invoke(query)
    for idx, doc in enumerate(standard_docs, 1):
        print(f"Document {idx} (Length: {len(doc.page_content)} characters):")
        print(f"  {doc.page_content}\n")

    # -----------------------------------------------------------------
    # Approach B: Contextual Compression Retriever (Compresses documents)
    # -----------------------------------------------------------------
    compressor = CustomSentenceExtractor()
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )

    print("--- [B] Contextual Compression Retrieval (Compressed Document) ---")
    compressed_docs = compression_retriever.invoke(query)
    for idx, doc in enumerate(compressed_docs, 1):
        print(f"Document {idx} (Length: {len(doc.page_content)} characters):")
        print(f"  {doc.page_content}\n")

    print("===============================================================")
    print("Why did this happen?")
    print("1. Standard retrieval returned the entire document, including ")
    print("   irrelevant details (who created FastAPI, and framework descriptions).")
    print("2. Contextual Compression parsed the document and returned only the")
    print("   sentence containing deployment instructions matching the query.")
    print("===============================================================")


if __name__ == "__main__":
    main()
