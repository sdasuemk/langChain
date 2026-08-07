"""
Chroma and FAISS MMR Retrieval Demo

This script demonstrates how to set up both Chroma and FAISS vector stores,
wrap them as MMR retrievers, and query them with identical data.

We use a custom deterministic mock embedding class to showcase the math 
and mechanism behind MMR without needing external APIs or model downloads.

Requirements:
    pip install langchain-chroma langchain-community faiss-cpu
"""

from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS


# =========================================================================
# 1. Custom Embedding Simulation
# =========================================================================
class FoodEmbeddings(Embeddings):
    """
    Simulates a 2D embedding space:
      - Coordinate [0]: Relevance to 'Pizza' search query
      - Coordinate [1]: Relevance to 'Sushi' style query
    """

    def __init__(self):
        # Deterministic vector mapping based on content keywords
        self.vectors = {
            "Joe's Pizza: Located on Bleecker St, serving legendary NY-style slices since 1975.": [0.99, 0.01],
            "John's of Bleecker Street: Famous coal-fired brick oven pizza, very close to Joe's.": [0.98, 0.02],
            "Bleecker Street Pizza: Known for their award-winning Nonna's slice, right on the corner.": [0.97, 0.03],
            "Masa: A highly premium Japanese restaurant serving world-class sushi on Columbus Circle.": [0.15, 0.98],
            "Shake Shack: A popular fast-casual spot famous for smash burgers, crinkle-cut fries, and milkshakes.": [0.60, 0.40]
        }
        self.default_vector = [0.0, 0.0]

    def embed_documents(self, texts):
        return [self.vectors.get(text, self.default_vector) for text in texts]

    def embed_query(self, text):
        # We query for pizza, meaning we want high values on Coordinate [0]
        return [1.0, 0.0]


def main():
    print("=== Chroma & FAISS MMR Retriever Demo ===")

    # Initialize our mock embedding model
    embeddings = FoodEmbeddings()

    # 2. Prepare sample documents
    documents = [
        Document(
            page_content="Joe's Pizza: Located on Bleecker St, serving legendary NY-style slices since 1975.",
            metadata={"cuisine": "italian"}
        ),
        Document(
            page_content="John's of Bleecker Street: Famous coal-fired brick oven pizza, very close to Joe's.",
            metadata={"cuisine": "italian"}
        ),
        Document(
            page_content="Bleecker Street Pizza: Known for their award-winning Nonna's slice, right on the corner.",
            metadata={"cuisine": "italian"}
        ),
        Document(
            page_content="Masa: A highly premium Japanese restaurant serving world-class sushi on Columbus Circle.",
            metadata={"cuisine": "japanese"}
        ),
        Document(
            page_content="Shake Shack: A popular fast-casual spot famous for smash burgers, crinkle-cut fries, and milkshakes.",
            metadata={"cuisine": "american"}
        ),
    ]

    query = "Find me a great pizza recommendation near Bleecker Street"

    # =========================================================================
    # PART A: Chroma MMR Retrieval
    # =========================================================================
    print("\n--- PART A: ChromaDB ---")
    print("Adding documents to Chroma in-memory store...")
    
    chroma_db = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name="ny_food_collection"
    )

    # Instantiate Chroma MMR Retriever
    chroma_mmr_retriever = chroma_db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 3,
            "fetch_k": 5,
            "lambda_mult": 0.2  # Emphasize diversity
        }
    )

    print(f"Query: '{query}'")
    chroma_results = chroma_mmr_retriever.invoke(query)
    print("\nChroma MMR Results:")
    for idx, doc in enumerate(chroma_results, 1):
        print(f"  {idx}. [{doc.metadata.get('cuisine').upper()}] {doc.page_content}")

    # =========================================================================
    # PART B: FAISS MMR Retrieval
    # =========================================================================
    print("\n--- PART B: FAISS (Facebook AI Similarity Search) ---")
    print("Adding documents to FAISS in-memory index...")
    
    faiss_db = FAISS.from_documents(
        documents=documents,
        embedding=embeddings
    )

    # Instantiate FAISS MMR Retriever
    faiss_mmr_retriever = faiss_db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 3,
            "fetch_k": 5,
            "lambda_mult": 0.2  # Emphasize diversity
        }
    )

    print(f"Query: '{query}'")
    faiss_results = faiss_mmr_retriever.invoke(query)
    print("\nFAISS MMR Results:")
    for idx, doc in enumerate(faiss_results, 1):
        print(f"  {idx}. [{doc.metadata.get('cuisine').upper()}] {doc.page_content}")


if __name__ == "__main__":
    main()
