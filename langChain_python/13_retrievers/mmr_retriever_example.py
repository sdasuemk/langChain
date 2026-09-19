"""
Maximal Marginal Relevance (MMR) Retriever Example

This script illustrates the concept, "why", "what", and "how" of MMR
using a clean, deterministic, custom-vector embedding simulation.

Requirements:
    pip install langchain-core
"""

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document


# =====================================================================
# 1. Custom Embedding Simulation
# =====================================================================
class FoodRecommenderEmbeddings(Embeddings):
    """
    A custom embedding generator designed to showcase MMR.
    We map foods into a 2D coordinate space:
      - X-axis [0]: Relevance to 'Pizza' search query
      - Y-axis [1]: Cuisine style diversity (0.0=Italian/Pizza, 0.5=Burgers, 1.0=Sushi)
    """

    def __init__(self):
        self.vectors = {
            "Joe's Pizza (Delicious Bleecker St Pizza)": [0.99, 0.01],
            "John's Pizza (Classic NY style Pizza)": [0.98, 0.02],
            "Bleecker Street Pizza (Famous slice shop)": [0.97, 0.03],
            "Masa (High-end Japanese Sushi)": [0.65, 0.95],
            "Shake Shack (Gourmet Burgers & Fries)": [0.75, 0.50]
        }
        self.default_vector = [0.0, 0.0]

    def embed_documents(self, texts):
        return [self.vectors.get(text, self.default_vector) for text in texts]

    def embed_query(self, text):
        # User query: "Recommend some delicious Pizza in Bleecker Street"
        # Maps perfectly to pizza relevance [1.0, 0.0]
        return [1.0, 0.0]


# =====================================================================
# 2. Main Logic Execution
# =====================================================================
def main():
    print("=== Maximal Marginal Relevance (MMR) Example ===")
    
    # 1. Initialize custom embedding model & Vector Store
    embeddings = FoodRecommenderEmbeddings()
    vectorstore = InMemoryVectorStore(embeddings)

    # 2. Add sample documents
    documents = [
        Document(page_content="Joe's Pizza (Delicious Bleecker St Pizza)"),
        Document(page_content="John's Pizza (Classic NY style Pizza)"),
        Document(page_content="Bleecker Street Pizza (Famous slice shop)"),
        Document(page_content="Masa (High-end Japanese Sushi)"),
        Document(page_content="Shake Shack (Gourmet Burgers & Fries)"),
    ]
    vectorstore.add_documents(documents)
    
    query = "Recommend some delicious Pizza in Bleecker Street"
    
    # -----------------------------------------------------------------
    # Approach A: Standard Similarity Search (Pure Relevance)
    # -----------------------------------------------------------------
    # This retrieves the top 3 documents based strictly on closest cosine similarity.
    similarity_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    
    print(f"\n[A] Standard Similarity Search (k=3)")
    print(f"Query: '{query}'")
    similarity_results = similarity_retriever.invoke(query)
    for idx, doc in enumerate(similarity_results, 1):
        print(f"  {idx}. {doc.page_content}")

    # -----------------------------------------------------------------
    # Approach B: MMR Search (Relevance + Diversity)
    # -----------------------------------------------------------------
    # This retrieves a larger pool (fetch_k=5) and selects the best 3 (k=3)
    # balancing query similarity with diversity against already selected items.
    # lambda_mult (0.0 to 1.0) controls the balance:
    #   - 1.0: Pure similarity (equivalent to standard search)
    #   - 0.0: Pure diversity (most different items possible)
    mmr_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 3,
            "fetch_k": 5,
            "lambda_mult": 0.2  # Emphasize diversity!
        }
    )

    print(f"\n[B] MMR Search (k=3, fetch_k=5, lambda_mult=0.2)")
    print(f"Query: '{query}'")
    mmr_results = mmr_retriever.invoke(query)
    for idx, doc in enumerate(mmr_results, 1):
        print(f"  {idx}. {doc.page_content}")

    print("\n================================================")
    print("Why did this happen?")
    print("1. Standard Similarity Search returned 3 pizza shops because they are ")
    print("   the closest vectors to the query. However, they are redundant.")
    print("2. MMR Search returned Joe's Pizza first (most relevant), then skipped")
    print("   the other pizza shops because they were too similar to Joe's Pizza.")
    print("   Instead, it selected Shake Shack (Burger) and Masa (Sushi) to provide")
    print("   a diverse recommendations list while remaining somewhat relevant.")
    print("================================================")


if __name__ == "__main__":
    main()
