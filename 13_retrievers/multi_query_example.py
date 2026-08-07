"""
Multi-Query Retriever Example in LangChain

This script demonstrates how to set up and use a Multi-Query Retriever.
To ensure this runs immediately without requiring an OpenAI or Anthropic API key,
we define a simple Mock LLM that simulates query expansion.

Requirements:
    pip install langchain-core langchain-community
"""

import os
from typing import List, Optional
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.embeddings import Embeddings
from langchain_classic.retrievers.multi_query import MultiQueryRetriever


# =========================================================================
# 1. Custom Embedding Simulation
# =========================================================================
class SimpleEmbeddings(Embeddings):
    """A basic keyword-matching embedding mapping for demonstration."""
    def __init__(self):
        self.vectors = {
            "Open file in Python: Use the built-in open() function with context managers.": [1.0, 0.0],
            "Python CSV reader: The 'csv' module helps read and write spreadsheet files.": [0.8, 0.1],
            "FastAPI deploy: Setup and serve APIs on AWS using docker and App Runner.": [0.1, 0.9],
        }
        self.default_vector = [0.0, 0.0]

    def embed_documents(self, texts):
        return [self.vectors.get(text, self.default_vector) for text in texts]

    def embed_query(self, text):
        # Queries matching "file" or "open" align with the first vector
        if "file" in text.lower() or "open" in text.lower() or "read" in text.lower():
            return [1.0, 0.0]
        # Queries matching "api" or "deploy" align with the second
        if "deploy" in text.lower() or "api" in text.lower():
            return [0.1, 0.9]
        return [0.5, 0.5]


# =========================================================================
# 2. Mock Language Model for Query Expansion
# =========================================================================
class MockChatModel(BaseChatModel):
    """
    Simulates a Chat Model generating query variations for MultiQueryRetriever.
    MultiQueryRetriever expects variations separated by newlines.
    """

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager = None,
        **kwargs
    ) -> ChatResult:
        # Simulated response: Generates 3 query variations on newlines
        simulated_queries = (
            "How to open a file in Python\n"
            "Python read file utilities\n"
            "Load CSV data in Python script"
        )
        message = AIMessage(content=simulated_queries)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    @property
    def _llm_type(self) -> str:
        return "mock-chat-model"


# =========================================================================
# 3. Main Execution
# =========================================================================
def main():
    print("=== Multi-Query Retriever Demo ===")

    # 1. Initialize custom vector store
    embeddings = SimpleEmbeddings()
    vectorstore = InMemoryVectorStore(embeddings)

    # 2. Add documents to the index
    documents = [
        Document(page_content="Open file in Python: Use the built-in open() function with context managers."),
        Document(page_content="Python CSV reader: The 'csv' module helps read and write spreadsheet files."),
        Document(page_content="FastAPI deploy: Setup and serve APIs on AWS using docker and App Runner."),
    ]
    vectorstore.add_documents(documents)
    print("Vector Store indexed with 3 documents.")

    # 3. Initialize the Mock LLM
    mock_llm = MockChatModel()

    # 4. Create the Multi-Query Retriever
    # Using `from_llm` wraps our base vector store retriever in an LLM query expansion layer.
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 1}) # we can use MMR as well
    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=mock_llm
    )

    # 5. Execute retrieval
    user_query = "Python file tool"
    print(f"\nUser Query: '{user_query}'")
    print("Running Multi-Query Retrieval...")
    
    docs = multi_query_retriever.invoke(user_query)

    print("\n--- Final Unified Retrieved Documents (Deduplicated) ---")
    for idx, doc in enumerate(docs, 1):
        print(f"  {idx}. {doc.page_content}")

    print("\n=======================================================")
    print("How to transition to a production LLM:")
    print("Simply replace 'mock_llm' with a real chat model, like:")
    print("  from langchain_openai import ChatOpenAI")
    print("  llm = ChatOpenAI(temperature=0, model='gpt-4o')")
    print("  retriever = MultiQueryRetriever.from_llm(retriever, llm)")
    print("=======================================================")


if __name__ == "__main__":
    main()
