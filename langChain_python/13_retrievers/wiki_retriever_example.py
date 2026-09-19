"""
Example of a Simple Wikipedia Retriever in LangChain.

This script demonstrates how to use `WikipediaRetriever` to retrieve relevant documents 
directly from Wikipedia for a query. This is a lightweight retrieval approach that 
doesn't require setting up a local vector database.

Requirements:
    pip install langchain-community wikipedia
"""

import os
from dotenv import load_dotenv
import wikipedia
from langchain_community.retrievers import WikipediaRetriever

# Load environment variables from .env file (if present)
load_dotenv()

# Set a custom user agent to avoid Wikimedia rate-limiting / block
wikipedia.set_user_agent("WikiRetrieverDemo/1.0 (study_notes@example.com)")


def main():
    # 1. Initialize the Wikipedia Retriever
    # - `top_k_results=3` specifies that we want to fetch the top 3 matching articles.
    # - `lang="en"` retrieves articles from the English version of Wikipedia.
    print("Initializing Wikipedia Retriever...")
    retriever = WikipediaRetriever(top_k_results=3, lang="en")

    # 2. Define the search query
    query = "Generative Artificial Intelligence"
    print(f"\nSearching Wikipedia for: '{query}'...")

    try:
        # 3. Retrieve relevant documents
        # The retriever returns a list of Document objects
        docs = retriever.invoke(query)

        # 4. Display the retrieved documents and their metadata
        print(f"\nSuccessfully retrieved {len(docs)} document(s):\n")
        
        for idx, doc in enumerate(docs, start=1):
            title = doc.metadata.get("title", "Unknown Title")
            source = doc.metadata.get("source", "Unknown Source")
            summary = doc.page_content[:300].replace("\n", " ") + "..."
            
            print(f"--- Document #{idx} ---")
            print(f"Title : {title}")
            print(f"Source: {source}")
            print(f"Snippet:\n  {summary}\n")

    except Exception as e:
        print(f"\nError occurred during retrieval: {e}")
        print("Please ensure the 'wikipedia' library is installed: pip install wikipedia")


if __name__ == "__main__":
    main()
