"""
LangChain RAG Pipeline Practical Example
=============================================================
This script demonstrates a real-world, practical RAG (Retrieval-Augmented Generation) 
pipeline that constructs a troubleshooting chatbot for software developers.

Key Components Demonstrated:
1. Document Loader: Load a troubleshooting guide using TextLoader.
2. Text Splitter: Chunk the document using RecursiveCharacterTextSplitter.
3. Vector Store: Index and store the text embeddings using ChromaDB (langchain_chroma).
4. LCEL Chain: Connect the retriever, prompt template, chat model, and parser 
   using the pipe (|) operator.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Document Loader
from langchain_community.document_loaders import TextLoader
# Text Splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Vector Store & Embeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEndpointEmbeddings
# LLM, Prompts & Chain Utilities
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ---------------------------------------------------------
# STEP 1: Environment Setup
# ---------------------------------------------------------
# Load environment variables from .env
load_dotenv()

if not os.environ.get("HUGGINGFACEHUB_API_TOKEN"):
    print("WARNING: HUGGINGFACEHUB_API_TOKEN not found in environment variables.")
    print("Please set it in your .env file to run Hugging Face models.")

# ---------------------------------------------------------
# STEP 2: Define Document Path
# ---------------------------------------------------------
guide_path = Path("asset/developer_troubleshooting_guide.txt")

if not guide_path.exists():
    raise FileNotFoundError(
        f"Required troubleshooting guide not found at {guide_path}. "
        "Please make sure it exists."
    )

try:
    # ---------------------------------------------------------
    # STEP 3: Load the Document (Doc Loader)
    # ---------------------------------------------------------
    print("\nLoading document...")
    loader = TextLoader(str(guide_path), encoding="utf-8")
    docs = loader.load()

    # ---------------------------------------------------------
    # STEP 4: Split the Document (Text Splitter)
    # ---------------------------------------------------------
    print("Splitting document into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(docs)
    print(f"Split document into {len(chunks)} chunks.")

    # ---------------------------------------------------------
    # STEP 5: Initialize Embeddings & Vector Store (ChromaDB)
    # ---------------------------------------------------------
    print("\nInitializing Hugging Face Endpoint Embeddings...")
    embeddings = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        task="feature-extraction"
    )

    # Initialize Chroma database saved locally to './chroma_db'
    print("Storing embeddings in ChromaDB...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    # Expose ChromaDB as a Retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 1})

    # ---------------------------------------------------------
    # STEP 6: Configure the LLM & Chat Model
    # ---------------------------------------------------------
    print("Connecting to Hugging Face Model Endpoint...")
    llm = HuggingFaceEndpoint(
        repo_id="deepseek-ai/DeepSeek-V4-Pro",
        task="text-generation",
        max_new_tokens=256,
        temperature=0.2
    )
    chat_model = ChatHuggingFace(llm=llm)

    # ---------------------------------------------------------
    # STEP 7: Define RAG Prompt & Helper Functions
    # ---------------------------------------------------------
    system_prompt = (
        "You are an IT helpdesk bot. Use the retrieved context below to answer "
        "the developer's question. Be concise and provide shell commands if possible. "
        "If the answer cannot be found in the context, say 'I cannot find the solution in the guide'.\n\n"
        "Retrieved Context:\n{context}"
    )
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}"),
    ])

    # Helper function to format list of documents into a single text block
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # ---------------------------------------------------------
    # STEP 8: Construct the LCEL RAG Chain (|)
    # ---------------------------------------------------------
    # The chain automatically handles:
    # 1. Retrieving relevant docs based on user query and formatting them.
    # 2. Feeding formatted context and user query into the prompt.
    # 3. Passing prompt payload to ChatHuggingFace.
    # 4. Converting model responses into strings using StrOutputParser.
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt_template
        | chat_model
        | StrOutputParser()
    )

    # ---------------------------------------------------------
    # STEP 9: Execute the Query
    # ---------------------------------------------------------
    query = "What should I do if poetry install throws SolverProblemError?"
    print(f"\n[User Query]: {query}")
    print("Invoking LCEL chain...")
    
    response = rag_chain.invoke(query)
    print("\n--- IT Helpdesk Bot Response ---")
    print(response)
    print("--------------------------------")

except Exception as e:
    print(f"\nAn error occurred during execution: {e}")
    print("Ensure you have a valid HUGGINGFACEHUB_API_TOKEN in your .env file.")

finally:
    # Cleanup database or connections if necessary
    pass
