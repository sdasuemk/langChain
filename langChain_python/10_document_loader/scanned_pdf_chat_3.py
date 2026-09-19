"""
LangChain Scanned PDF Loading & Interactive Chat Tutorial
=========================================================
This file demonstrates how to load a scanned (image-only) PDF document 
using Optical Character Recognition (OCR), index it into a vector store, 
and establish an interactive chat session using LangChain Expression Language (LCEL).

Key Concepts:
- Scanned PDF Parsing: Configures PyPDFLoader with RapidOCRBlobParser 
  to extract textual content embedded inside scanned document images.
- UnstructuredPDFLoader vs PyPDFLoader: Explains pros and cons of using Unstructured loaders.
- InMemoryVectorStore: A lightweight vector database built into langchain_core.
- LCEL RAG Chain: Connects retriever, context formatter, prompt, and model using 
  the pipe (|) operator, bypassing legacy chain wrappers.
"""

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Check if OCR parser dependencies are available
try:
    from langchain_community.document_loaders.parsers.images import RapidOCRBlobParser
    RAPID_OCR_AVAILABLE = True
except ImportError:
    RAPID_OCR_AVAILABLE = False

# ---------------------------------------------------------
# STEP 1: Environment Setup
# ---------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------
# STEP 2: Load the PDF (Comparing PyPDFLoader & UnstructuredPDFLoader)
# ---------------------------------------------------------
#
# FAQ: Is UnstructuredPDFLoader a good option?
#
# PROS of UnstructuredPDFLoader:
# 1. Powerful Layout Recognition: Extracts headers, footers, narrative text, 
#    and tables with high precision.
# 2. Native Multi-Format Support: Can handle scanned PDFs, images, HTML, Word files, etc.
# 3. Built-in OCR: Integrates with Tesseract (using "hi_res" strategy) to extract scanned text automatically.
#
# CONS of UnstructuredPDFLoader (Why PyPDFLoader is used here instead):
# 1. Heavy Local Dependencies: Requires system-level installations of Tesseract OCR, 
#    Poppler (for PDF rendering), libmagic, and other OS packages. Setting these up on 
#    Windows can be very challenging.
# 2. Heavy Package Size: The unstructured python package has a huge dependency tree.
# 3. Alternative (Unstructured API): Can be run via hosted API, but requires an API key 
#    and sends document data to a third-party server.
#
# PyPDFLoader (with optional RapidOCR) is a lightweight alternative that runs locally 
# without heavy system dependencies.

pdf_path = "asset/llm-book.pdf" # Replace with your scanned PDF file path

if RAPID_OCR_AVAILABLE:
    print(f"RapidOCR detected. Loading '{pdf_path}' with OCR image extraction...")
    ocr_parser = RapidOCRBlobParser()
    loader = PyPDFLoader(
        file_path=pdf_path,
        extract_images=True,
        images_parser=ocr_parser
    )
else:
    print(f"RapidOCR parser not found. Loading '{pdf_path}' with standard text extraction...")
    print("--> Tip: To enable OCR extraction for scanned PDFs, install:")
    print("    pip install rapidocr-onnxruntime pillow\n")
    loader = PyPDFLoader(file_path=pdf_path)

docs = loader.load()

# ---------------------------------------------------------
# STEP 3: Split the Document into Chunks
# ---------------------------------------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(docs)
print(f"Split document into {len(chunks)} chunks.")

# ---------------------------------------------------------
# STEP 4: Setup API-Based Embeddings & Vector Store
# ---------------------------------------------------------
embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    task="feature-extraction"
)
vector_store = InMemoryVectorStore.from_documents(chunks, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# ---------------------------------------------------------
# STEP 5: Initialize the Chat LLM
# ---------------------------------------------------------
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.5
)
chat_model = ChatHuggingFace(llm=llm)

# ---------------------------------------------------------
# STEP 6: Construct RAG Chain Using LCEL Pipe (|) Operators
# ---------------------------------------------------------
# 1. Define prompt template mapping context and question
system_prompt = (
    "You are a helpful assistant. Use the following context retrieved from the "
    "uploaded PDF to answer the question. If you don't know the answer, state "
    "clearly that you don't know based on the document.\n\n"
    "Context:\n{context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{question}"),
])

# 2. Define a helper function to concatenate document contents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 3. Construct RAG pipeline
# Flow:
# - Input (string question) flows into "question" via RunnablePassthrough.
# - Input also flows into retriever, gets documents, formats them via format_docs, 
#   and binds to "context".
# - Dictionary {"context": ..., "question": ...} passes into prompt.
# - Prompt passes into chat model.
# - Chat model outputs response, parsed into clean string by StrOutputParser.
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | chat_model
    | StrOutputParser()
)

# ---------------------------------------------------------
# STEP 7: Interactive Chat Loop
# ---------------------------------------------------------
print("\n=== PDF Chat Session Started ===")
print("You can now ask questions about the PDF document. Type 'exit' to quit.\n")

while True:
    try:
        user_input = input("You: ")
        if user_input.strip().lower() == "exit":
            print("Ending chat session. Goodbye!")
            break
        if not user_input.strip():
            continue
        
        print("Thinking...")
        # Invoke our LCEL chain with the user's question string
        response = rag_chain.invoke(user_input)
        
        print(f"\nAI: {response}\n")
        print("-" * 50)
    except KeyboardInterrupt:
        print("\nEnding chat session. Goodbye!")
        break
    except Exception as e:
        print(f"Error during query: {e}\n")
