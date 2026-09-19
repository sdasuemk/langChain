import os
import sys
import io
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Import our transcript fetching helper
from youtube_transcript import fetch_youtube_transcript

# Configure standard output to use UTF-8 encoding (prevents Windows terminal encoding errors)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Load environment variables
load_dotenv()

def run_rag_step(query: str, retriever, prompt_template, chat_model):
    """
    Executes the RAG pipeline explicitly, step-by-step:
    1. Retrieval: retrieves matching chunks from the FAISS database.
    2. Augmentation: constructs context_text and formats the prompt template.
    3. Generation: calls the LLM with the formatted prompt and prints the answer.
    """
    print("\n--- 1. RETRIEVAL STEP ---")
    print(f"Retrieving matching document chunks for: '{query}'...")
    results = retriever.invoke(query)
    print(f"Retrieved {len(results)} relevant chunks.")
    
    print("\n--- 2. AUGMENTATION STEP ---")
    # Generate context_text by concatenating all results
    context_text = "\n\n".join(doc.page_content for doc in results)
    print(f"Context length: {len(context_text)} characters.")
    
    # Format/Augment prompt template
    augmented_prompt = prompt_template.invoke({
        "context": context_text,
        "question": query
    })
    print("Prompt successfully augmented with retrieved context.")
    
    print("\n--- 3. GENERATION STEP ---")
    print("Invoking Chat LLM with augmented prompt...")
    try:
        # Execute Generation explicitly
        response_message = chat_model.invoke(augmented_prompt)
        print("\n=== AI Response ===")
        print(response_message.content)
    except Exception as e:
        print(f"Error during LLM generation: {e}")

def main():
    # 1. Fetch transcript
    default_url = "https://www.youtube.com/watch?v=zjkBMFhNj_g"
    video_url = default_url
    query_argument = None

    if len(sys.argv) > 1:
        video_url = sys.argv[1]
        # Check if more arguments were passed (which will form our search query)
        if len(sys.argv) > 2:
            query_argument = " ".join(sys.argv[2:])
    else:
        video_url = input(f"Enter YouTube URL (press Enter for default '{default_url}'): ").strip()
        if not video_url:
            video_url = default_url

    doc = fetch_youtube_transcript(video_url)
    if not doc:
        print("Failed to load transcript. Exiting.")
        return

    # 2. Split transcript into chunks
    print("\nSplitting transcript into text chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents([doc])
    print(f"Split transcript into {len(chunks)} document chunks.")

    # Show a sample chunk
    if chunks:
        print("\n=== Sample Chunk #1 ===")
        print(chunks[0].page_content)
        print(f"Metadata: {chunks[0].metadata}")

    # 3. Initialize Embedding Model
    print("\nInitializing API-based Hugging Face Embeddings...")
    huggingface_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    if not huggingface_token:
        print("WARNING: HUGGINGFACEHUB_API_TOKEN not found in environment variables. Please check your .env file.")
        
    embeddings = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=huggingface_token
    )

    # 4. Generate Embeddings and create FAISS index in-memory
    print("Embedding chunks and building in-memory FAISS database...")
    try:
        vector_store = FAISS.from_documents(chunks, embeddings)
        print("FAISS vector database successfully created in-memory!")
    except Exception as e:
        print(f"Error during embedding or FAISS indexing: {e}")
        return

    # 5. Create a Retriever interface
    print("Instantiating LangChain Retriever...")
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    # 6. Initialize the Language Model (Generation Stage)
    print("\nInitializing API-based Hugging Face Chat LLM...")
    llm = HuggingFaceEndpoint(
        repo_id="deepseek-ai/DeepSeek-V4-Pro",
        task="text-generation",
        max_new_tokens=512,
        temperature=0.5,
        huggingfacehub_api_token=huggingface_token
    )
    chat_model = ChatHuggingFace(llm=llm)

    # 7. Define the Prompt Template
    system_prompt = (
        "You are a helpful assistant. Use the following context retrieved from the YouTube video transcript "
        "to answer the question. If you don't know the answer, say that you don't know.\n\n"
        "Context:\n{context}"
    )
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}"),
    ])

    # 8. Let the user perform QA queries using RAG step-by-step
    if query_argument:
        print(f"\nRunning test query from command-line: '{query_argument}'")
        run_rag_step(query_argument, retriever, prompt_template, chat_model)
    else:
        print("\n=== YouTube Video QA RAG Chat (Step-by-Step) ===")
        while True:
            query = input("\nAsk a question about the video (or 'q' to quit): ").strip()
            if not query or query.lower() == 'q':
                break
            run_rag_step(query, retriever, prompt_template, chat_model)

if __name__ == "__main__":
    main()
