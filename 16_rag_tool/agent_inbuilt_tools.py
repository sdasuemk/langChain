import os
import sys
import io
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_classic.agents import create_react_agent, AgentExecutor

# Import built-in tools
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

# Configure standard output to use UTF-8 encoding (prevents Windows terminal encoding errors)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Load environment variables
load_dotenv()

# Add 15_langchain_rag folder to Python path for transcript module import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "15_langchain_rag")))
from youtube_transcript import fetch_youtube_transcript

# Initialize FAISS database and Retriever globally for our tool to access
retriever = None

def init_retriever(video_url: str):
    """
    Fetches the transcript, chunks it, embeds it, and returns a retriever interface.
    """
    global retriever
    doc = fetch_youtube_transcript(video_url)
    if not doc:
        print("Failed to load transcript.")
        return False

    print("\nSplitting transcript into text chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents([doc])
    print(f"Split transcript into {len(chunks)} document chunks.")

    print("\nInitializing Hugging Face Embeddings...")
    huggingface_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    embeddings = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=huggingface_token
    )

    print("Building FAISS index...")
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    print("FAISS retriever instantiated successfully!")
    return True

# --- Custom Tool Definitions ---

@tool
def query_video_transcript(query: str) -> str:
    """Use this tool to search and answer questions about the YouTube video content and transcript. 
    Input should be a specific search query regarding information mentioned in the video."""
    print(f"\n[Custom Tool Execution: query_video_transcript] Searching database for query: '{query}'...")
    if retriever is None:
        return "Error: FAISS retriever is not initialized."
    try:
        results = retriever.invoke(query)
        context = "\n\n".join(doc.page_content for doc in results)
        print(f"[Custom Tool Execution: query_video_transcript] Retrieved {len(results)} chunks.")
        return context
    except Exception as e:
        return f"Error querying transcript: {e}"

# --- Main Entrypoint ---

def main():
    default_url = "https://www.youtube.com/watch?v=zjkBMFhNj_g"
    video_url = default_url
    query = None

    # Handle command-line arguments:
    # arg 1: video URL
    # arg 2+: query input
    if len(sys.argv) > 1:
        video_url = sys.argv[1]
        if len(sys.argv) > 2:
            query = " ".join(sys.argv[2:])
    else:
        video_url = input(f"Enter YouTube URL (press Enter for default '{default_url}'): ").strip()
        if not video_url:
            video_url = default_url

    # Initialize retriever
    if not init_retriever(video_url):
        return

    # Check for Hugging Face token
    huggingface_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    if not huggingface_token:
        print("ERROR: HUGGINGFACEHUB_API_TOKEN not found in environment. Please check your .env file.")
        return

    # Initialize LLM and Chat Wrapper
    print("\nInitializing API-based Hugging Face Chat LLM...")
    llm = HuggingFaceEndpoint(
        repo_id="deepseek-ai/DeepSeek-V4-Pro",
        task="text-generation",
        max_new_tokens=512,
        temperature=0.1,  # Lower temperature is critical for agent tool format compliance
        huggingfacehub_api_token=huggingface_token
    )
    chat_model = ChatHuggingFace(llm=llm)

    # Instantiate Inbuilt Tools
    print("\nInstantiating Inbuilt Tools (DuckDuckGo Search & Wikipedia)...")
    web_search = DuckDuckGoSearchRun()
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

    # Build the tools list
    tools = [
        query_video_transcript,
        web_search,
        wikipedia
    ]

    # Setup the ReAct prompt template
    react_template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}"""

    prompt = PromptTemplate.from_template(react_template)

    # Create the ReAct Agent
    print("Constructing ReAct Agent...")
    agent = create_react_agent(chat_model, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=8
    )
    print("ReAct Agent successfully constructed!")

    # Execute
    if query:
        print(f"\nRunning query: '{query}'")
        try:
            response = agent_executor.invoke({"input": query})
            print("\n=== Agent Final Response ===")
            print(response["output"])
        except Exception as e:
            print(f"Error executing agent: {e}")
    else:
        print("\n=== YouTube Agent QA Chat (with Web Search & Wikipedia) ===")
        while True:
            user_input = input("\nAsk the Agent a question (or 'q' to quit): ").strip()
            if not user_input or user_input.lower() == 'q':
                break
            
            try:
                response = agent_executor.invoke({"input": user_input})
                print("\n=== Agent Final Response ===")
                print(response["output"])
            except Exception as e:
                print(f"Error executing agent: {e}")

if __name__ == "__main__":
    main()
