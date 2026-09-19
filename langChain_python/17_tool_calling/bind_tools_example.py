import os
import sys
import io
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.tools import tool

# Configure standard output to use UTF-8 encoding (prevents Windows terminal encoding errors)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Load environment variables
load_dotenv()

# --- 1. Define custom tools ---

@tool
def get_current_weather(location: str) -> str:
    """Use this tool to get the current weather and temperature for a specific city location.
    The input parameter 'location' must be a city name (e.g. 'Paris', 'London')."""
    return f"The weather in {location} is currently 22 degrees and sunny."

@tool
def add_numbers(a: int, b: int) -> int:
    """Use this tool to add two numbers together.
    Inputs must be integers."""
    return a + b

def main():
    # Check for Hugging Face token
    huggingface_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    if not huggingface_token:
        print("ERROR: HUGGINGFACEHUB_API_TOKEN not found in environment. Please check your .env file.")
        return

    # --- 2. Initialize LLM and Chat Model wrapper ---
    print("Initializing API-based Hugging Face Chat LLM...")
    llm = HuggingFaceEndpoint(
        repo_id="deepseek-ai/DeepSeek-V4-Pro",
        task="text-generation",
        max_new_tokens=512,
        temperature=0.1,  # Lower temperature is critical for structured tool calling
        huggingfacehub_api_token=huggingface_token
    )
    chat_model = ChatHuggingFace(llm=llm)

    # --- 3. Bind tools to the Chat Model ---
    # This attaches the JSON schemas of get_current_weather and add_numbers to the model.
    print("\nBinding tools [get_current_weather, add_numbers] to the Chat Model...")
    tools = [get_current_weather, add_numbers]
    model_with_tools = chat_model.bind_tools(tools)
    print("Tools bound successfully!")

    # --- 4. Run queries to test Tool Calling behavior ---
    
    # Query A: Triggers the weather tool
    query_a = "What is the weather in Paris right now?"
    if len(sys.argv) > 1:
        query_a = sys.argv[1]

    print(f"\n--- Testing Query A: '{query_a}' ---")
    try:
        response_a = model_with_tools.invoke(query_a)
        
        print("\n=== RAW LLM RESPONSE MESSAGE ===")
        print(response_a)
        
        print("\n=== PARSED RESPONSE CONTENT ===")
        print(f"Text Content: {response_a.content}")
        print(f"Tool Calls:   {response_a.tool_calls}")
        
        if response_a.tool_calls:
            for i, tool_call in enumerate(response_a.tool_calls):
                print(f"\n--- Tool Call #{i+1} ---")
                print(f"  Tool Name: {tool_call['name']}")
                print(f"  Arguments: {tool_call['args']}")
                print(f"  Call ID:   {tool_call['id']}")
        else:
            print("\nNo tool was selected by the model for this query.")
            
    except Exception as e:
        print(f"Error during query execution: {e}")

    # Query B: Triggers no tools (General Knowledge)
    query_b = "What is the capital of France?"
    print(f"\n--- Testing Query B (General Knowledge): '{query_b}' ---")
    try:
        response_b = model_with_tools.invoke(query_b)
        
        print("\n=== PARSED RESPONSE CONTENT ===")
        print(f"Text Content: {response_b.content}")
        print(f"Tool Calls:   {response_b.tool_calls}")
        
        if not response_b.tool_calls:
            print("\nSuccessfully bypassed tool calling for general knowledge query!")
            
    except Exception as e:
        print(f"Error during query execution: {e}")

if __name__ == "__main__":
    main()
