# Tutorial: LangChain Toolkits (What, Why, How)

This guide explains the concept of **Toolkits** in LangChain, why they are used to organize tools, and provides end-to-end examples of both **built-in** and **custom** toolkits.

---

## 1. What is a Toolkit?

A **Toolkit** is a collection of related tools designed to be used together to accomplish a specific domain task. 

For example:
- A **GitHub Toolkit** might contain tools to: `create_issue`, `list_pull_requests`, `commit_code`, and `merge_branch`.
- A **SQL Database Toolkit** might contain tools to: `query_database`, `list_tables`, and `describe_table_schema`.

In LangChain, all toolkits inherit from the `BaseToolkit` class and expose a standard interface: a `.get_tools()` method that returns a list of tools.

---

## 2. Why Use a Toolkit? (Modularity & State Sharing)

While you can define individual tools using `@tool` and pass them in a list, Toolkits provide two major advantages:

1.  **Encapsulation and Simplicity**: Instead of importing 10 separate tools, you import one Toolkit and load all its tools with a single call to `.get_tools()`.
2.  **Shared Configuration/State**: Tools inside a toolkit often need to share resources (like database connections, API keys, browser sessions, or directories). A Toolkit holds this shared configuration in its class properties and passes it to the tools when instantiating them.

---

## 3. How to Use Built-in Toolkits

LangChain provides several built-in toolkits. One common built-in toolkit is the **`FileManagementToolkit`**, which provides tools to safely read, write, list, and search files in a specific local directory.

### Example: Instantiating the File Toolkit
```python
from langchain_community.agent_toolkits import FileManagementToolkit

# Instantiate the toolkit, defining the root folder context
file_toolkit = FileManagementToolkit(root_dir="./sandbox")

# Retrieve the tools list
file_tools = file_toolkit.get_tools()
# This returns tools like: WriteFileTool, ReadFileTool, ListDirectoryTool, etc.
```

---

## 4. How to Create a Custom Toolkit

Creating a custom toolkit involves:
1.  Subclassing **`BaseToolkit`** (which inherits from Pydantic `BaseModel`).
2.  Declaring shared attributes (Pydantic fields) like API clients, active sessions, or retriever objects.
3.  Implementing the **`get_tools(self) -> List[BaseTool]`** method to instantiate and return the associated tools.

### Example: Custom Video RAG Toolkit
Suppose we want to bundle our YouTube QA capabilities into a toolkit that can search the video and download captions:

```python
from typing import List
from langchain_core.tools import BaseToolkit, BaseTool, tool
from langchain_core.vectorstores import VectorStoreRetriever

class VideoRAGToolkit(BaseToolkit):
    """A toolkit for analyzing YouTube video transcripts."""
    
    # Shared retriever state required by the tools
    retriever: VectorStoreRetriever
    
    def get_tools(self) -> List[BaseTool]:
        """Returns the list of tools in this toolkit."""
        
        # Define the tools, referencing the shared retriever state
        @tool
        def query_transcript(query: str) -> str:
            """Queries the YouTube video transcript database for relevant chunks."""
            docs = self.retriever.invoke(query)
            return "\n\n".join(doc.page_content for doc in docs)

        @tool
        def get_video_source() -> str:
            """Gets the original YouTube video ID and URL source metadata."""
            # Accessing the first document's metadata via retriever
            docs = self.retriever.invoke("")
            if docs:
                return f"Source ID: {docs[0].metadata.get('source')} | URL: {docs[0].metadata.get('url')}"
            return "No metadata found."

        return [query_transcript, get_video_source]
```

---

## 5. Simple End-to-End Example Script

Here is a complete, runnable script showing how to load a **built-in File Toolkit** and a **custom RAG Toolkit** together into a single agent.

```python
import os
import sys
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import BaseToolkit, BaseTool, tool

# Built-in File Toolkit imports
from langchain_community.agent_toolkits import FileManagementToolkit

# Import our transcript fetching helper
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "15_langchain_rag")))
from youtube_transcript import fetch_youtube_transcript

load_dotenv()

# --- 1. Define Custom YouTube RAG Toolkit ---

class YouTubeRAGToolkit(BaseToolkit):
    """Custom toolkit containing tools to query the YouTube video database."""
    retriever: any  # We pass the FAISS retriever
    
    def get_tools(self) -> list[BaseTool]:
        @tool
        def query_video_transcript(query: str) -> str:
            """Use this tool to search and answer questions about the YouTube video content and transcript."""
            results = self.retriever.invoke(query)
            return "\n\n".join(doc.page_content for doc in results)
            
        return [query_video_transcript]

# --- 2. Main Script Setup ---

def main():
    # Setup our video retriever (FAISS database)
    url = "https://www.youtube.com/watch?v=zjkBMFhNj_g"
    # (Initialize embeddings, database, etc.)
    # retriever = ...
    
    # Instantiate Custom YouTube RAG Toolkit
    video_toolkit = YouTubeRAGToolkit(retriever=retriever)
    video_tools = video_toolkit.get_tools()
    
    # Instantiate Built-in File Management Toolkit
    # This gives the agent tools like 'write_file', 'read_file', 'list_dir'
    os.makedirs("./workspace_sandbox", exist_ok=True)
    file_toolkit = FileManagementToolkit(root_dir="./workspace_sandbox")
    file_tools = file_toolkit.get_tools()
    
    # Combine tools list
    all_tools = video_tools + file_tools
    
    # Initialize Conversational LLM
    llm = HuggingFaceEndpoint(
        repo_id="deepseek-ai/DeepSeek-V4-Pro",
        task="text-generation",
        max_new_tokens=512,
        temperature=0.1,
        huggingfacehub_api_token=os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    )
    chat_model = ChatHuggingFace(llm=llm)
    
    # Construct Agent
    react_template = """You have access to the following tools:
{tools}

Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (repeat Thought/Action/Action Input/Observation)
Thought: I now know the final answer
Final Answer: the final answer

Begin!
Question: {input}
Thought: {agent_scratchpad}"""

    prompt = PromptTemplate.from_template(react_template)
    agent = create_react_agent(chat_model, all_tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=all_tools, verbose=True)
    
    # Run a multi-toolkit query:
    # 1. Search Llama 2 file names using the YouTube RAG Toolkit.
    # 2. Write those file names to a text file using the File Management Toolkit.
    query = (
        "Find the two files that make up llama 2 from the video transcript. "
        "Then, write those file names to a text file named 'llama_files.txt' using your file writing tool."
    )
    print(f"\nRunning Query: {query}")
    agent_executor.invoke({"input": query})

if __name__ == "__main__":
    main()
```

---

## 6. Chaining Toolkit Tools in LCEL Pipelines

Because all tools retrieved from a toolkit inherit from `BaseTool` (which implements the `Runnable` protocol), they can be piped directly into LCEL chains using the bitwise OR `|` operator, just like any other LangChain component.

### Example: Tool Pipeline Chaining

Suppose we want to read a configuration file using the `ReadFileTool` from the `FileManagementToolkit`, parse it, and then query our custom `query_video_transcript` tool using that parsed parameter.

Instead of running an agent, we can build a static, deterministic pipeline chain:

```python
from langchain_core.runnables import RunnableLambda
from langchain_community.agent_toolkits import FileManagementToolkit

# 1. Instantiate the toolkit and retrieve individual tools
file_toolkit = FileManagementToolkit(root_dir="./workspace_sandbox")
tools_dict = {t.name: t for t in file_toolkit.get_tools()}

# Retrieve specific tools by name from the toolkit
read_tool = tools_dict["read_file"]

# 2. Define a parsing step
def extract_topic_from_config(file_content: str) -> str:
    # E.g. parses "topic=llama2" from file
    for line in file_content.splitlines():
        if line.startswith("topic="):
            return line.split("=")[1].strip()
    return "llama 2"

# 3. Create the LCEL Pipeline Chain
# Read file -> Extract query topic -> Query Video RAG Tool
lcel_tool_pipeline = (
    RunnableLambda(lambda x: {"file_path": x})  # Format input dictionary
    | read_tool                                 # Run read tool from toolkit
    | RunnableLambda(extract_topic_from_config) # Parse file content
    | query_video_transcript                    # Run custom RAG tool
)

# 4. Invoke the chain directly
# This reads 'config.txt', extracts the topic, and queries the RAG database
result = lcel_tool_pipeline.invoke("config.txt")
print(result)
```

In this pipeline, data flows seamlessly from a **built-in toolkit tool** (`read_file`) through custom parsing, and finally into a **custom tool** (`query_video_transcript`), without requiring agent reasoning loops!

---

## 7. Layman Example: A Custom Calculator Toolkit & LCEL Chaining

Here is a super simple, layman-friendly example. We will:
1. Define two custom calculator tools: one that adds 5, and one that multiplies by 2.
2. Group them together in a custom `SimpleCalculatorToolkit`.
3. Extract the tools from the toolkit.
4. Chain them together in a pipeline using `|` to compute: `(Input + 5) * 2`.

### Complete Layman Script Example

```python
from typing import List
from langchain_core.tools import BaseToolkit, BaseTool, tool
from langchain_core.runnables import RunnableLambda

# Step 1: Define your custom tools
@tool
def add_five(x: int) -> int:
    """Adds 5 to the input number."""
    return x + 5

@tool
def multiply_by_two(x: int) -> int:
    """Multiplies the input number by 2."""
    return x * 2


# Step 2: Group them inside a Custom Toolkit
class SimpleCalculatorToolkit(BaseToolkit):
    """A toolkit that groups basic math tools."""
    
    def get_tools(self) -> List[BaseTool]:
        # Returns our custom math tools
        return [add_five, multiply_by_two]


# Step 3: Instantiate toolkit and retrieve tools
calculator_toolkit = SimpleCalculatorToolkit()
tools = calculator_toolkit.get_tools()

add_tool = tools[0]       # The add_five tool
multiply_tool = tools[1]  # The multiply_by_two tool


# Step 4: Chain them in a pipeline using |
# Since add_tool outputs a raw number (e.g. 15), and multiply_tool expects a dictionary 
# input matching its schema {"x": value}, we place a RunnableLambda in between to format the data.
math_pipeline = (
    add_tool 
    | RunnableLambda(lambda out: {"x": int(out)}) 
    | multiply_tool
)


# Step 5: Invoke the pipeline chain
# Goal: Compute (10 + 5) * 2
result = math_pipeline.invoke({"x": 10})
print("Result of (10 + 5) * 2 is:", result) # Output: 30
```

