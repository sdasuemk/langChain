# Tutorial: RAG as a LangChain Tool & ReAct Agents

This folder contains implementations of **ReAct Agents** equipped with custom tools (RAG Tool and Calculator) and **built-in tools** (DuckDuckGo Search and Wikipedia). The agents can answer complex questions by deciding dynamically whether to query the video database, search the live web, look up Wikipedia articles, or perform math operations.

---

## 1. What is a RAG Tool?

In LangChain, a **Tool** is a utility that an agent can interact with. It consists of:
- **A Name**: E.g., `query_video_transcript`.
- **A Description**: A clear docstring explaining *what* the tool does. The agent reads this description to decide whether to call the tool.
- **An Execution Function**: The code that runs when the tool is invoked.

A **RAG Tool** wraps a document retriever. Instead of hardcoding document retrieval into a pipeline, the tool exposes retrieval as an capability that the LLM can call on-demand.

---

## 2. Why Wrap RAG as a Tool? (Agency & Routing)

Direct RAG chains (procedural or LCEL) are static and linear:
```
[User Question] ──► [Retrieve Chunks] ──► [Format Prompt] ──► [LLM Generation]
```
If the user asks: *"What is 10 + 20?"* or *"Who is the President of France?"*, the retriever will still run search queries on the YouTube database, retrieving irrelevant chunks and filling the context window with noise.

By wrapping RAG as a **Tool**, we give the agent **agency**:
- **Selective Retrieval**: If the question is about the video, the agent chooses to call `query_video_transcript`. If it's a general question, the agent answers from its pre-trained weights without wasting database retrieval resources.
- **Multi-Step Reasoning**: If the question requires combining information (e.g., *"What is the size of Llama 2 parameters in gigabits?"*), the agent can:
  1. Retrieve the parameters file size (140 GB) from the RAG tool.
  2. Call the `calculator` tool to multiply `140 * 8` to get the gigabits.
  3. Synthesize the final answer.

---

## 3. How to Define and Use Tools (The Code)

In [`agent_rag.py`](file:///c:/Coding/langChain/16_rag_tool/agent_rag.py), we use LangChain's `@tool` decorator to define our tools:

```python
from langchain_core.tools import tool

# Defining the RAG Tool
@tool
def query_video_transcript(query: str) -> str:
    """Use this tool to search and answer questions about the YouTube video content and transcript.
    Input should be a specific search query regarding information mentioned in the video."""
    results = retriever.invoke(query)
    return "\n\n".join(doc.page_content for doc in results)

# Defining a Calculator Tool
@tool
def calculator(expression: str) -> str:
    """Use this tool to evaluate simple mathematical expressions (e.g. '140 * 8').
    Input should be a string containing numbers and arithmetic operators (+, -, *, /)."""
    clean_expr = "".join(c for c in expression if c in "0123456789+-*/() ")
    return str(eval(clean_expr))
```

---

## 4. How to Create Your Own Custom Tool (Step-by-Step)

Creating your own custom tool in LangChain is very straightforward. Follow these steps:

1.  **Import the `@tool` decorator**:
    ```python
    from langchain_core.tools import tool
    ```
2.  **Define your function and decorate it**: Use the `@tool` decorator above your Python function.
3.  **Add Type Annotations**: Declare the type of input arguments and the return type.
4.  **Add a Docstring**: Write a clear description of **what** the tool does and **when** it should be called. The LLM reads this description to decide whether it needs to invoke your tool.

### Simple Code Example (Weather Tool)
```python
from langchain_core.tools import tool

@tool
def get_current_weather(location: str) -> str:
    """
    Use this tool to find the current weather temperature and conditions for a specific city location.
    The input parameter 'location' must be a city name (e.g., 'London' or 'Paris').
    """
    # Inside your tool, implement the logic (API calls, database query, etc.)
    mock_weather_db = {
        "london": "18°C, Cloudy with light rain.",
        "paris": "22°C, Sunny and clear skies.",
        "new york": "25°C, Humid with thunderstorms."
    }
    city = location.lower().strip()
    return mock_weather_db.get(city, f"Could not find weather data for '{location}'.")
```

---

## 5. ReAct Agent Reasoning Trace

We use the standard **ReAct (Reasoning and Acting)** framework loop:
```
Thought ──► Action (Select Tool) ──► Action Input ──► Observation (Tool Result)
   ▲                                                            │
   └─────────────────── Repeat loop if needed ──────────────────┘
```

Here is a step-by-step example trace for the query:
> *"What are the files that make up llama 2, and what is the size of the parameters file in gigabits (multiply gigabytes by 8)?"*

1.  **Thought**: I need to find out what files make up Llama 2 and their size. I should search the video transcript first.
2.  **Action**: `query_video_transcript`
3.  **Action Input**: `files that make up Llama 2 and parameters file size`
4.  **Observation**: The Llama 2 model consists of two files: the `parameters` file (140 gigabytes because it has 70 billion parameters, 2 bytes each) and the `run` file.
5.  **Thought**: I now know the parameters file is 140 GB. I need to calculate its size in gigabits by multiplying 140 by 8.
6.  **Action**: `calculator`
7.  **Action Input**: `140 * 8`
8.  **Observation**: `1120`
9.  **Thought**: I now have the files and the calculated size in gigabits. I can formulate the final answer.
10. **Final Answer**: Llama 2 consists of two files: the parameters file (140 GB) and the run file. The size of the parameters file in gigabits is 1,120 gigabits.

---

## 6. Execution & Verification (Custom Tools)

To run the agent equipped with custom tools (RAG + Calculator):
```powershell
.venv\Scripts\python 16_rag_tool/agent_rag.py "https://www.youtube.com/watch?v=zjkBMFhNj_g" "What are the two files that make up llama 2, and what is the parameters file size in gigabits (multiply gigabytes by 8)?"
```
This initializes the database, binds the tools to the DeepSeek model, executes the ReAct reasoning loops, and outputs the final answer.

---

## 7. Built-in Tools (DuckDuckGo & Wikipedia)

LangChain provides a rich set of **built-in tools** for common tasks like web search, database querying, code execution, and looking up articles.

In [`agent_inbuilt_tools.py`](file:///c:/Coding/langChain/16_rag_tool/agent_inbuilt_tools.py), we integrate:
1.  **DuckDuckGo Search (`DuckDuckGoSearchRun`)**: Used to perform live, rate-limit-friendly web searches to get real-time internet information.
2.  **Wikipedia (`WikipediaQueryRun`)**: Wraps the Wikipedia API to retrieve rich summaries of encyclopedic concepts.

### How Built-in Tools are Instantiated
Unlike custom tools defined with `@tool`, built-in tools are classes that we import and instantiate directly:

```python
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

# 1. Instantiate the live web search tool
web_search = DuckDuckGoSearchRun()

# 2. Instantiate the Wikipedia query tool with its API wrapper
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
```

These tools have their own pre-configured names, descriptions, and logic, making them ready to be bound to any agent immediately.

---

## 8. Execution & Verification (Built-in Tools)

To run the agent equipped with both custom RAG and built-in search/Wikipedia tools:
```powershell
.venv\Scripts\python 16_rag_tool/agent_inbuilt_tools.py "https://www.youtube.com/watch?v=zjkBMFhNj_g" "What are the two files that make up llama 2? Also, search the web to find when Llama 3 was released."
```
This query triggers:
1.  `query_video_transcript` to find the Llama 2 file names.
2.  `duckduckgo_search` to find the release date of Llama 3 on the live web.
3.  The agent combines both findings into a unified output!
