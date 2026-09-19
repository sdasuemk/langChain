# Tutorial: Tool Calling & Tool Binding in LangChain

This folder contains implementations demonstrating both **Tool Binding** (attaching tools to models) and the complete **Tool Calling Execution Loop** (running tool functions and returning results to the model).

---

## 1. What is Tool Calling & Tool Binding?

### What is Tool Calling?
**Tool Calling** is a feature supported by modern chat models (such as OpenAI, Gemini, Claude, Llama 3, and DeepSeek) that allows them to generate structured arguments for functions that you define, rather than just generating conversational text.
*Note*: The model itself does not execute the tool; it only *chooses* which tool to call and generates the *parameters* (arguments) needed to run it. Your application code is responsible for executing the function.

### What is Tool Binding?
**Tool Binding** is the process of attaching your custom tools to a Chat Model. In LangChain, this is done using the `.bind_tools()` method.
```
[Your Custom Tools] ──► .bind_tools() ──► [ChatModel with Tools Bound]
```

---

## 2. Why Use Tool Binding? (vs ReAct Parsing)

In older models, agents used the **ReAct** prompting style. This required the model to output a specific text format (e.g., `Action: calculator` and `Action Input: 10 + 10`) which our code parsed using regular expressions. This was prone to parsing errors if the model deviated from the requested formatting.

**Tool Binding** is superior because:
1.  **API Integration**: Modern LLM APIs support native function calling. The API directly accepts tool schemas and returns structured JSON responses representing tool calls.
2.  **No Regular Expression Parsing**: The model returns a structured `AIMessage` with a pre-parsed `.tool_calls` attribute.
3.  **Automatic JSON Validation**: Arguments are formatted directly as JSON, reducing syntax errors.

---

## 3. How Tool Binding Works (The Code)

In [`bind_tools_example.py`](file:///c:/Coding/langChain/17_tool_calling/bind_tools_example.py), we define tools and bind them to the chat model:

### Step A: Define the Tools
Use the `@tool` decorator. The docstrings and type annotations will form the schema:
```python
from langchain_core.tools import tool

@tool
def get_current_weather(location: str) -> str:
    """Use this tool to get the current weather and temperature for a specific city location."""
    return f"The weather in {location} is currently 22 degrees and sunny."
```

### Step B: Bind Tools to the Model
Use `.bind_tools()` on the instantiated chat model:
```python
# Bind the tools to create a new model reference
model_with_tools = chat_model.bind_tools([get_current_weather])
```

### Step C: Invoke and Inspect Tool Calls
When you call the bound model, it returns an `AIMessage` containing a `.tool_calls` list:
```python
response = model_with_tools.invoke("What is the weather in Paris?")

# Inspect the tool calls list
print(response.tool_calls)
```

---

## 4. The Structure of `tool_calls`

If the model decides to run a tool, `.tool_calls` returns a list of dictionaries in the following format:
```python
[
    {
        'name': 'get_current_weather',     # Name of the tool to execute
        'args': {'location': 'Paris'},     # Parsed argument dictionary
        'id': 'call_abcd1234'              # Unique call ID (sent by the LLM API)
    }
]
```

If the query is a general question (e.g., *"What is the capital of France?"*), the model bypasses tool calling. The `.tool_calls` list will be **empty** (`[]`), and the normal text output will be returned in `.content`.

---

## 5. Execution & Verification (Tool Binding)

To run the tool binding demonstration script:
```powershell
.venv\Scripts\python 17_tool_calling/bind_tools_example.py
```
This script will output:
1.  The raw model response message showing the tool call metadata structure.
2.  The parsed tool name, arguments dictionary, and call ID.
3.  Verification that general queries bypass tool calls and output standard text content.

---

## 6. Topic 2: The Tool Calling Execution Loop

Once the model generates tool calls, the application is responsible for executing them and sending the outputs back to the model. This complete round-trip process is called the **Tool Calling Execution Loop**:

```
[User Query] ──► [Model (bind_tools)] ──(generates tool_calls)──┐
                                                                ▼
[Final Answer] ◄── [Model] ◄── (feeds history) ◄── [Execute Tools Locally]
```

This is implemented in [`tool_calling_example.py`](file:///c:/Coding/langChain/17_tool_calling/tool_calling_example.py).

---

## 7. The Role of `ToolMessage` & `tool_call_id`

When returning tool results to the chat model, we cannot simply append a standard text message. We must use a specialized message type called **`ToolMessage`**.

### Linking Call IDs
Most chat models support calling multiple tools simultaneously (parallel tool calling). To let the LLM know which tool result corresponds to which tool call, the `ToolMessage` must be initialized with the unique `tool_call_id` provided by the model:

```python
from langchain_core.messages import ToolMessage

# 1. Execute the tool locally
result = get_current_weather.invoke(tool_call['args'])

# 2. Wrap result in ToolMessage, binding it to the specific call ID
tool_message = ToolMessage(
    content=str(result), 
    tool_call_id=tool_call['id']  # Matches 'call_abcd1234'
)
```

---

## 8. Complete Message History Flow Diagram

To get a final answer from the model based on the tools, we must send the **entire message history** back to the LLM. The sequence of messages is structured as follows:

```
┌──────────────────┐
│   HumanMessage   │  <── User's original query (e.g. "What is weather in Paris?")
└────────┬─────────┘
         ▼
┌──────────────────┐
│    AIMessage     │  <── Model response containing the structured 'tool_calls' list
└────────┬─────────┘
         ▼
┌──────────────────┐
│   ToolMessage    │  <── Tool result #1 (linked to its unique tool_call_id)
└────────┬─────────┘
         ▼
┌──────────────────┐
│   ToolMessage    │  <── Tool result #2 (linked to its unique tool_call_id)
└────────┬─────────┘
         ▼
┌──────────────────┐
│    AIMessage     │  <── Final synthesized response from the model
└──────────────────┘
```

---

## 9. Execution & Verification (Tool Calling Loop)

To run the complete tool execution loop demonstration:
```powershell
.venv\Scripts\python 17_tool_calling/tool_calling_example.py
```
This script will output:
1.  **Stage 1**: The initial LLM query and the generated structured `tool_calls`.
2.  **Stage 2**: Live local execution of the tools and instantiation of corresponding `ToolMessage` objects.
3.  **Stage 3**: Compilation of the message history containing User, AI, and Tool messages.
4.  **Stage 4**: Invocation of the model with full history to generate the final synthesized conversational response.
