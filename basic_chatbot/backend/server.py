import os
import json
import uuid
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root or current working dir
env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END, add_messages
from typing import Annotated, TypedDict
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langgraph.checkpoint.memory import MemorySaver

# 1. Initialize HuggingFace Model
MODEL_NAME = "deepseek-ai/DeepSeek-V4-Pro"
endpoint = HuggingFaceEndpoint(
    repo_id=MODEL_NAME,
    task="text-generation",
    max_new_tokens=512,
    temperature=0.7,
)
llm = ChatHuggingFace(llm=endpoint)

# 2. Define State
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# 3. Node Logic
def chatbot_node(state: ChatState) -> dict:
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

# 4. Checkpointer & Graph Compilation
checkpoint_saver = MemorySaver()
graph_builder = StateGraph(ChatState)
graph_builder.add_node("chatbot", chatbot_node)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

app_graph = graph_builder.compile(checkpointer=checkpoint_saver)

# 5. FastAPI Application
app = FastAPI(title="LangGraph HuggingFace Chatbot API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = "default_thread"

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "default_thread": "default_thread"
    }

@app.get("/api/history/{thread_id}")
def get_history(thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    state = app_graph.get_state(config)
    messages_data = []
    if state and state.values and "messages" in state.values:
        for msg in state.values["messages"]:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            messages_data.append({"role": role, "content": msg.content})
    return {"thread_id": thread_id, "messages": messages_data}

@app.delete("/api/history/{thread_id}")
def clear_history(thread_id: str):
    # Simply create a new thread or reset
    return {"status": "cleared", "thread_id": thread_id}

@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    thread_id = req.thread_id or str(uuid.uuid4())
    user_text = req.message.strip()

    if not user_text:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    config = {"configurable": {"thread_id": thread_id}}

    def event_generator():
        # First send thinking event so frontend shows thinking animation immediately
        yield f"data: {json.dumps({'type': 'thinking', 'status': 'Thinking...'})}\n\n"

        full_response = ""
        first_token = True

        try:
            events = app_graph.stream(
                {"messages": [HumanMessage(content=user_text)]},
                stream_mode="messages",
                config=config,
            )

            for msg_chunk, metadata in events:
                if metadata.get("langgraph_node") == "chatbot" and msg_chunk.content:
                    if first_token:
                        yield f"data: {json.dumps({'type': 'start'})}\n\n"
                        first_token = False
                    
                    full_response += msg_chunk.content
                    yield f"data: {json.dumps({'type': 'token', 'token': msg_chunk.content})}\n\n"

            yield f"data: {json.dumps({'type': 'done', 'full_response': full_response, 'thread_id': thread_id})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
