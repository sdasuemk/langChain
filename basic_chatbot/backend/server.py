import json
import uuid
import importlib.util
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

# 1. Load Environment Variables
env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)
load_dotenv()

# 2. Import the LangGraph Compiled App from the separate graph file (35_basic_chatbot.py)
GRAPH_FILE = Path(__file__).resolve().parent / "35_basic_chatbot.py"
spec = importlib.util.spec_from_file_location("chatbot_graph", GRAPH_FILE)
graph_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(graph_module)

app_graph = graph_module.app
MODEL_NAME = getattr(graph_module.endpoint, "repo_id", "deepseek-ai/DeepSeek-V4-Pro")

# 3. FastAPI Application
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
        "default_thread": "default_thread",
        "graph_source": str(GRAPH_FILE.name),
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
    return {"status": "cleared", "thread_id": thread_id}

@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    thread_id = req.thread_id or str(uuid.uuid4())
    user_text = req.message.strip()

    if not user_text:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    config = {"configurable": {"thread_id": thread_id}}

    def event_generator():
        # 1. Immediate thinking event for client UI animation
        yield f"data: {json.dumps({'type': 'thinking', 'status': 'Thinking...'})}\n\n"

        full_response = ""
        first_token = True

        try:
            # Stream tokens directly from the imported LangGraph instance
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
