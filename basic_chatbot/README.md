# 🤖 LangGraph + Hugging Face React Chatbot

A modern full-stack conversational AI application built with **LangGraph**, **Hugging Face (`DeepSeek-V4-Pro`)**, **FastAPI**, and **React (Vite)**.

---

## 🌟 Features
- **Real-Time Token Streaming**: Streams tokens live from Hugging Face using SSE (`text/event-stream`).
- **Thinking Indicator**: Glowing animated pulse status while the model prepares responses.
- **Conversation Memory (`MemorySaver`)**: Multi-turn dialogue persistence powered by LangGraph state checkpointing.
- **Thread Management**: Start new chats or switch between saved session threads seamlessly.
- **Modern Glassmorphic Dark UI**: Custom CSS design with Google Fonts (`Outfit` & `Inter`), prompt suggestions, copy-to-clipboard, auto-expanding input, and smooth micro-animations.

---

## 🚀 How to Run

### Step 1: Start the Backend (FastAPI + LangGraph)
Open a terminal in the project root:
```powershell
cd c:\Coding\langChain
.\.venv\Scripts\Activate.ps1
cd basic_chatbot\backend
python server.py
```
> The API will be live at `http://localhost:8000` (Swagger docs available at `http://localhost:8000/docs`).

---

### Step 2: Start the React Frontend (Vite)
Open a second terminal:
```powershell
cd c:\Coding\langChain\basic_chatbot\frontend
npm run dev
```
> Open your browser at `http://localhost:5173`.
