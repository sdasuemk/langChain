import React, { useState, useEffect, useRef } from 'react';
import {
  Send,
  Bot,
  User,
  Sparkles,
  Plus,
  Trash2,
  Copy,
  Check,
  PanelLeftClose,
  PanelLeftOpen,
  RefreshCw,
  Cpu,
  MessageSquare
} from 'lucide-react';

const API_BASE_URL = 'http://localhost:8000';

const SUGGESTIONS = [
  {
    prompt: "What is the capital of India and what makes it special?",
    tag: "Geography & Facts"
  },
  {
    prompt: "How does LangGraph state checkpointing work with MemorySaver?",
    tag: "LangGraph Architecture"
  },
  {
    prompt: "Explain how reducers and add_messages combine conversation state.",
    tag: "AI Engineering"
  },
  {
    prompt: "Write a python function to binary search an array with explanations.",
    tag: "Python Code"
  }
];

export default function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isThinking, setIsThinking] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [threadId, setThreadId] = useState('thread_1');
  const [threads, setThreads] = useState(['thread_1']);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [serverOnline, setServerOnline] = useState(false);
  const [copiedId, setCopiedId] = useState(null);
  const [toast, setToast] = useState('');

  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);

  // Auto-scroll to bottom of messages
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isThinking]);

  // Check backend server health
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/api/health`, { method: 'GET' });
        if (res.ok) {
          setServerOnline(true);
        } else {
          setServerOnline(false);
        }
      } catch (err) {
        setServerOnline(false);
      }
    };
    checkHealth();
    const interval = setInterval(checkHealth, 6000);
    return () => clearInterval(interval);
  }, []);

  // Fetch history when threadId changes
  useEffect(() => {
    const fetchHistory = async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/api/history/${threadId}`);
        if (res.ok) {
          const data = await res.json();
          if (data.messages && data.messages.length > 0) {
            setMessages(
              data.messages.map((m, idx) => ({
                id: `history-${idx}`,
                role: m.role,
                content: m.content,
                timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
              }))
            );
            return;
          }
        }
      } catch (err) {
        // server might not be running or empty thread
      }
      setMessages([]);
    };
    fetchHistory();
  }, [threadId]);

  const showToast = (msg) => {
    setToast(msg);
    setTimeout(() => setToast(''), 2500);
  };

  const handleCopy = (id, text) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    showToast('Copied to clipboard');
    setTimeout(() => setCopiedId(null), 2000);
  };

  const createNewThread = () => {
    const newId = `thread_${Date.now().toString().slice(-4)}`;
    setThreads(prev => [newId, ...prev]);
    setThreadId(newId);
    setMessages([]);
    showToast(`Started new conversation: ${newId}`);
  };

  const clearCurrentChat = async () => {
    try {
      await fetch(`${API_BASE_URL}/api/history/${threadId}`, { method: 'DELETE' });
    } catch (e) {
      // ignore
    }
    setMessages([]);
    showToast('Conversation cleared');
  };

  const handleSend = async (textToSend) => {
    const text = (textToSend || input).trim();
    if (!text || isStreaming) return;

    setInput('');
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }

    const userMessageId = `user-${Date.now()}`;
    const botMessageId = `bot-${Date.now()}`;
    const timestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    // Append user message
    setMessages(prev => [
      ...prev,
      { id: userMessageId, role: 'user', content: text, timestamp }
    ]);

    setIsThinking(true);
    setIsStreaming(true);

    try {
      const response = await fetch(`${API_BASE_URL}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text, thread_id: threadId })
      });

      if (!response.ok) {
        throw new Error(`Server returned HTTP ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let botContent = '';
      let addedBotMessage = false;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunkStr = decoder.decode(value, { stream: true });
        const lines = chunkStr.split('\n');

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const jsonStr = line.replace('data: ', '').trim();
          if (!jsonStr) continue;

          try {
            const data = JSON.parse(jsonStr);

            if (data.type === 'thinking') {
              setIsThinking(true);
            } else if (data.type === 'start') {
              setIsThinking(false);
              if (!addedBotMessage) {
                setMessages(prev => [
                  ...prev,
                  { id: botMessageId, role: 'assistant', content: '', timestamp }
                ]);
                addedBotMessage = true;
              }
            } else if (data.type === 'token') {
              setIsThinking(false);
              if (!addedBotMessage) {
                setMessages(prev => [
                  ...prev,
                  { id: botMessageId, role: 'assistant', content: '', timestamp }
                ]);
                addedBotMessage = true;
              }
              botContent += data.token;
              setMessages(prev =>
                prev.map(m => (m.id === botMessageId ? { ...m, content: botContent } : m))
              );
            } else if (data.type === 'done') {
              setIsThinking(false);
              setIsStreaming(false);
            } else if (data.type === 'error') {
              setIsThinking(false);
              setIsStreaming(false);
              setMessages(prev => [
                ...prev,
                {
                  id: `err-${Date.now()}`,
                  role: 'assistant',
                  content: `⚠️ Error: ${data.error}`,
                  timestamp
                }
              ]);
            }
          } catch (e) {
            console.error('Error parsing SSE event:', e);
          }
        }
      }
    } catch (err) {
      console.warn('Backend unavailable, running in local fallback mode:', err);
      // Fallback response for offline demonstration
      setIsThinking(false);
      setTimeout(() => {
        setMessages(prev => [
          ...prev,
          {
            id: botMessageId,
            role: 'assistant',
            content: `⚠️ Could not reach backend server at ${API_BASE_URL}.\n\nPlease ensure your FastAPI server is running with:\n\`uvicorn server:app --reload\` inside \`basic_chatbot/backend\`.`,
            timestamp
          }
        ]);
        setIsStreaming(false);
      }, 500);
    } finally {
      setIsThinking(false);
      setIsStreaming(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleTextareaInput = (e) => {
    setInput(e.target.value);
    e.target.style.height = 'auto';
    e.target.style.height = `${Math.min(e.target.scrollHeight, 160)}px`;
  };

  return (
    <div className="app-container">
      {/* Toast Notification */}
      {toast && (
        <div className="toast">
          <Check size={16} color="#10b981" />
          <span>{toast}</span>
        </div>
      )}

      {/* Sidebar */}
      <aside className={`sidebar ${sidebarOpen ? '' : 'collapsed'}`}>
        <div className="sidebar-header">
          <div className="brand-badge">
            <div className="brand-icon-box">
              <Bot size={20} />
            </div>
            <div>
              <div className="brand-title">LangGraph AI</div>
              <div className="brand-subtitle">DeepSeek-V4-Pro</div>
            </div>
          </div>
        </div>

        <div className="sidebar-actions">
          <button id="new-chat-btn" className="new-chat-btn" onClick={createNewThread}>
            <Plus size={16} />
            <span>New Chat</span>
          </button>
        </div>

        <div className="threads-list">
          <span className="threads-label">Saved Sessions (MemorySaver)</span>
          {threads.map(tid => (
            <div
              key={tid}
              className={`thread-item ${tid === threadId ? 'active' : ''}`}
              onClick={() => setThreadId(tid)}
            >
              <div className="thread-info">
                <MessageSquare size={14} />
                <span>{tid}</span>
              </div>
            </div>
          ))}
        </div>

        <div className="sidebar-footer">
          <div className="model-pill">
            <span style={{ display: 'flex', alignItems: 'center' }}>
              <span
                className="status-dot"
                style={{
                  backgroundColor: serverOnline ? '#10b981' : '#f43f5e',
                  boxShadow: `0 0 8px ${serverOnline ? '#10b981' : '#f43f5e'}`
                }}
              />
              {serverOnline ? 'Backend Online' : 'Backend Offline'}
            </span>
            <span style={{ fontSize: '0.7rem', color: 'var(--text-dim)' }}>Port 8000</span>
          </div>
        </div>
      </aside>

      {/* Main Chat Interface */}
      <main className="chat-main">
        {/* Topbar */}
        <header className="chat-topbar">
          <div className="topbar-left">
            <button
              id="toggle-sidebar-btn"
              className="icon-btn"
              onClick={() => setSidebarOpen(!sidebarOpen)}
              title={sidebarOpen ? 'Collapse Sidebar' : 'Expand Sidebar'}
            >
              {sidebarOpen ? <PanelLeftClose size={18} /> : <PanelLeftOpen size={18} />}
            </button>
            <div className="chat-title-group">
              <h1>
                Chatbot
                <span className="chip">LangGraph Checkpoint</span>
              </h1>
            </div>
          </div>

          <div className="topbar-right">
            <button
              id="clear-chat-btn"
              className="icon-btn"
              onClick={clearCurrentChat}
              title="Clear Current Chat"
            >
              <Trash2 size={16} />
            </button>
          </div>
        </header>

        {/* Message Flow Area */}
        <div className="messages-container">
          {messages.length === 0 && !isThinking ? (
            <div className="welcome-screen">
              <div className="welcome-logo">
                <Sparkles size={32} />
              </div>
              <h2 className="welcome-title">How can I assist you today?</h2>
              <p className="welcome-desc">
                Powered by LangGraph conversational state, MemorySaver checkpointer, and Hugging Face's DeepSeek model.
              </p>

              <div className="suggestions-grid">
                {SUGGESTIONS.map((s, idx) => (
                  <div
                    key={idx}
                    className="suggestion-card"
                    onClick={() => handleSend(s.prompt)}
                  >
                    <div className="suggestion-prompt">{s.prompt}</div>
                    <div className="suggestion-tag">{s.tag}</div>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <>
              {messages.map((msg) => (
                <div key={msg.id} className={`message-row ${msg.role}`}>
                  {msg.role === 'assistant' && (
                    <div className="avatar bot">
                      <Bot size={18} />
                    </div>
                  )}

                  <div className="message-content-wrapper">
                    <div className="message-bubble">
                      <div style={{ whiteSpace: 'pre-wrap' }}>{msg.content}</div>
                      {isStreaming && msg.id === messages[messages.length - 1]?.id && (
                        <span className="typing-cursor" />
                      )}
                    </div>

                    <div className="message-meta">
                      <span>{msg.timestamp}</span>
                      {msg.role === 'assistant' && (
                        <button
                          className="action-icon-btn"
                          onClick={() => handleCopy(msg.id, msg.content)}
                          title="Copy message"
                        >
                          {copiedId === msg.id ? <Check size={12} /> : <Copy size={12} />}
                          <span>{copiedId === msg.id ? 'Copied' : 'Copy'}</span>
                        </button>
                      )}
                    </div>
                  </div>

                  {msg.role === 'user' && (
                    <div className="avatar user">
                      <User size={18} />
                    </div>
                  )}
                </div>
              ))}

              {/* Glowing Thinking Indicator */}
              {isThinking && (
                <div className="message-row assistant">
                  <div className="avatar bot">
                    <Bot size={18} />
                  </div>
                  <div className="message-content-wrapper">
                    <div className="thinking-box">
                      <Sparkles size={16} className="thinking-sparkle" />
                      <span>DeepSeek is thinking...</span>
                    </div>
                  </div>
                </div>
              )}

              <div ref={messagesEndRef} />
            </>
          )}
        </div>

        {/* Input Bar */}
        <div className="chat-input-wrapper">
          <div className="chat-input-container">
            <div className="input-box">
              <textarea
                id="chat-textarea"
                ref={textareaRef}
                className="chat-textarea"
                rows={1}
                placeholder="Message LangGraph Assistant... (Press Enter to send, Shift+Enter for newline)"
                value={input}
                onChange={handleTextareaInput}
                onKeyDown={handleKeyDown}
                disabled={isStreaming}
              />
              <button
                id="send-message-btn"
                className="send-btn"
                onClick={() => handleSend()}
                disabled={!input.trim() || isStreaming}
                title="Send message"
              >
                <Send size={18} />
              </button>
            </div>
            <div className="input-footer-text">
              LangGraph Multi-turn State • Session ID: <code>{threadId}</code> • Hugging Face Inference
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
