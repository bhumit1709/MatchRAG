import { useState, useRef, useEffect, useCallback } from "react";
import ChatMessage from "./components/ChatMessage";
import ThinkingIndicator from "./components/ThinkingIndicator";
import ExampleChips from "./components/ExampleChips";
import "./index.css";

const API_BASE = "http://localhost:5001";

export default function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [serverOnline, setServerOnline] = useState(null); // null=checking, true, false

  const chatEndRef = useRef(null);
  const textareaRef = useRef(null);

  // ── Auto-scroll to bottom on new messages ─────────────
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  // ── Auto-resize textarea ───────────────────────────────
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 120) + "px";
  }, [input]);

  // ── Check server status on mount ──────────────────────
  useEffect(() => {
    fetch(`${API_BASE}/api/status`)
      .then((r) => r.json())
      .then((d) => setServerOnline(d.indexed))
      .catch(() => setServerOnline(false));
  }, []);

  // ── Send a question to the API ─────────────────────────
  const sendMessage = useCallback(async (question) => {
    const q = (question || input).trim();
    if (!q || loading) return;

    setInput("");
    setMessages((prev) => [...prev, { role: "user", text: q }]);
    setLoading(true);

    try {
      const res = await fetch(`${API_BASE}/api/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q }),
      });
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      setMessages((prev) => [
        ...prev,
        { role: "bot", text: data.answer, elapsed: data.elapsed },
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          role: "bot",
          text: `⚠️ Error: ${err.message || "Could not reach the server."}`,
        },
      ]);
    } finally {
      setLoading(false);
    }
  }, [input, loading]);

  // ── Handle keydown (Enter to send, Shift+Enter for newline) ──
  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const isEmpty = messages.length === 0 && !loading;

  return (
    <>
      {/* Ambient orbs */}
      <div className="bg-orbs">
        <div className="orb orb-1" />
        <div className="orb orb-2" />
      </div>

      <div className="app">
        {/* ── Header ── */}
        <header className="header">
          <div className="header-left">
            <span className="header-icon">🏏</span>
            <div>
              <div className="header-title">Cricket Match RAG</div>
              <div className="header-subtitle">India vs West Indies · T20 World Cup</div>
            </div>
          </div>
          <div className="header-badges">
            <span className="badge">nomic-embed-text</span>
            <span className="badge">mistral</span>
            <span className={`badge ${serverOnline ? "badge-green" : ""}`}>
              {serverOnline === null
                ? "● Connecting…"
                : serverOnline
                  ? "● Indexed"
                  : "● Server offline"}
            </span>
          </div>
        </header>

        {/* ── Chat area ── */}
        <main className="chat-area">
          {isEmpty ? (
            <ExampleChips onSelect={(q) => sendMessage(q)} />
          ) : (
            <>
              {messages.map((msg, i) => (
                <ChatMessage
                  key={i}
                  role={msg.role}
                  text={msg.text}
                  elapsed={msg.elapsed}
                />
              ))}
              {loading && <ThinkingIndicator />}
            </>
          )}
          <div ref={chatEndRef} />
        </main>

        {/* ── Input bar ── */}
        <footer className="input-bar">
          <div className="input-row">
            <textarea
              ref={textareaRef}
              className="input-field"
              placeholder="Ask anything about the match…"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              rows={1}
              disabled={loading}
            />
            <button
              className="send-btn"
              onClick={() => sendMessage()}
              disabled={loading || !input.trim()}
              title="Send (Enter)"
            >
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                <line x1="22" y1="2" x2="11" y2="13" />
                <polygon points="22 2 15 22 11 13 2 9 22 2" />
              </svg>
            </button>
          </div>
          <p className="input-hint">
            Enter to send · Shift+Enter for new line · Powered by Ollama (local AI)
          </p>
        </footer>
      </div>
    </>
  );
}
