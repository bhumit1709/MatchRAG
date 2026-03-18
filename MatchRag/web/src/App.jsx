import { useState, useRef, useEffect, useCallback } from "react";
import ChatMessage from "./components/ChatMessage";
import ThinkingIndicator from "./components/ThinkingIndicator";
import ExampleChips from "./components/ExampleChips";
import PipelineInspector from "./components/PipelineInspector";

// Import the downloaded celebration image
import celebrationImg from "./assets/celebration.jpg";

import "./index.css";

const API_BASE = "http://localhost:5001";

function newSessionId() {
  return crypto.randomUUID();
}

export default function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [serverOnline, setServerOnline] = useState(null);
  const [runtimeMeta, setRuntimeMeta] = useState({
    embed_model: "local embeddings",
    llm_model: "local llama.cpp",
  });
  const [sessionId, setSessionId] = useState(() => newSessionId());

  const chatEndRef = useRef(null);
  const textareaRef = useRef(null);
  // Mirrors `messages` synchronously so we can read length before setState commits
  const messagesRef = useRef([]);

  // ── Auto-scroll ────────────────────────────────────────────────────────────
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  // ── Auto-resize textarea ───────────────────────────────────────────────────
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 120) + "px";
  }, [input]);

  // ── Check server status on mount ──────────────────────────────────────────
  useEffect(() => {
    fetch(`${API_BASE}/api/status`)
      .then((r) => r.json())
      .then((d) => {
        setServerOnline(d.indexed);
        setRuntimeMeta({
          embed_model: d.embed_model || "local embeddings",
          llm_model: d.llm_model || "local llama.cpp",
        });
      })
      .catch(() => setServerOnline(false));
  }, []);

  // ── New Chat ───────────────────────────────────────────────────────────────
  const handleNewChat = useCallback(async () => {
    try {
      await fetch(`${API_BASE}/api/session/clear`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId }),
      });
    } catch (_) { }
    const next = newSessionId();
    setSessionId(next);
    setMessages([]);
    messagesRef.current = [];
    setInput("");
  }, [sessionId]);

  // ── Send a question (SSE streaming) ───────────────────────────────────────
  const sendMessage = useCallback(async (question) => {
    const q = (question || input).trim();
    if (!q || loading) return;

    setInput("");
    setLoading(true);

    // ── Compute indices BEFORE any setState so they're stable across renders ─
    // messagesRef.current always reflects the latest committed messages array.
    // React StrictMode invokes setState callbacks twice in dev, so we must NOT
    // derive the index inside a setState callback — use the ref instead.
    const botIdx = messagesRef.current.length + 1; // +1 for the user message

    // Append user message + empty bot placeholder in a single batch
    setMessages((prev) => {
      const next = [
        ...prev,
        { role: "user", text: q },
        { role: "bot", text: "", streaming: true },
      ];
      messagesRef.current = next;
      return next;
    });

    try {
      const res = await fetch(`${API_BASE}/api/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q, session_id: sessionId }),
      });

      if (!res.ok || !res.body) {
        let msg = `HTTP ${res.status}`;
        try { const j = await res.json(); msg = j.error || msg; } catch (_) { }
        throw new Error(msg);
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        let newlineIdx;
        while ((newlineIdx = buffer.indexOf("\n")) !== -1) {
          const line = buffer.slice(0, newlineIdx).trim();
          buffer = buffer.slice(newlineIdx + 1);

          if (!line.startsWith("data: ")) continue;
          let event;
          try { event = JSON.parse(line.slice(6)); } catch (_) { continue; }

          if (event.type === "meta") {
            // Store pipeline metadata on the bot message placeholder
            setMessages((prev) => {
              const updated = [...prev];
              const cur = updated[botIdx] ?? { role: "bot", text: "", streaming: true };
              updated[botIdx] = { ...cur, meta: event };
              messagesRef.current = updated;
              return updated;
            });
          } else if (event.type === "token") {
            const token = event.content;
            setMessages((prev) => {
              const updated = [...prev];
              const cur = updated[botIdx] ?? { role: "bot", text: "", streaming: true };
              updated[botIdx] = { ...cur, text: cur.text + token };
              messagesRef.current = updated;
              return updated;
            });
          } else if (event.type === "done") {
            setMessages((prev) => {
              const updated = [...prev];
              const cur = updated[botIdx] ?? { role: "bot", text: "" };
              const nextMeta = cur.meta
                ? { ...cur.meta, stage_timings_ms: event.stage_timings_ms || cur.meta.stage_timings_ms }
                : cur.meta;
              updated[botIdx] = { ...cur, streaming: false, elapsed: event.elapsed, meta: nextMeta };
              messagesRef.current = updated;
              return updated;
            });
          } else if (event.type === "error") {
            throw new Error(event.message || "Server error");
          }
        }
      }
    } catch (err) {
      const errText = `⚠️ ${err.message || "Could not reach the server."}`;
      setMessages((prev) => {
        const updated = [...prev];
        if (updated[botIdx]) {
          updated[botIdx] = { role: "bot", text: errText, streaming: false };
        } else {
          updated.push({ role: "bot", text: errText, streaming: false });
        }
        messagesRef.current = updated;
        return updated;
      });
    } finally {
      setLoading(false);
    }
  }, [input, loading, sessionId]);

  // ── Keyboard handler ───────────────────────────────────────────────────────
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

      {/* Celebration UI */}
      <div className="celebration-container">
        <img src={celebrationImg} alt="Victory Celebration" className="celebration-bg-image" />
        <div className="celebration-emoji e1">🏆</div>
        <div className="celebration-emoji e2">🇮🇳</div>
        <div className="celebration-emoji e3">✨</div>
        <div className="celebration-emoji e4">🎉</div>
        <div className="celebration-emoji e5">🏏</div>
        <div className="celebration-emoji e6">🎊</div>
      </div>

      <div className="app">
        {/* ── Header ── */}
        <header className="header">
          <div className="header-left">
            <span className="header-icon">🏏</span>
            <div>
              <div className="header-title">Cricket Match RAG</div>
              <div className="header-subtitle">🏆 Champions 2026 · T20 World Cup</div>
            </div>
          </div>
          <div className="header-badges">
            <span className="badge">{runtimeMeta.embed_model}</span>
            <span className="badge">{runtimeMeta.llm_model}</span>
            <span className={`badge ${serverOnline ? "badge-green" : ""}`}>
              {serverOnline === null
                ? "● Connecting…"
                : serverOnline
                  ? "● Indexed"
                  : "● Server offline"}
            </span>
            <button
              className="new-chat-btn"
              onClick={handleNewChat}
              title="Start a new conversation (clears memory)"
            >
              ＋ New Chat
            </button>
          </div>
        </header>

        {/* ── Chat area ── */}
        <main className="chat-area">
          {isEmpty ? (
            <ExampleChips onSelect={(q) => sendMessage(q)} />
          ) : (
            <>
              {messages.map((msg, i) => {
                // Skip rendering the empty bot placeholder — ThinkingIndicator handles that
                if (msg.role === "bot" && msg.streaming && !msg.text) return null;
                return (
                  <div key={i}>
                    <ChatMessage
                      role={msg.role}
                      text={msg.text}
                      elapsed={msg.elapsed}
                      streaming={msg.streaming}
                    />
                    {msg.role === "bot" && msg.meta && (
                      <PipelineInspector meta={msg.meta} />
                    )}
                  </div>
                );
              })}
              {/* Show ThinkingIndicator while waiting for first token */}
              {loading && messages[messages.length - 1]?.text === "" && (
                <ThinkingIndicator />
              )}
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
