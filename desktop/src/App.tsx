import { useEffect, useRef, useState } from "react";
import "./App.css";

type Role = "user" | "assistant" | "tool" | "notice" | "error";
type Msg = { role: Role; text: string };

const WS_URL = "ws://127.0.0.1:8756/ws";

function App() {
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [connected, setConnected] = useState(false);
  const [busy, setBusy] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const streamRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    let ws: WebSocket;
    let retry: ReturnType<typeof setTimeout>;
    const connect = () => {
      ws = new WebSocket(WS_URL);
      wsRef.current = ws;
      ws.onopen = () => setConnected(true);
      ws.onclose = () => {
        setConnected(false);
        retry = setTimeout(connect, 1500);
      };
      ws.onerror = () => ws.close();
      ws.onmessage = (e) => handleEvent(JSON.parse(e.data));
    };
    connect();
    return () => {
      clearTimeout(retry);
      ws && ws.close();
    };
  }, []);

  useEffect(() => {
    const el = streamRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages, busy]);

  const handleEvent = (ev: any) => {
    setMessages((prev) => {
      const next = prev.slice();
      const last = next[next.length - 1];
      const toAssistant = (text: string, replace = false) => {
        if (last && last.role === "assistant")
          next[next.length - 1] = { ...last, text: replace ? text : last.text + text };
        else next.push({ role: "assistant", text });
      };
      switch (ev.type) {
        case "content":
          toAssistant(ev.text);
          break;
        case "content_replace":
          toAssistant(ev.text, true);
          break;
        case "tool_start":
          next.push({ role: "tool", text: `→ ${ev.name}(${JSON.stringify(ev.args)})` });
          break;
        case "tool_end":
          next.push({ role: "tool", text: `✓ ${ev.name}` });
          break;
        case "notice":
          next.push({ role: "notice", text: String(ev.text).trim() });
          break;
        case "error":
          next.push({ role: "error", text: ev.text });
          break;
        case "done":
          setBusy(false);
          break;
      }
      return next;
    });
  };

  const send = () => {
    const text = input.trim();
    if (!text || !connected || busy) return;
    setMessages((p) => [...p, { role: "user", text }]);
    setInput("");
    setBusy(true);
    wsRef.current?.send(JSON.stringify({ type: "message", text }));
  };

  return (
    <div className="app">
      <header className="topbar">
        <span className="brand">Argent</span>
        <span className={`status ${connected ? "ok" : "off"}`}>
          {connected ? "connected" : "offline"}
        </span>
      </header>

      <div className="stream" ref={streamRef}>
        {messages.length === 0 && (
          <div className="empty">
            {connected
              ? "Ask Argent anything to begin."
              : "Backend offline — run  python argent_server.py"}
          </div>
        )}
        {messages.map((m, i) => (
          <div key={i} className={`msg ${m.role}`}>
            <div className="bubble">{m.text}</div>
          </div>
        ))}
        {busy && (
          <div className="msg assistant">
            <div className="bubble typing">…</div>
          </div>
        )}
      </div>

      <div className="composer">
        <textarea
          value={input}
          placeholder={
            connected ? "Message Argent…   (Enter to send, Shift+Enter for newline)" : "Backend offline"
          }
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              send();
            }
          }}
        />
        <button onClick={send} disabled={!connected || busy}>
          Send
        </button>
      </div>
    </div>
  );
}

export default App;
