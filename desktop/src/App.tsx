import { useEffect, useRef, useState } from "react";
import "./App.css";

type Role = "user" | "assistant" | "tool" | "notice" | "error";
type Msg = { role: Role; text: string };
type Approval = { id: number; action: string; destructive: boolean; grant_key: string | null };

const WS_URL = "ws://127.0.0.1:8756/ws";

function App() {
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [connected, setConnected] = useState(false);
  const [busy, setBusy] = useState(false);
  const [pending, setPending] = useState<Approval | null>(null);
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
    if (ev.type === "approval_request") {
      setPending({
        id: ev.id,
        action: ev.action,
        destructive: !!ev.destructive,
        grant_key: ev.grant_key ?? null,
      });
      return;
    }
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

  const respond = (decision: "deny" | "once" | "always") => {
    if (!pending) return;
    wsRef.current?.send(JSON.stringify({ type: "approval_reply", id: pending.id, decision }));
    const verb = decision === "deny" ? "denied" : decision === "always" ? "always allowed" : "approved";
    setMessages((p) => [...p, { role: "notice", text: `Approval — ${pending.action}: ${verb}` }]);
    setPending(null);
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

      {pending && (
        <div className="approval-overlay">
          <div className={`approval-card ${pending.destructive ? "danger" : ""}`}>
            <div className="approval-title">
              {pending.destructive ? "⚠️ Destructive action" : "Permission required"}
            </div>
            <div className="approval-action">{pending.action}</div>
            <div className="approval-actions">
              <button className="deny" onClick={() => respond("deny")}>
                Deny
              </button>
              {pending.grant_key && !pending.destructive && (
                <button className="always" onClick={() => respond("always")}>
                  Always allow ‘{pending.grant_key}’
                </button>
              )}
              <button className="approve" onClick={() => respond("once")}>
                Approve
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
