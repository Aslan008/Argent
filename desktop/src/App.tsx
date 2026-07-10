import { useEffect, useRef, useState } from "react";
import "./App.css";

type Role = "user" | "assistant" | "tool" | "notice" | "error";
type DiffStatus = "pending" | "accepted" | "rejected";
type Msg = {
  role: Role;
  text: string;
  diff?: { file: string; diff: string };
  diffStatus?: DiffStatus;
};
type Approval = { id: number; action: string; destructive: boolean; grant_key: string | null };
type Checkpoint = { sha: string; label: string };

const WS_URL = "ws://127.0.0.1:8756/ws";

/** Short display name: last two path segments. */
function shortFile(path: string): string {
  const parts = path.split(/[\\/]/).filter(Boolean);
  return parts.slice(-2).join("/");
}

function diffLineClass(line: string): string {
  if (line.startsWith("+++") || line.startsWith("---")) return "diff-meta";
  if (line.startsWith("@@")) return "diff-hunk";
  if (line.startsWith("+")) return "diff-add";
  if (line.startsWith("-")) return "diff-del";
  return "diff-ctx";
}

function App() {
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [connected, setConnected] = useState(false);
  const [busy, setBusy] = useState(false);
  const [pending, setPending] = useState<Approval | null>(null);
  const [checkpoints, setCheckpoints] = useState<Checkpoint[]>([]);
  const [rewindTarget, setRewindTarget] = useState<Checkpoint | null>(null);
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
    if (ev.type === "checkpoint") {
      setCheckpoints((prev) => [...prev, { sha: ev.sha, label: ev.label }]);
      return;
    }
    if (ev.type === "rewind_result") {
      if (ev.ok) {
        // Everything after the target checkpoint no longer exists.
        setCheckpoints((prev) => {
          const idx = prev.findIndex((c) => c.sha === ev.sha);
          return idx >= 0 ? prev.slice(0, idx + 1) : prev;
        });
      }
      setMessages((p) => [...p, { role: ev.ok ? "notice" : "error", text: ev.text }]);
      return;
    }
    if (ev.type === "undo_result") {
      setMessages((prev) => {
        const next = prev.slice();
        if (!ev.ok) {
          // Restore failed — the change is still on disk, put the card back.
          for (let i = next.length - 1; i >= 0; i--) {
            const m = next[i];
            if (m.diff && m.diff.file === ev.file && m.diffStatus === "rejected") {
              next[i] = { ...m, diffStatus: "pending" };
              break;
            }
          }
        }
        next.push({ role: ev.ok ? "notice" : "error", text: ev.text });
        return next;
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
        case "diff":
          next.push({
            role: "tool",
            text: "",
            diff: { file: ev.file, diff: ev.diff },
            diffStatus: "pending",
          });
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

  const acceptDiff = (index: number) => {
    setMessages((prev) =>
      prev.map((m, i) => (i === index ? { ...m, diffStatus: "accepted" as DiffStatus } : m))
    );
  };

  const rejectDiff = (index: number, file: string) => {
    // Optimistic: the card flips immediately; a failed undo_result flips it back.
    setMessages((prev) =>
      prev.map((m, i) => (i === index ? { ...m, diffStatus: "rejected" as DiffStatus } : m))
    );
    wsRef.current?.send(JSON.stringify({ type: "undo_file", file }));
  };

  const confirmRewind = () => {
    if (!rewindTarget) return;
    wsRef.current?.send(JSON.stringify({ type: "rewind", sha: rewindTarget.sha }));
    setRewindTarget(null);
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
        {messages.map((m, i) =>
          m.diff ? (
            <div key={i} className={`diff-card ${m.diffStatus}`}>
              <div className="diff-head">
                <span className="diff-file" title={m.diff.file}>
                  {shortFile(m.diff.file)}
                </span>
                {m.diffStatus === "pending" ? (
                  <span className="diff-actions">
                    <button
                      className="diff-reject"
                      disabled={busy || !connected}
                      title="Restore this file to its pre-edit snapshot"
                      onClick={() => rejectDiff(i, m.diff!.file)}
                    >
                      Reject
                    </button>
                    <button className="diff-accept" onClick={() => acceptDiff(i)}>
                      Accept
                    </button>
                  </span>
                ) : (
                  <span className={`diff-badge ${m.diffStatus}`}>
                    {m.diffStatus === "accepted" ? "✓ accepted" : "↩ rejected"}
                  </span>
                )}
              </div>
              <pre className="diff-body">
                {m.diff.diff.split("\n").map((line, j) => (
                  <span key={j} className={diffLineClass(line)}>
                    {line}
                    {"\n"}
                  </span>
                ))}
              </pre>
            </div>
          ) : (
            <div key={i} className={`msg ${m.role}`}>
              <div className="bubble">{m.text}</div>
            </div>
          )
        )}
        {busy && (
          <div className="msg assistant">
            <div className="bubble typing">…</div>
          </div>
        )}
      </div>

      {checkpoints.length > 0 && (
        <div className="timeline">
          <span className="timeline-label" title="Time machine — click a node to rewind">
            🕰
          </span>
          {checkpoints.map((c) => (
            <button
              key={c.sha}
              className="timeline-node"
              disabled={busy || !connected}
              title={`${c.label} — rewind to before this turn`}
              onClick={() => setRewindTarget(c)}
            >
              {c.sha}
            </button>
          ))}
        </div>
      )}

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

      {rewindTarget && (
        <div className="approval-overlay">
          <div className="approval-card danger">
            <div className="approval-title">🕰 Rewind</div>
            <div className="approval-action">
              Откатить всё дерево к чекпоинту {rewindTarget.sha} — «{rewindTarget.label}»?
              Незакоммиченные изменения будут сохранены в git stash.
            </div>
            <div className="approval-actions">
              <button className="deny" onClick={() => setRewindTarget(null)}>
                Отмена
              </button>
              <button className="approve" onClick={confirmRewind}>
                Откатить
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
