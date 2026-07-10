import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import "./App.css";

type DiffStatus = "pending" | "accepted" | "rejected";
type Msg =
  | { kind: "user"; text: string }
  | { kind: "assistant"; text: string }
  | { kind: "thinking"; text: string }
  | { kind: "tool"; name: string; args: string; result?: string }
  | { kind: "diff"; file: string; diff: string; status: DiffStatus }
  | { kind: "notice"; text: string }
  | { kind: "error"; text: string };

type Approval = { id: number; action: string; destructive: boolean; grant_key: string | null };
type Checkpoint = { sha: string; label: string };
type HeaderState = {
  model: string;
  provider: string;
  tier: string;
  context: { tokens: number; max: number; percent: number };
};

const WS_URL = "ws://127.0.0.1:8756/ws";
const RESULT_PREVIEW_LIMIT = 3000;

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

function ctxClass(percent: number): string {
  if (percent >= 85) return "ctx-high";
  if (percent >= 60) return "ctx-mid";
  return "ctx-low";
}

function truncate(text: string, limit: number): string {
  return text.length > limit ? text.slice(0, limit) + `\n… (${text.length - limit} chars truncated)` : text;
}

function App() {
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [connected, setConnected] = useState(false);
  const [busy, setBusy] = useState(false);
  const [pending, setPending] = useState<Approval | null>(null);
  const [checkpoints, setCheckpoints] = useState<Checkpoint[]>([]);
  const [rewindTarget, setRewindTarget] = useState<Checkpoint | null>(null);
  const [header, setHeader] = useState<HeaderState | null>(null);
  const [vibe, setVibe] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const streamRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    let ws: WebSocket;
    let retry: ReturnType<typeof setTimeout>;
    const connect = () => {
      ws = new WebSocket(WS_URL);
      wsRef.current = ws;
      ws.onopen = () => {
        setConnected(true);
        ws.send(JSON.stringify({ type: "get_state" }));
      };
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
    switch (ev.type) {
      case "approval_request":
        setPending({
          id: ev.id,
          action: ev.action,
          destructive: !!ev.destructive,
          grant_key: ev.grant_key ?? null,
        });
        return;
      case "state":
        setHeader({ model: ev.model, provider: ev.provider, tier: ev.tier, context: ev.context });
        setVibe(!!ev.vibe);
        // Server truth, newest first — display oldest -> newest.
        setCheckpoints([...(ev.checkpoints ?? [])].reverse());
        return;
      case "vibe_state":
        setVibe(!!ev.enabled);
        return;
      case "checkpoint":
        setCheckpoints((prev) => [...prev, { sha: ev.sha, label: ev.label }]);
        return;
      case "rewind_result":
        if (ev.ok) wsRef.current?.send(JSON.stringify({ type: "get_state" }));
        setMessages((p) => [...p, { kind: ev.ok ? "notice" : "error", text: ev.text }]);
        return;
      case "undo_result":
        setMessages((prev) => {
          const next = prev.slice();
          if (!ev.ok) {
            // Restore failed — the change is still on disk, put the card back.
            for (let i = next.length - 1; i >= 0; i--) {
              const m = next[i];
              if (m.kind === "diff" && m.file === ev.file && m.status === "rejected") {
                next[i] = { ...m, status: "pending" };
                break;
              }
            }
          }
          next.push({ kind: ev.ok ? "notice" : "error", text: ev.text });
          return next;
        });
        return;
    }

    setMessages((prev) => {
      const next = prev.slice();
      const last = next[next.length - 1];
      switch (ev.type) {
        case "content":
          if (last && last.kind === "assistant")
            next[next.length - 1] = { ...last, text: last.text + ev.text };
          else next.push({ kind: "assistant", text: ev.text });
          break;
        case "content_replace":
          if (last && last.kind === "assistant") next[next.length - 1] = { ...last, text: ev.text };
          else next.push({ kind: "assistant", text: ev.text });
          break;
        case "thinking":
          if (last && last.kind === "thinking")
            next[next.length - 1] = { ...last, text: last.text + ev.text };
          else next.push({ kind: "thinking", text: ev.text });
          break;
        case "tool_start":
          next.push({ kind: "tool", name: ev.name, args: JSON.stringify(ev.args, null, 2) });
          break;
        case "tool_end": {
          for (let i = next.length - 1; i >= 0; i--) {
            const m = next[i];
            if (m.kind === "tool" && m.name === ev.name && m.result === undefined) {
              next[i] = { ...m, result: String(ev.result) };
              break;
            }
          }
          break;
        }
        case "diff":
          next.push({ kind: "diff", file: ev.file, diff: ev.diff, status: "pending" });
          break;
        case "notice":
          next.push({ kind: "notice", text: String(ev.text).trim() });
          break;
        case "error":
          next.push({ kind: "error", text: ev.text });
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
    setMessages((p) => [...p, { kind: "user", text }]);
    setInput("");
    setBusy(true);
    wsRef.current?.send(JSON.stringify({ type: "message", text }));
  };

  const stop = () => {
    wsRef.current?.send(JSON.stringify({ type: "stop" }));
  };

  const toggleVibe = () => {
    wsRef.current?.send(JSON.stringify({ type: "vibe", enabled: !vibe }));
  };

  const respond = (decision: "deny" | "once" | "always") => {
    if (!pending) return;
    wsRef.current?.send(JSON.stringify({ type: "approval_reply", id: pending.id, decision }));
    const verb = decision === "deny" ? "denied" : decision === "always" ? "always allowed" : "approved";
    setMessages((p) => [...p, { kind: "notice", text: `Approval — ${pending.action}: ${verb}` }]);
    setPending(null);
  };

  const acceptDiff = (index: number) => {
    setMessages((prev) =>
      prev.map((m, i) =>
        i === index && m.kind === "diff" ? { ...m, status: "accepted" as DiffStatus } : m
      )
    );
  };

  const rejectDiff = (index: number, file: string) => {
    // Optimistic: the card flips immediately; a failed undo_result flips it back.
    setMessages((prev) =>
      prev.map((m, i) =>
        i === index && m.kind === "diff" ? { ...m, status: "rejected" as DiffStatus } : m
      )
    );
    wsRef.current?.send(JSON.stringify({ type: "undo_file", file }));
  };

  const confirmRewind = () => {
    if (!rewindTarget) return;
    wsRef.current?.send(JSON.stringify({ type: "rewind", sha: rewindTarget.sha }));
    setRewindTarget(null);
  };

  const renderMsg = (m: Msg, i: number) => {
    switch (m.kind) {
      case "user":
        return (
          <div key={i} className="msg user">
            <div className="bubble">{m.text}</div>
          </div>
        );
      case "assistant":
        return (
          <div key={i} className="msg assistant">
            <div className="bubble md">
              <ReactMarkdown>{m.text}</ReactMarkdown>
            </div>
          </div>
        );
      case "thinking":
        return (
          <details key={i} className="thinking">
            <summary>🧠 Reasoning</summary>
            <div className="thinking-body">{m.text}</div>
          </details>
        );
      case "tool":
        return (
          <details key={i} className="tool-chip">
            <summary>
              <span className={`tool-dot ${m.result === undefined ? "running" : "done"}`} />
              <span className="tool-name">{m.name}</span>
              <span className="tool-state">{m.result === undefined ? "running…" : "done"}</span>
            </summary>
            <div className="tool-detail">
              <div className="tool-section">args</div>
              <pre>{m.args}</pre>
              {m.result !== undefined && (
                <>
                  <div className="tool-section">result</div>
                  <pre>{truncate(m.result, RESULT_PREVIEW_LIMIT)}</pre>
                </>
              )}
            </div>
          </details>
        );
      case "diff":
        return (
          <div key={i} className={`diff-card ${m.status}`}>
            <div className="diff-head">
              <span className="diff-file" title={m.file}>
                {shortFile(m.file)}
              </span>
              {m.status === "pending" ? (
                <span className="diff-actions">
                  <button
                    className="diff-reject"
                    disabled={busy || !connected}
                    title="Restore this file to its pre-edit snapshot"
                    onClick={() => rejectDiff(i, m.file)}
                  >
                    Reject
                  </button>
                  <button className="diff-accept" onClick={() => acceptDiff(i)}>
                    Accept
                  </button>
                </span>
              ) : (
                <span className={`diff-badge ${m.status}`}>
                  {m.status === "accepted" ? "✓ accepted" : "↩ rejected"}
                </span>
              )}
            </div>
            <pre className="diff-body">
              {m.diff.split("\n").map((line, j) => (
                <span key={j} className={diffLineClass(line)}>
                  {line}
                  {"\n"}
                </span>
              ))}
            </pre>
          </div>
        );
      case "notice":
        return (
          <div key={i} className="msg notice">
            <div className="bubble">{m.text}</div>
          </div>
        );
      case "error":
        return (
          <div key={i} className="msg error">
            <div className="bubble">{m.text}</div>
          </div>
        );
    }
  };

  return (
    <div className="app">
      <header className="topbar">
        <span className="brand">Argent</span>
        {header && (
          <span className="header-state">
            <span className="hs-model" title={`provider: ${header.provider}`}>
              {header.model}
            </span>
            <span className="hs-tier">{header.tier}</span>
            <span className={`hs-ctx ${ctxClass(header.context.percent)}`}>
              ctx {header.context.percent}%
            </span>
          </span>
        )}
        <span className="topbar-spacer" />
        <button
          className={`vibe-toggle ${vibe ? "on" : ""}`}
          disabled={!connected}
          title="Vibe mode: auto-approve safe actions + checkpoint every turn"
          onClick={toggleVibe}
        >
          🌴 vibe {vibe ? "on" : "off"}
        </button>
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
        {messages.map(renderMsg)}
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
        {busy ? (
          <button className="stop-btn" onClick={stop} disabled={!connected}>
            Stop
          </button>
        ) : (
          <button onClick={send} disabled={!connected}>
            Send
          </button>
        )}
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
