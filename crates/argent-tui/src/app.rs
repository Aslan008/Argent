use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::time::Instant;
use crate::events::{ApprovalPrompt, ArgentEvent, HistoryItem};
use crate::renderer::{BlockRenderer, FallbackRenderer};
use crate::speed::SpeedMeter;
pub use crate::selection::{MouseSelection, TextPos};

/// The most empty space that can be scrolled open under the newest output (the
/// frame also keeps it under half the chat's height).
pub const OVERSCROLL_MAX: usize = 20;

/// Wheel notches closer together than this count as one continuous turn.
const WHEEL_STREAK_GAP: std::time::Duration = std::time::Duration::from_millis(70);
use crate::stream::StreamState;
use crate::theme::UiConfig;

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct ApprovalData {
    pub id: usize,
    pub action: String,
    pub destructive: bool,
    pub grant_key: Option<String>,
    /// The REPL's wording of the question; `None` falls back to the same text.
    pub prompt: Option<ApprovalPrompt>,
    /// Row the arrow keys are on in the choice list.
    pub selected: usize,
}

impl ApprovalData {
    /// The question to draw: what the bridge sent, or the REPL's own wording
    /// rebuilt from the same two facts (a session key, destructiveness).
    pub fn resolved_prompt(&self) -> ApprovalPrompt {
        if let Some(p) = &self.prompt {
            return p.clone();
        }
        let question = "Разрешить это действие?".to_string();
        match (&self.grant_key, self.destructive) {
            (Some(key), false) => ApprovalPrompt {
                question,
                choices: vec![
                    "✓ Да (однократно)".to_string(),
                    format!("✓ Да, и всегда разрешать '{key}' в этой сессии"),
                    "✕ Нет".to_string(),
                ],
                decisions: vec!["once".into(), "always".into(), "deny".into()],
                confirm: false,
            },
            _ => ApprovalPrompt {
                question,
                choices: vec!["Да".to_string(), "Нет".to_string()],
                decisions: vec!["once".into(), "deny".into()],
                confirm: true,
            },
        }
    }

    /// The decision under the pointer.
    pub fn selected_decision(&self) -> String {
        let p = self.resolved_prompt();
        p.decisions
            .get(self.selected.min(p.decisions.len().saturating_sub(1)))
            .cloned()
            .unwrap_or_else(|| "deny".to_string())
    }
}

#[derive(Debug, Clone)]
pub struct SelectModalData {
    pub id: String,
    pub title: String,
    pub options: Vec<String>,
    pub selected_idx: usize,
    pub filter: String,
    pub multi: bool,
    pub checked: Vec<bool>,
}

impl SelectModalData {
    pub fn new(id: String, title: String, options: Vec<String>, default_val: Option<String>) -> Self {
        Self::new_full(id, title, options, default_val, false, None)
    }

    pub fn new_full(
        id: String,
        title: String,
        options: Vec<String>,
        default_val: Option<String>,
        multi: bool,
        initial_checked: Option<Vec<bool>>,
    ) -> Self {
        let filter = if options.is_empty() {
            default_val.clone().unwrap_or_default()
        } else {
            String::new()
        };
        let selected_idx = if let Some(ref def) = default_val {
            options.iter().position(|o| o == def).unwrap_or(0)
        } else {
            0
        };
        let checked = match initial_checked {
            Some(mut v) => {
                v.resize(options.len(), false);
                v
            }
            None => vec![false; options.len()],
        };
        Self {
            id,
            title,
            options,
            selected_idx,
            filter,
            multi,
            checked,
        }
    }

    pub fn toggle_current(&mut self) {
        if self.multi && self.selected_idx < self.checked.len() {
            self.checked[self.selected_idx] = !self.checked[self.selected_idx];
        }
    }

    /// Matching options as `(index into self.options, option)`.
    ///
    /// `selected_idx` is always an index into the FULL `options` list, never
    /// into this filtered view — see `move_up`/`move_down`, which translate.
    pub fn filtered_options(&self) -> Vec<(usize, &String)> {
        let f = self.filter.to_lowercase();
        self.options
            .iter()
            .enumerate()
            .filter(|(_, opt)| f.is_empty() || opt.to_lowercase().contains(&f))
            .collect()
    }

    pub fn move_up(&mut self) {
        let filtered = self.filtered_options();
        if filtered.is_empty() {
            return;
        }
        let next = match filtered.iter().position(|(i, _)| *i == self.selected_idx) {
            None | Some(0) => filtered.len() - 1,
            Some(p) => p - 1,
        };
        self.selected_idx = filtered[next].0;
    }

    pub fn move_down(&mut self) {
        let filtered = self.filtered_options();
        if filtered.is_empty() {
            return;
        }
        let next = match filtered.iter().position(|(i, _)| *i == self.selected_idx) {
            None => 0,
            Some(p) => (p + 1) % filtered.len(),
        };
        self.selected_idx = filtered[next].0;
    }

    pub fn selected_value(&self) -> Option<String> {
        if self.multi {
            let selected: Vec<String> = self.options
                .iter()
                .enumerate()
                .filter(|(i, _)| self.checked.get(*i).copied().unwrap_or(false))
                .map(|(_, opt)| opt.clone())
                .collect();
            return Some(serde_json::to_string(&selected).unwrap_or_else(|_| "[]".to_string()));
        }
        let filtered = self.filtered_options();
        if filtered.is_empty() {
            // Nothing matched: treat what the user typed as a free-text answer.
            return if self.filter.is_empty() {
                None
            } else {
                Some(self.filter.clone())
            };
        }
        // Prefer the tracked selection when it survived filtering, else the
        // first match — never the raw filter text while options still match.
        filtered
            .iter()
            .find(|(i, _)| *i == self.selected_idx)
            .or_else(|| filtered.first())
            .map(|(_, opt)| (*opt).clone())
    }

    /// Update tracked selection after the filter text changes.
    /// Preserves the currently selected option if it still matches the filter,
    /// otherwise moves selection to the first matching option.
    pub fn on_filter_changed(&mut self) {
        let filtered = self.filtered_options();
        if !filtered.iter().any(|(orig, _)| *orig == self.selected_idx) {
            if let Some((first_orig, _)) = filtered.first() {
                self.selected_idx = *first_orig;
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct CompletionItem {
    pub name: String,
    pub help: String,
}

/// The input as it was when Tab started cycling through suggestions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompletionCycle {
    /// The whole input and cursor before the first Tab (Esc restores them).
    pub original: String,
    pub original_cursor: usize,
    /// The text before the cursor then; the popup keeps filtering on it.
    pub typed: String,
    /// The text after the cursor, kept when a file mention is completed.
    pub tail: String,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub enum MessageItem {
    User(String),
    Assistant {
        content: String,
        thinking: String,
        thinking_expanded: bool,
        is_streaming_thinking: bool,
        is_streaming_content: bool,
        thinking_duration: Option<f64>,
        thinking_start: Option<Instant>,
        /// Which prefix of `content` has been committed, block by block (see
        /// `stream.rs`). Default for an item that was not built by streaming.
        stream: StreamState,
    },
    /// The model's second-pass verification (`◈ Перепроверка`), a reasoning
    /// block of its own.
    Rethink {
        text: String,
        expanded: bool,
        is_streaming: bool,
        duration: Option<f64>,
        start: Option<Instant>,
    },
    Tool {
        name: String,
        args: HashMap<String, serde_json::Value>,
        result: Option<String>,
        duration_secs: Option<f64>,
        expanded: bool,
        is_running: bool,
    },
    Diff {
        file: String,
        diff: String,
    },
    /// The REPL's own post-response snapshot, as Rich markup lines, plus the
    /// `Context: …` meter printed under it.
    Snapshot {
        lines: Vec<String>,
        context_line: Option<String>,
    },
    /// Live card shown while the model streams a tool call's raw arguments.
    ToolGenerating { name: String, bytes: usize, text: String },
    /// Raw command output (may carry SGR colour) or a plain notice.
    Notice(String),
    /// A line styled like the REPL's `print_system` (muted).
    System(String),
    /// The rule at the top of a fresh session.
    Banner(String),
    Error {
        text: String,
        hint: Option<String>,
    },
    Checkpoint {
        sha: String,
        label: String,
    },
}

impl MessageItem {
    /// An assistant message in the idle state, for callers that do not stream.
    pub fn assistant(content: impl Into<String>, thinking: impl Into<String>) -> Self {
        MessageItem::Assistant {
            content: content.into(),
            thinking: thinking.into(),
            thinking_expanded: false,
            is_streaming_thinking: false,
            is_streaming_content: false,
            thinking_duration: None,
            thinking_start: None,
            stream: StreamState::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct AppStatus {
    pub model: String,
    pub provider: String,
    pub tier: String,
    pub ctx_tokens: usize,
    pub ctx_max: usize,
    pub ctx_percent: f64,
    /// Flattened reasoning level (off/auto/low/medium/high/xhigh) reported by
    /// the bridge. Reflects the config the agent reads on every request.
    pub thinking_level: String,
    /// 'compact' | 'verbose' | 'off'.
    pub snapshot_mode: String,
    pub session_prompt: f64,
    pub session_completion: f64,
    pub session_cost: f64,
    pub session_requests: usize,
    /// The REPL toolbar's label for the reasoning setting (`AUTO+rethink`).
    pub thinking_label: String,
    /// `CHAT` or `VIBE`, as the REPL toolbar shows it.
    pub mode: String,
    /// `sum 5k tok · $0.01`, empty until something was spent.
    pub session_summary: String,
    pub vibe: bool,
    pub cwd: String,
    pub turn_counter: usize,
    /// The bridge has a conversation state to go back to ("back to here" is offered).
    pub can_rewind: bool,
    /// The agent's checklist for the work in hand (`todo_write`), shown above the prompt.
    pub todos: Vec<TodoItem>,
    pub start_time: Instant,
}

/// One step of the agent's checklist. `status` is `pending`, `in_progress` or `completed`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TodoItem {
    pub content: String,
    pub status: String,
}

impl TodoItem {
    pub fn is_done(&self) -> bool {
        self.status == "completed"
    }
    pub fn is_active(&self) -> bool {
        self.status == "in_progress"
    }
}

/// The checklist the bridge reports as JSON text (`[{"content": .., "status": ..}, ..]`).
/// Anything unreadable is an empty list: a broken payload must not break the screen.
pub fn parse_todos(raw: &str) -> Vec<TodoItem> {
    let Ok(items) = serde_json::from_str::<Vec<serde_json::Value>>(raw.trim()) else {
        return Vec::new();
    };
    items
        .into_iter()
        .filter_map(|v| {
            let content = v.get("content")?.as_str()?.trim().to_string();
            if content.is_empty() {
                return None;
            }
            let status = v.get("status").and_then(|s| s.as_str()).unwrap_or("pending").to_string();
            Some(TodoItem { content, status })
        })
        .collect()
}

impl Default for AppStatus {
    fn default() -> Self {
        Self {
            model: "Argent".to_string(),
            provider: "".to_string(),
            tier: "".to_string(),
            ctx_tokens: 0,
            ctx_max: 32768,
            ctx_percent: 0.0,
            thinking_level: "auto".to_string(),
            snapshot_mode: "compact".to_string(),
            session_prompt: 0.0,
            session_completion: 0.0,
            session_cost: 0.0,
            session_requests: 0,
            thinking_label: String::new(),
            mode: "CHAT".to_string(),
            session_summary: String::new(),
            vibe: false,
            cwd: std::env::current_dir()
                .map(|p| p.file_name().unwrap_or_default().to_string_lossy().to_string())
                .unwrap_or_else(|_| "Argent".to_string()),
            turn_counter: 0,
            can_rewind: false,
            todos: Vec::new(),
            start_time: Instant::now(),
        }
    }
}

impl AppStatus {
    pub fn session_elapsed_formatted(&self) -> String {
        let secs = self.start_time.elapsed().as_secs();
        let hours = secs / 3600;
        let mins = (secs % 3600) / 60;
        let s = secs % 60;
        if hours > 0 {
            format!("{:02}:{:02}:{:02}", hours, mins, s)
        } else {
            format!("{:02}:{:02}", mins, s)
        }
    }
}

/// Directories never walked by @-mention completion.
const PRUNE_DIRS: [&str; 8] = [
    "node_modules", "target", ".venv", "venv", "__pycache__", "dist", "build", ".git",
];

/// How many times the bridge may refuse the same queued prompt in a row before
/// the frontend gives up on it.
///
/// A refused prompt is put back at the head of the queue and the main loop
/// retries on the very next tick, so without a cap a permanently rejected turn
/// livelocks the loop and starves every other queued message.
const MAX_TURN_REJECTS: usize = 3;

/// What the frontend should do with a submitted prompt.
#[derive(Debug, PartialEq, Eq)]
pub enum Submit {
    /// Send it to the bridge as a turn.
    Dispatch,
    /// Handled entirely inside the TUI; nothing to send.
    Handled,
}

/// Truncate for single-line queue display, matching the legacy CLI's 65-char cap.
fn queue_preview(s: &str) -> String {
    const LIMIT: usize = 65;
    if s.chars().count() > LIMIT {
        let head: String = s.chars().take(LIMIT - 3).collect();
        format!("{head}\u{2026}")
    } else {
        s.to_string()
    }
}

/// Everything after the first whitespace-delimited word.
fn strip_first_word(s: &str) -> &str {
    match s.find(char::is_whitespace) {
        Some(i) => s[i..].trim_start(),
        None => "",
    }
}

/// The part of `s` whose characters START in display columns `from..to`.
pub(crate) fn slice_columns(s: &str, from: usize, to: usize) -> String {
    let mut col = 0usize;
    let mut out = String::new();
    for ch in s.chars() {
        let w = unicode_width::UnicodeWidthChar::width(ch).unwrap_or(0);
        if col >= from && col < to {
            out.push(ch);
        }
        col += w;
        if col >= to {
            break;
        }
    }
    out
}

/// `8.2s` under a minute, then `1m 24s` / `1h 12m`.
fn format_time(secs: f64) -> String {
    if secs < 60.0 {
        format!("{secs:.1}s")
    } else if secs < 3600.0 {
        format!("{}m {}s", (secs / 60.0) as i64, (secs % 60.0) as i64)
    } else {
        format!("{}h {}m", (secs / 3600.0) as i64, ((secs % 3600.0) / 60.0) as i64)
    }
}

/// `1234` -> `1.2k`, `1500000` -> `1.5M` (matches `usage_tracker._short`).
fn short_tokens(n: f64) -> String {
    let n = n.max(0.0);
    let scale = |v: f64, suffix: &str| -> String {
        let s = format!("{v:.1}");
        format!("{}{suffix}", s.trim_end_matches(".0"))
    };
    if n >= 1_000_000.0 {
        scale(n / 1_000_000.0, "M")
    } else if n >= 1000.0 {
        scale(n / 1000.0, "k")
    } else {
        format!("{}", n.round() as i64)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClickAction {
    ToggleThinking(usize), // message index
    ToggleTool(usize),     // message index
    CycleThinkingLevel,
    OpenModelPicker,
    OpenProviderPicker,
    CopyResponse(usize),   // message index
    CopyLastResponse,
    CopyLastCode,
    CopyCode(String),
    ToggleMouseCapture,
    /// The "back to here" button under the last message the person sent.
    RewindLast,
    /// The "newest" chip shown while the chat is scrolled back.
    ScrollToBottom,
}

#[derive(Debug, Clone)]
pub struct ClickableRegion {
    pub x: u16,
    pub y: u16,
    pub width: u16,
    pub height: u16,
    pub action: ClickAction,
}

/// The transcript row at the TOP of the viewport while the chat is scrolled back,
/// and the scroll offset it was measured at.
///
/// The offset counts rows up from the newest output, so anything that arrives
/// below the viewport (a tool result, the next paragraph of an answer) would slide
/// the text the user is reading upward. Pinning the top row and recomputing the
/// offset from it each frame keeps what they are reading where it is. `offset`
/// says which scroll position the anchor belongs to: once the user scrolls, the
/// two differ and the anchor is dropped in favour of where they scrolled to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScrollAnchor {
    pub item: usize,
    pub row: usize,
    pub offset: usize,
}

#[allow(dead_code)]
pub struct App {
    pub clickable_regions: std::cell::RefCell<Vec<ClickableRegion>>,
    pub mouse_capture_enabled: bool,
    pub status: AppStatus,
    pub messages: Vec<MessageItem>,
    pub input: String,
    pub cursor_pos: usize,
    pub input_history: Vec<String>,
    pub history_idx: Option<usize>,
    pub active_approval: Option<ApprovalData>,
    pub active_select: Option<SelectModalData>,
    pub is_busy: bool,
    pub is_command_in_flight: bool,
    /// A stop was requested and acknowledged; the turn is still unwinding, so
    /// `is_busy` stays true and the message queue stays paused.
    pub is_cancelling: bool,
    pub message_queue: Vec<String>,
    /// Consecutive `turn_was_rejected` answers for the prompt at the head of the
    /// queue. Reset as soon as the bridge produces any turn output.
    pub reject_streak: usize,
    pub scroll_offset: usize,
    pub max_scroll: std::cell::Cell<usize>,
    /// The scroll offset the last frame actually used (it clamps to what exists).
    pub scroll_used: std::cell::Cell<Option<usize>>,
    /// What keeps a scrolled-back chat still while output arrives (see `ScrollAnchor`).
    pub scroll_anchor: std::cell::Cell<Option<ScrollAnchor>>,
    /// Rows of empty space the user has scrolled open under the newest output.
    pub overscroll: usize,
    pub auto_scroll: bool,
    /// Set by an `exit` event from the bridge (e.g. the `/exit` command). The
    /// main loop checks it and shuts down cleanly.
    pub should_quit: bool,
    pub completions: Vec<CompletionItem>,
    pub completion_idx: usize,
    pub show_completions: bool,
    pub file_completions: Vec<String>,
    pub file_completion_idx: usize,
    pub show_file_completions: bool,
    /// Set while Tab is cycling through suggestions: the text the user had typed
    /// before the first Tab. The popup keeps listing what matches *that*, not the
    /// suggestion that is previewed in the input, so Tab can go on cycling.
    pub completion_cycle: Option<CompletionCycle>,
    pub saved_draft: Option<String>,
    /// Emacs-style kill ring. Consecutive kills with the same `unit` are
    /// concatenated, so Ctrl+K twice then Ctrl+Y restores both.
    pub kill_ring: Vec<String>,
    /// `unit` of the most recent kill, for append semantics.
    pub last_kill_unit: Option<&'static str>,
    /// Set by `yank` so a following `yank_pop` can replace the insertion.
    pub yanked: Option<String>,
    pub undo_stack: Vec<(String, usize)>,
    /// The level cycle offered by the bridge, used by `cycle_thinking_level`.
    pub thinking_levels: Vec<String>,
    /// Set when a level change could not be applied, shown as a notice.
    pub thinking_error: Option<String>,
    /// Anchor of a Shift-arrow selection in char indices.
    pub sel_anchor: Option<usize>,
    /// Mirror of the CLI's `show_tool_results` setting.
    pub show_tool_results: bool,
    /// False while the window is unfocused, so stray keystrokes are ignored.
    pub focused: bool,
    /// Name of the interactive tool currently running, if any.
    pub interactive_running: Option<String>,
    /// Incremental reverse history search (Ctrl+R) state.
    pub search_active: bool,
    pub search_query: String,
    pub search_idx: Option<usize>,
    pub search_saved: Option<String>,
    pub tick: u64,
    pub spinner_frames: [&'static str; 10],
    /// Index into `messages` of the collapsible block the user is working with:
    /// an assistant reasoning accordion or a tool card. `None` when the
    /// transcript holds nothing collapsible.
    pub focus_idx: Option<usize>,
    /// True once the user has moved the block focus themselves. While it is
    /// false the selection follows newly arrived blocks, so Tab acts on the
    /// latest reply; once pinned it stays where the user put it, so several
    /// blocks can be opened independently.
    pub focus_pinned: bool,
    /// Renders static blocks: the Python bridge in production, a small Rust
    /// approximation when none is attached (tests) or when a render fails.
    pub renderer: Box<dyn BlockRenderer>,
    /// Theme and display settings handed over by the bridge.
    pub ui: UiConfig,
    /// Rendered blocks, keyed by position and validated by a content fingerprint.
    pub view_cache: RefCell<crate::view::ViewCache>,
    /// The first render failure not yet reported to the user.
    pub render_error: RefCell<Option<String>>,
    /// When the running turn began; `None` while idle. Drives the `(0:08)` timers.
    pub turn_started: Option<Instant>,
    /// Start of the current live phase (reasoning, generating, a tool).
    pub phase_started: Instant,
    /// How fast the model is generating, timed from when its output arrived.
    pub speed: SpeedMeter,
    /// Something visible changed since the last frame was drawn.
    pub dirty: Cell<bool>,
    /// Pins the spinner to one frame instead of following the clock, so a frame
    /// can be compared against the REPL's (the tests set it).
    pub spinner_frame: Cell<Option<usize>>,
    /// A selection made with the mouse in the transcript. Both ends are places in
    /// the conversation, not screen cells, so scrolling does not disturb it.
    pub mouse_sel: Option<MouseSelection>,
    /// The mouse button is down and the selection is still being dragged.
    pub selecting: bool,
    /// Where the pointer last was during the drag, in screen cells.
    pub last_drag: Option<(u16, u16)>,
    /// The screen row the drag began on.
    pub press_row: Option<u16>,
    /// The pointer is held above (-1) or below (1) the chat: it scrolls on its own.
    pub drag_edge: Option<i8>,
    /// The last press, to tell a double or triple click from three single ones.
    pub last_click: Option<(Instant, u16, u16, u8)>,
    /// When the chat last scrolled on its own during a drag (paces the scrolling).
    pub last_edge_scroll: Option<Instant>,
    /// The last wheel notch: when, which way, and how many notches in a row.
    pub last_wheel: Option<(Instant, bool, u8)>,
    /// What the rows last drawn in the transcript are, for selecting and copying.
    pub view_text: RefCell<ViewText>,
}

/// The transcript rows as last drawn: their text and where each one comes from.
#[derive(Debug, Default, Clone)]
pub struct ViewText {
    /// Screen row of the first transcript row.
    pub top: u16,
    pub rows: Vec<String>,
    /// For each row, the message it belongs to and its index inside that message.
    pub anchors: Vec<(usize, usize)>,
    /// For each row, the columns that are text (the rest is rail, margin, padding).
    pub cols: Vec<crate::view::Cols>,
    /// Columns the transcript was laid out at, and how many rows the chat area has.
    pub width: u16,
    pub view_h: u16,
}

impl App {
    pub fn new() -> Self {
        Self {
            renderer: Box::new(FallbackRenderer),
            ui: UiConfig::default(),
            view_cache: RefCell::new(crate::view::ViewCache::default()),
            render_error: RefCell::new(None),
            turn_started: None,
            phase_started: Instant::now(),
            speed: SpeedMeter::new(),
            dirty: Cell::new(true),
            spinner_frame: Cell::new(None),
            mouse_sel: None,
            selecting: false,
            last_drag: None,
            press_row: None,
            drag_edge: None,
            last_click: None,
            last_edge_scroll: None,
            last_wheel: None,
            view_text: RefCell::new(ViewText::default()),
            status: AppStatus::default(),
            messages: Vec::new(),
            input: String::new(),
            cursor_pos: 0,
            input_history: Vec::new(),
            history_idx: None,
            saved_draft: None,
            kill_ring: Vec::new(),
            last_kill_unit: None,
            yanked: None,
            undo_stack: Vec::new(),
            search_active: false,
            search_query: String::new(),
            search_idx: None,
            search_saved: None,
            thinking_levels: vec![
                "off".into(),
                "auto".into(),
                "low".into(),
                "medium".into(),
                "high".into(),
                "xhigh".into(),
            ],
            thinking_error: None,
            sel_anchor: None,
            show_tool_results: false,
            focused: true,
            interactive_running: None,
            active_approval: None,
            active_select: None,
            is_busy: false,
            is_command_in_flight: false,
            is_cancelling: false,
            message_queue: Vec::new(),
            reject_streak: 0,
            scroll_offset: 0,
            max_scroll: std::cell::Cell::new(0),
            scroll_used: std::cell::Cell::new(None),
            scroll_anchor: std::cell::Cell::new(None),
            overscroll: 0,
            auto_scroll: true,
            should_quit: false,
            completions: Vec::new(),
            completion_idx: 0,
            show_completions: false,
            file_completions: Vec::new(),
            file_completion_idx: 0,
            show_file_completions: false,
            completion_cycle: None,
            tick: 0,
            focus_idx: None,
            focus_pinned: false,
            spinner_frames: ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"],
            clickable_regions: std::cell::RefCell::new(Vec::new()),
            mouse_capture_enabled: false,
        }
    }

    pub fn register_clickable(&self, x: u16, y: u16, width: u16, height: u16, action: ClickAction) {
        if width > 0 && height > 0 {
            self.clickable_regions.borrow_mut().push(ClickableRegion {
                x,
                y,
                width,
                height,
                action,
            });
        }
    }

    pub fn clear_clickables(&self) {
        self.clickable_regions.borrow_mut().clear();
    }

    pub fn get_clicked_action(&self, col: u16, row: u16) -> Option<ClickAction> {
        let regions = self.clickable_regions.borrow();
        for r in regions.iter().rev() {
            if col >= r.x
                && col < r.x.saturating_add(r.width)
                && row >= r.y
                && row < r.y.saturating_add(r.height)
            {
                return Some(r.action.clone());
            }
        }
        None
    }

    pub fn toggle_block_by_idx(&mut self, idx: usize) -> bool {
        if idx >= self.messages.len() {
            return false;
        }
        match &mut self.messages[idx] {
            MessageItem::Assistant { thinking, thinking_expanded, .. } if !thinking.is_empty() => {
                *thinking_expanded = !*thinking_expanded;
                self.focus_idx = Some(idx);
                self.focus_pinned = true;
                true
            }
            MessageItem::Tool { expanded, .. } => {
                *expanded = !*expanded;
                self.focus_idx = Some(idx);
                self.focus_pinned = true;
                true
            }
            MessageItem::Rethink { expanded, text, .. } if !text.is_empty() => {
                *expanded = !*expanded;
                self.focus_idx = Some(idx);
                self.focus_pinned = true;
                true
            }
            _ => false,
        }
    }

    pub fn current_spinner(&self) -> &'static str {
        self.spinner_frames[(self.tick as usize / 2) % self.spinner_frames.len()]
    }

    pub fn update_status(&mut self, state: HashMap<String, String>) {
        // A different model counts tokens differently: what the speed meter learned
        // from the old one's token counts no longer applies.
        let changed = |new: Option<&String>, old: &str| new.is_some_and(|n| n != old && !old.is_empty());
        if changed(state.get("model"), &self.status.model) || changed(state.get("provider"), &self.status.provider) {
            self.speed.forget_calibration();
        }
        if let Some(m) = state.get("model") { self.status.model = m.clone(); }
        if let Some(p) = state.get("provider") { self.status.provider = p.clone(); }
        if let Some(t) = state.get("tier") { self.status.tier = t.clone(); }
        if let Some(c) = state.get("cwd") { self.status.cwd = c.clone(); }
        if let Some(v) = state.get("vibe") {
            let v_lower = v.trim().to_lowercase();
            self.status.vibe = v_lower == "true" || v_lower == "1";
        }
        if let Some(raw) = state.get("todos") {
            let todos = parse_todos(raw);
            if todos != self.status.todos {
                self.status.todos = todos;
            }
        }
        if let Some(v) = state.get("can_rewind") {
            let v = v.trim().to_lowercase();
            self.status.can_rewind = v == "true" || v == "1";
        }
        if let Some(tc) = state.get("turn_counter") {
            if let Ok(val) = tc.parse::<usize>() { self.status.turn_counter = val; }
        }
        if let Some(tok) = state.get("ctx_tokens") {
            if let Ok(val) = tok.parse::<usize>() { self.status.ctx_tokens = val; }
        }
        if let Some(m) = state.get("ctx_max") {
            if let Ok(val) = m.parse::<usize>() { self.status.ctx_max = val; }
        }
        if let Some(pct) = state.get("ctx_percent") {
            if let Ok(val) = pct.parse::<f64>() { self.status.ctx_percent = val; }
        }
        if let Some(lvl) = state.get("thinking_level") {
            let lvl = lvl.trim();
            if !lvl.is_empty() { self.status.thinking_level = lvl.to_string(); }
        }
        if let Some(m) = state.get("snapshot_mode") {
            let m = m.trim();
            if !m.is_empty() { self.status.snapshot_mode = m.to_string(); }
        }
        if let Some(l) = state.get("thinking_label") {
            self.status.thinking_label = l.trim().to_string();
        }
        if let Some(m) = state.get("mode") {
            let m = m.trim();
            if !m.is_empty() {
                self.status.mode = m.to_string();
            }
        }
        if let Some(s) = state.get("session_summary") {
            self.status.session_summary = s.trim().to_string();
        }
        let usage_keys = ["session_prompt", "session_completion", "session_cost", "session_requests"];
        for (i, key) in usage_keys.iter().enumerate() {
            if let Some(v) = state.get(*key) {
                if let Ok(f) = v.parse::<f64>() {
                    match i {
                        0 => self.status.session_prompt = f,
                        1 => self.status.session_completion = f,
                        2 => self.status.session_cost = f,
                        _ => self.status.session_requests = f as usize,
                    }
                }
            }
        }
    }

    /// Record that a turn is in flight; the first sign of life starts its timer.
    fn begin_turn_if_idle(&mut self) {
        if self.turn_started.is_none() {
            let now = Instant::now();
            self.turn_started = Some(now);
            self.phase_started = now;
        }
    }

    /// `messages.insert` that keeps the scroll anchor on the same message.
    fn insert_message(&mut self, at: usize, item: MessageItem) {
        if let Some(mut a) = self.scroll_anchor.get() {
            if a.item >= at {
                a.item += 1;
                self.scroll_anchor.set(Some(a));
            }
        }
        self.messages.insert(at, item);
    }

    /// `messages.retain(|m| !gone(m))` that keeps the scroll anchor on the same
    /// message, or drops it when that message is the one removed.
    fn drop_messages(&mut self, gone: impl Fn(&MessageItem) -> bool) {
        if let Some(mut a) = self.scroll_anchor.get() {
            let mut before = 0;
            let mut anchor_gone = false;
            for (i, m) in self.messages.iter().enumerate() {
                if gone(m) {
                    if i < a.item {
                        before += 1;
                    } else if i == a.item {
                        anchor_gone = true;
                    }
                }
            }
            if anchor_gone {
                self.scroll_anchor.set(None);
            } else if before > 0 {
                a.item -= before;
                self.scroll_anchor.set(Some(a));
            }
        }
        self.messages.retain(|m| !gone(m));
    }

    /// The message the "back to here" button sits under: the newest one the person sent
    /// to the agent. A `/command` or `!shell` line is not part of the conversation the
    /// agent can be returned to. None while a turn runs, a question is open, or the
    /// bridge reports nothing to go back to.
    pub fn rewind_target(&self) -> Option<usize> {
        if !self.status.can_rewind
            || self.is_busy
            || self.is_command_in_flight
            || self.active_approval.is_some()
            || self.active_select.is_some()
        {
            return None;
        }
        self.messages.iter().rposition(|m| match m {
            MessageItem::User(t) => {
                let t = t.trim_start();
                !t.starts_with('/') && !t.starts_with('!')
            }
            _ => false,
        })
    }

    /// The bridge went back to before the person's last message: forget that message
    /// and everything after it, and hand its text back in the prompt to edit or resend.
    pub fn apply_rewind(&mut self, text: &str, detail: &str) {
        let at = self
            .messages
            .iter()
            .rposition(|m| matches!(m, MessageItem::User(t) if t == text))
            .or_else(|| {
                self.messages.iter().rposition(|m| {
                    matches!(m, MessageItem::User(t) if !t.trim_start().starts_with('/') && !t.trim_start().starts_with('!'))
                })
            });
        if let Some(at) = at {
            self.messages.truncate(at);
        }
        self.scroll_anchor.set(None);
        self.view_cache.borrow_mut().clear();
        self.mouse_sel = None;
        self.selecting = false;
        self.clear_selection();
        self.status.can_rewind = false;

        self.record_undo();
        let draft = std::mem::take(&mut self.input);
        self.input = if draft.trim().is_empty() { text.to_string() } else { format!("{text}\n{draft}") };
        self.cursor_pos = text.chars().count();
        self.history_idx = None;
        let note = if detail.is_empty() {
            "↩ Вернулись к состоянию до сообщения.".to_string()
        } else {
            format!("↩ Вернулись к состоянию до сообщения. {detail}")
        };
        self.messages.push(MessageItem::Notice(note));
        self.auto_scroll = true;
        self.dirty.set(true);
    }

    /// Put a saved conversation back on the screen.
    ///
    /// The blocks are added as finished items. Nothing here touches the turn state (`is_busy`,
    /// the turn clock, the queue): a restored dialogue is history, not a turn in progress.
    pub fn apply_history(&mut self, items: Vec<HistoryItem>, replace: bool) {
        self.finish_streams();
        if replace {
            self.messages.clear();
            self.scroll_anchor.set(None);
            self.view_cache.borrow_mut().clear();
            self.mouse_sel = None;
            self.selecting = false;
            self.clear_selection();
            self.focus_idx = None;
            self.focus_pinned = false;
        }
        for item in items {
            self.messages.push(match item {
                HistoryItem::User(text) => MessageItem::User(text),
                HistoryItem::Assistant { text, thinking } => MessageItem::assistant(text, thinking),
                HistoryItem::Tool { name, args, result } => MessageItem::Tool {
                    name,
                    args,
                    result: Some(result),
                    duration_secs: None,
                    expanded: false,
                    is_running: false,
                },
                HistoryItem::Note(text) => MessageItem::System(text),
            });
        }
        self.auto_scroll = true;
        self.scroll_offset = 0;
        self.dirty.set(true);
    }

    fn pop_message(&mut self) {
        if let Some(a) = self.scroll_anchor.get() {
            if a.item + 1 >= self.messages.len() {
                self.scroll_anchor.set(None);
            }
        }
        self.messages.pop();
    }

    /// Append a line-like item (diff, notice, error, checkpoint).
    ///
    /// While a tool runs, the REPL shows only a transient spinner for it and
    /// prints its result when it FINISHES; anything printed in between lands above
    /// that result. The running tool's placeholder therefore stays last, so the
    /// final order matches.
    fn push_static(&mut self, item: MessageItem) {
        if matches!(self.messages.last(), Some(MessageItem::Tool { is_running: true, .. })) {
            let at = self.messages.len() - 1;
            self.insert_message(at, item);
        } else {
            self.messages.push(item);
        }
    }

    /// Close any reasoning/answer stream that is still open: commit the rest of
    /// the answer and freeze the reasoning timers. Idempotent.
    fn finish_streams(&mut self) {
        for item in self.messages.iter_mut().rev() {
            match item {
                MessageItem::Assistant {
                    content,
                    is_streaming_thinking,
                    is_streaming_content,
                    thinking_start,
                    thinking_duration,
                    stream,
                    ..
                } => {
                    if *is_streaming_thinking {
                        *is_streaming_thinking = false;
                        if let Some(start) = thinking_start {
                            *thinking_duration = Some(start.elapsed().as_secs_f64());
                        }
                    }
                    if *is_streaming_content {
                        stream.finalize(content);
                    }
                    *is_streaming_content = false;
                    break;
                }
                MessageItem::Rethink { is_streaming, start, duration, .. } if *is_streaming => {
                    *is_streaming = false;
                    if let Some(s) = start {
                        *duration = Some(s.elapsed().as_secs_f64());
                    }
                    // A rethink sits between the answer and what follows; keep
                    // looking for the assistant block it belongs to.
                }
                MessageItem::Tool { .. } | MessageItem::User(_) => break,
                _ => {}
            }
        }
    }

    pub fn handle_event(&mut self, event: ArgentEvent) {
        self.handle_event_at(event, Instant::now());
    }

    /// `handle_event` for an event that reached the sink at `at`: the generation
    /// speed is timed by that, not by when the loop got round to the event.
    pub fn handle_event_at(&mut self, event: ArgentEvent, at: Instant) {
        match &event {
            ArgentEvent::Content(text) | ArgentEvent::Thinking(text) | ArgentEvent::Rethink(text) => {
                self.speed.feed(text, at);
            }
            ArgentEvent::ToolGenerating { delta, .. } => self.speed.feed(delta, at),
            ArgentEvent::Usage { completion: Some(n), .. } => self.speed.settle(*n),
            // A request is over before its tool runs, and before the turn ends: a
            // count that never came must not be matched with the next request's text.
            ArgentEvent::ToolStart { .. } | ArgentEvent::Error { .. } | ArgentEvent::Done => {
                self.speed.end_request();
            }
            _ => {}
        }
        let blocks_before = self.collapsible_blocks().len();
        self.dirty.set(true);
        // New output follows the newest line: the empty space under it closes.
        if !matches!(event, ArgentEvent::UiStyle(_)) {
            self.overscroll = 0;
        }
        match event {
            ArgentEvent::Content(delta) => {
                self.begin_turn_if_idle();
                self.is_busy = true;
                self.is_command_in_flight = false;
                self.reject_streak = 0;
                let mut restart_phase = false;
                if let Some(MessageItem::Assistant {
                    content,
                    is_streaming_content,
                    is_streaming_thinking,
                    thinking_start,
                    thinking_duration,
                    stream,
                    ..
                }) = self.messages.last_mut()
                {
                    if *is_streaming_thinking {
                        *is_streaming_thinking = false;
                        if let Some(start) = thinking_start {
                            *thinking_duration = Some(start.elapsed().as_secs_f64());
                        }
                    }
                    if !*is_streaming_content {
                        restart_phase = true;
                    }
                    *is_streaming_content = true;
                    content.push_str(&delta);
                    stream.advance(content);
                } else {
                    restart_phase = true;
                    let mut item = MessageItem::assistant(delta, "");
                    if let MessageItem::Assistant { content, is_streaming_content, stream, .. } = &mut item {
                        *is_streaming_content = true;
                        stream.advance(content);
                    }
                    self.messages.push(item);
                }
                if restart_phase {
                    self.phase_started = Instant::now();
                }
            }
            ArgentEvent::ContentReplace(full_text) => {
                self.is_command_in_flight = false;
                if let Some(MessageItem::Assistant { content, stream, .. }) = self.messages.last_mut() {
                    *content = full_text;
                    stream.replace(content);
                } else {
                    let mut item = MessageItem::assistant(full_text, "");
                    if let MessageItem::Assistant { content, is_streaming_content, stream, .. } = &mut item {
                        *is_streaming_content = true;
                        stream.replace(content);
                    }
                    self.messages.push(item);
                }
            }
            ArgentEvent::Thinking(delta) => {
                self.begin_turn_if_idle();
                self.is_busy = true;
                self.is_command_in_flight = false;
                self.reject_streak = 0;
                let expand_by_default = self.ui.thinking_mode == "full";
                // Reasoning that resumes after the answer already started is a
                // NEW reasoning block, exactly as the REPL prints it.
                let reuse = matches!(
                    self.messages.last(),
                    Some(MessageItem::Assistant { content, .. }) if content.is_empty()
                );
                let mut restart_phase = true;
                if reuse {
                    if let Some(MessageItem::Assistant {
                        thinking,
                        is_streaming_thinking,
                        thinking_start,
                        ..
                    }) = self.messages.last_mut()
                    {
                        restart_phase = !*is_streaming_thinking;
                        *is_streaming_thinking = true;
                        if thinking_start.is_none() {
                            *thinking_start = Some(Instant::now());
                        }
                        thinking.push_str(&delta);
                    }
                } else {
                    self.finish_streams();
                    self.messages.push(MessageItem::Assistant {
                        content: String::new(),
                        thinking: delta,
                        thinking_expanded: expand_by_default,
                        is_streaming_thinking: true,
                        is_streaming_content: false,
                        thinking_duration: None,
                        thinking_start: Some(Instant::now()),
                        stream: StreamState::default(),
                    });
                }
                if restart_phase {
                    self.phase_started = Instant::now();
                }
            }
            ArgentEvent::Rethink(delta) => {
                self.begin_turn_if_idle();
                self.is_busy = true;
                self.is_command_in_flight = false;
                self.reject_streak = 0;
                if let Some(MessageItem::Rethink { text, is_streaming: true, .. }) = self.messages.last_mut() {
                    text.push_str(&delta);
                } else {
                    self.finish_streams();
                    self.messages.push(MessageItem::Rethink {
                        text: delta,
                        expanded: self.ui.thinking_mode == "full",
                        is_streaming: true,
                        duration: None,
                        start: Some(Instant::now()),
                    });
                    self.phase_started = Instant::now();
                }
            }
            ArgentEvent::ToolStart { name, args } => {
                self.begin_turn_if_idle();
                self.is_busy = true;
                self.is_command_in_flight = false;
                self.reject_streak = 0;
                self.drop_messages(|item| matches!(item, MessageItem::ToolGenerating { name: n, .. } if n == &name));
                self.interactive_running = if App::is_interactive_tool(&name) {
                    Some(name.clone())
                } else {
                    None
                };
                self.finish_streams();
                self.phase_started = Instant::now();
                self.messages.push(MessageItem::Tool {
                    name,
                    args,
                    result: None,
                    duration_secs: None,
                    expanded: false,
                    is_running: true,
                });
            }
            ArgentEvent::ToolEnd { name, result, duration_secs } => {
                if self.interactive_running.as_deref() == Some(name.as_str()) {
                    self.interactive_running = None;
                }
                for item in self.messages.iter_mut().rev() {
                    if let MessageItem::Tool { name: t_name, result: t_res, duration_secs: t_dur, is_running, .. } = item {
                        if *is_running && t_name == &name {
                            *is_running = false;
                            *t_res = Some(result.clone());
                            *t_dur = duration_secs;
                            break;
                        }
                    }
                }
                self.phase_started = Instant::now();
            }
            ArgentEvent::ApprovalRequest { id, action, destructive, grant_key, prompt } => {
                self.active_approval = Some(ApprovalData {
                    id,
                    action,
                    destructive,
                    grant_key,
                    prompt,
                    selected: 0,
                });
            }
            ArgentEvent::Checkpoint { sha, label } => {
                self.push_static(MessageItem::Checkpoint { sha, label });
            }
            ArgentEvent::Rewound { text, detail } => self.apply_rewind(&text, &detail),
            ArgentEvent::History { items, replace } => self.apply_history(items, replace),
            ArgentEvent::Diff { file, diff } => {
                self.push_static(MessageItem::Diff { file, diff });
            }
            ArgentEvent::ToolGenerating { name, bytes, delta } => {
                // Update the existing card in place so the counter does not emit a
                // new row for every streamed token.
                let mut updated = false;
                for item in self.messages.iter_mut().rev() {
                    if let MessageItem::ToolGenerating { name: n, bytes: b, text } = item {
                        if *n == name {
                            *b = (*b).max(bytes);
                            text.push_str(&delta);
                            updated = true;
                            break;
                        }
                    }
                }
                if !updated {
                    self.begin_turn_if_idle();
                    self.finish_streams();
                    self.phase_started = Instant::now();
                    self.messages.push(MessageItem::ToolGenerating { name, bytes, text: delta });
                }
            }
            ArgentEvent::Notice(text) => {
                self.push_static(MessageItem::Notice(text));
            }
            ArgentEvent::System(text) => {
                self.push_static(MessageItem::System(text));
            }
            ArgentEvent::Banner(version) => {
                self.messages.push(MessageItem::Banner(version));
            }
            ArgentEvent::Error { text, hint } => {
                self.drop_messages(|item| matches!(item, MessageItem::ToolGenerating { .. }));
                // The hint travels inside the item: the REPL's own error block
                // prints it under the message, so a separate row would repeat it.
                self.push_static(MessageItem::Error { text, hint });
            }
            ArgentEvent::Usage { tokens, max, percent, .. } => {
                // A provider's count of generated tokens says nothing about how full
                // the context is: only a report that carries those numbers moves the
                // gauge (otherwise every finished request blanked it until the next
                // state refresh).
                if tokens > 0 || max > 0 {
                    self.status.ctx_tokens = tokens;
                    if max > 0 {
                        self.status.ctx_max = max;
                        self.status.ctx_percent = percent;
                    }
                }
            }
            ArgentEvent::SelectRequest {
                id,
                title,
                options,
                default_value,
                multi,
                initial_checked,
            } => {
                self.active_select = Some(SelectModalData::new_full(
                    id,
                    title,
                    options,
                    default_value,
                    multi,
                    initial_checked,
                ));
            }
            ArgentEvent::TurnSnapshot {
                total_s,
                tool_calls,
                prompt_tokens,
                completion_tokens,
                cached_tokens,
                cost,
                ctx_tokens,
                ctx_max,
                ctx_percent,
                lines,
                context_line,
                ..
            } => {
                if self.status.snapshot_mode == "off" && lines.is_empty() {
                    // Nothing to show and no bridge text to defer to.
                } else if !lines.is_empty() {
                    self.messages.push(MessageItem::Snapshot { lines, context_line });
                } else {
                    // No bridge-built text (an older bridge): format the compact
                    // line the REPL would, from the raw numbers.
                    let mut parts: Vec<String> = vec![format_time(total_s)];
                    if let (Some(p), Some(c)) = (prompt_tokens, completion_tokens) {
                        let mut tok = format!("{}\u{2192}{} tok", short_tokens(p), short_tokens(c));
                        if let Some(cached) = cached_tokens {
                            if cached > 0.0 && p > 0.0 {
                                tok.push_str(&format!(
                                    " \u{b7} cached {} ({}%)",
                                    short_tokens(cached),
                                    (cached / p * 100.0) as i64
                                ));
                            }
                        }
                        parts.push(tok);
                    }
                    if tool_calls > 0 {
                        parts.push(format!("{tool_calls} tools"));
                    }
                    if let (Some(t), Some(m), Some(p)) = (ctx_tokens, ctx_max, ctx_percent) {
                        if m > 0.0 {
                            parts.push(format!(
                                "ctx {}/{} ({:.0}%)",
                                short_tokens(t),
                                short_tokens(m),
                                p
                            ));
                        }
                    }
                    if let Some(c) = cost {
                        if c > 0.0 {
                            parts.push(if c < 0.1 { format!("${c:.4}") } else { format!("${c:.2}") });
                        }
                    }
                    let mut out = vec![format!("  [dim]\u{2504} {}[/dim]", parts.join(" \u{b7} "))];
                    if let Some(p) = ctx_percent {
                        if p > 85.0 {
                            out.push(format!(
                                "  [bold yellow]\u{25b2} Контекст заполнен на {p:.0}% \u{2014} рекомендуется /clear или /compact[/bold yellow]"
                            ));
                        }
                    }
                    self.messages.push(MessageItem::Snapshot { lines: out, context_line });
                }
            }
            ArgentEvent::Exit => {
                self.should_quit = true;
            }
            ArgentEvent::UiStyle(name) => {
                if let Some(style) = crate::theme::UiStyle::parse(&name) {
                    // Blocks are cached per width; their look depends on the style.
                    if self.ui.set_style(style) {
                        self.view_cache.borrow_mut().clear();
                    }
                    self.dirty.set(true);
                }
            }
            ArgentEvent::Done => {
                self.is_busy = false;
                self.is_command_in_flight = false;
                self.is_cancelling = false;
                self.turn_started = None;
                self.drop_messages(|item| matches!(item, MessageItem::ToolGenerating { .. }));
                self.finish_streams();
                for item in self.messages.iter_mut().rev() {
                    if let MessageItem::Tool { is_running, result, .. } = item {
                        if *is_running {
                            *is_running = false;
                            if result.is_none() {
                                *result = Some("завершено".to_string());
                            }
                        }
                    }
                }
            }
            ArgentEvent::Cancelling => {
                // Stay busy: the turn is still unwinding. Only the label changes.
                // A `Cancelling` that arrives after `Done` has no turn left to
                // unwind, so it must NOT re-busy the UI — the queue would stall
                // forever with no worker left to send the clearing `Done`.
                self.is_cancelling = self.is_busy;
            }
        }
        if self.auto_scroll {
            self.scroll_offset = 0;
        }
        // A newly arrived collapsible block takes the selection while the user
        // has not chosen one themselves, so Tab always hits the latest reply.
        if !self.focus_pinned && self.collapsible_blocks().len() > blocks_before {
            self.focus_idx = None;
        }
    }

    /// Put a prompt back at the FRONT of the queue after the bridge refused to
    /// start it, and drop the `User` bubble we optimistically added for it.
    pub fn requeue_rejected_message(&mut self, prompt: String) {
        self.reject_streak = self.reject_streak.saturating_add(1);
        if let Some(MessageItem::User(u)) = self.messages.last() {
            if *u == prompt {
                self.pop_message();
            }
        }
        self.is_busy = false;
        self.is_command_in_flight = false;
        self.turn_started = None;
        if self.reject_streak >= MAX_TURN_REJECTS {
            // Dropping the prompt instead of requeueing it is what stops the
            // retry loop; the user is told why rather than left with a queue
            // that silently stops moving.
            self.reject_streak = 0;
            self.message_queue.retain(|m| *m != prompt);
            self.messages.push(MessageItem::Error {
                text: format!("Не удалось начать запрос: {prompt}"),
                hint: Some(format!(
                    "Мост отклонил его {MAX_TURN_REJECTS} раза подряд. \
                     Проверьте провайдера, лимиты и авторизацию."
                )),
            });
            return;
        }
        self.message_queue.insert(0, prompt);
    }

    /// Take the next queued prompt, or `None` if a turn is still in flight.
    ///
    /// The busy check lives here rather than only at the call site so that a
    /// cancelled-but-unwinding turn (which stays `is_busy`) can never have the
    /// next message started on top of it.
    pub fn pop_next_queued_message(&mut self) -> Option<String> {
        if self.is_busy {
            return None;
        }
        if !self.message_queue.is_empty() {
            let next_text = self.message_queue.remove(0);
            self.messages.push(MessageItem::User(next_text.clone()));
            self.is_command_in_flight = next_text.starts_with('/') || next_text.starts_with('!');
            self.is_busy = true;
            self.begin_turn_if_idle();
            Some(next_text)
        } else {
            None
        }
    }

    pub fn toggle_last_reasoning(&mut self) {
        for item in self.messages.iter_mut().rev() {
            if let MessageItem::Assistant { thinking_expanded, thinking, .. } = item {
                if !thinking.is_empty() {
                    *thinking_expanded = !*thinking_expanded;
                    break;
                }
            }
        }
    }

    pub fn toggle_last_tool(&mut self) {
        for item in self.messages.iter_mut().rev() {
            if let MessageItem::Tool { expanded, .. } = item {
                *expanded = !*expanded;
                break;
            }
        }
    }

    // ── collapsible-block focus (Ctrl+N / Ctrl+P, then Tab) ─────────────────

    /// Indices into `messages` of every collapsible block, oldest first: an
    /// assistant reasoning accordion or a tool card.
    pub fn collapsible_blocks(&self) -> Vec<usize> {
        self.messages
            .iter()
            .enumerate()
            .filter(|(_, m)| match m {
                MessageItem::Assistant { thinking, .. } => !thinking.is_empty(),
                MessageItem::Rethink { text, .. } => !text.is_empty(),
                MessageItem::Tool { .. } => true,
                _ => false,
            })
            .map(|(i, _)| i)
            .collect()
    }

    /// True when `idx` is the block the user is currently working with.
    pub fn is_focused_block(&self, idx: usize) -> bool {
        self.focus_idx == Some(idx)
    }

    /// Re-anchor the selection, defaulting to the newest block.
    fn sync_focus(&mut self) {
        let blocks = self.collapsible_blocks();
        if blocks.is_empty() {
            self.focus_idx = None;
            return;
        }
        self.focus_idx = match self.focus_idx {
            Some(i) if blocks.contains(&i) => Some(i),
            _ => blocks.last().copied(),
        };
    }

    fn move_block_focus(&mut self, delta: isize) -> bool {
        let blocks = self.collapsible_blocks();
        if blocks.is_empty() {
            self.focus_idx = None;
            return false;
        }
        let cur = self.focus_idx.and_then(|i| blocks.iter().position(|&b| b == i));
        let len = blocks.len() as isize;
        let next = match cur {
            // No selection yet: enter from whichever end the user came from.
            None => {
                if delta >= 0 {
                    blocks.len() - 1
                } else {
                    0
                }
            }
            Some(p) => (((p as isize) + delta).rem_euclid(len)) as usize,
        };
        self.focus_idx = Some(blocks[next]);
        self.focus_pinned = true;
        true
    }

    pub fn focus_next_block(&mut self) -> bool {
        self.move_block_focus(1)
    }

    pub fn focus_prev_block(&mut self) -> bool {
        self.move_block_focus(-1)
    }

    /// Expand or collapse the focused block. Returns false when there is
    /// nothing collapsible to act on.
    pub fn toggle_focused_block(&mut self) -> bool {
        self.sync_focus();
        let Some(i) = self.focus_idx else { return false };
        match &mut self.messages[i] {
            MessageItem::Assistant {
                thinking,
                thinking_expanded,
                ..
            } if !thinking.is_empty() => {
                *thinking_expanded = !*thinking_expanded;
                true
            }
            MessageItem::Tool { expanded, .. } => {
                *expanded = !*expanded;
                true
            }
            MessageItem::Rethink { expanded, text, .. } if !text.is_empty() => {
                *expanded = !*expanded;
                true
            }
            _ => false,
        }
    }

    /// Expand or collapse every reasoning block at once.
    pub fn toggle_all_reasoning(&mut self, expand: bool) -> usize {
        let mut n = 0usize;
        for m in self.messages.iter_mut() {
            if let MessageItem::Assistant {
                thinking,
                thinking_expanded,
                ..
            } = m
            {
                if !thinking.is_empty() {
                    *thinking_expanded = expand;
                    n += 1;
                }
            }
        }
        n
    }

    /// Handle the prompts that only the TUI can answer.
    ///
    /// Everything else -- including `/exit`, `/clear` and friends -- is dispatched
    /// to the bridge so the real command handler runs. Intercepting them here
    /// used to skip `/exit`'s chat export and hide the agent-side history reset
    /// that `/clear` performs.
    ///
    /// `/queue` must live here because the queue is Rust state: the Python
    /// `input_queue` singleton the old CLI reads is never written to by this
    /// frontend, so the legacy `/queue` handler would always report an empty
    /// queue.
    pub fn handle_local_command(&mut self, prompt: &str) -> Submit {
        let trimmed = prompt.trim();
        let mut parts = trimmed.splitn(2, char::is_whitespace);
        let cmd = parts.next().unwrap_or("");
        let rest = parts.next().unwrap_or("").trim();

        if !matches!(cmd, "/queue" | "/qmsg") {
            return Submit::Dispatch;
        }

        let sub = rest.split_whitespace().next().unwrap_or("").to_lowercase();
        let arg = strip_first_word(rest);
        let len = self.message_queue.len();

        match sub.as_str() {
            "" | "list" | "ls" | "show" => {
                if len == 0 {
                    self.messages.push(MessageItem::Notice("\u{041e}\u{0447}\u{0435}\u{0440}\u{0435}\u{0434}\u{044c} \u{0441}\u{043e}\u{043e}\u{0431}\u{0449}\u{0435}\u{043d}\u{0438}\u{0439} \u{043f}\u{0443}\u{0441}\u{0442}\u{0430}".into()));
                } else {
                    let items: Vec<String> = self
                        .message_queue
                        .iter()
                        .enumerate()
                        .map(|(i, m)| format!("  {}. {}", i + 1, queue_preview(m)))
                        .collect();
                    self.messages.push(MessageItem::Notice(format!(
                        "\u{041e}\u{0447}\u{0435}\u{0440}\u{0435}\u{0434}\u{044c} \u{0441}\u{043e}\u{043e}\u{0431}\u{0449}\u{0435}\u{043d}\u{0438}\u{0439} ({len}):\n{}",
                        items.join("\n")
                    )));
                }
            }
            "clear" | "clean" | "reset" => {
                self.message_queue.clear();
                self.messages.push(MessageItem::Notice(format!(
                    "\u{041e}\u{0447}\u{0435}\u{0440}\u{0435}\u{0434}\u{044c} \u{043e}\u{0447}\u{0438}\u{0449}\u{0435}\u{043d}\u{0430} ({len})"
                )));
            }
            "rm" | "remove" | "del" | "pop" => {
                match arg.trim().parse::<usize>() {
                    Ok(v) if v >= 1 && v <= len => {
                        let removed = self.message_queue.remove(v - 1);
                        self.messages.push(MessageItem::Notice(format!(
                            "\u{0423}\u{0434}\u{0430}\u{043b}\u{0435}\u{043d}\u{043e} \u{0438}\u{0437} \u{043e}\u{0447}\u{0435}\u{0440}\u{0435}\u{0434}\u{0438}: {}",
                            queue_preview(&removed)
                        )));
                    }
                    _ => self.messages.push(MessageItem::Error { text: format!(
                        "\u{0423}\u{043a}\u{0430}\u{0436}\u{0438}\u{0442}\u{0435} \u{043d}\u{043e}\u{043c}\u{0435}\u{0440} \u{043e}\u{0442} 1 \u{0434}\u{043e} {len}: /queue rm <\u{2116}>"
                    ), hint: None }),
                }
            }
            "edit" | "change" => {
                let mut it = arg.trim().splitn(2, char::is_whitespace);
                let idx = it.next().unwrap_or("").parse::<usize>();
                let new_text = it.next().unwrap_or("").trim().to_string();
                match idx {
                    Ok(v) if v >= 1 && v <= len && !new_text.is_empty() => {
                        self.message_queue[v - 1] = new_text;
                        self.messages.push(MessageItem::Notice(format!(
                            "\u{041e}\u{0431}\u{043d}\u{043e}\u{0432}\u{043b}\u{0435}\u{043d}\u{043e} \u{0441}\u{043e}\u{043e}\u{0431}\u{0449}\u{0435}\u{043d}\u{0438}\u{0435} {v} \u{0432} \u{043e}\u{0447}\u{0435}\u{0440}\u{0435}\u{0434}\u{0438}"
                        )));
                    }
                    _ => self.messages.push(MessageItem::Error { text: 
                        "\u{0424}\u{043e}\u{0440}\u{043c}\u{0430}\u{0442}: /queue edit <\u{2116}> <\u{043d}\u{043e}\u{0432}\u{044b}\u{0439} \u{0442}\u{0435}\u{043a}\u{0441}\u{0442}>".into(), hint: None }),
                }
            }
            "add" | "push" => {
                let text = arg.trim();
                if text.is_empty() {
                    self.messages.push(MessageItem::Error { text: 
                        "\u{0424}\u{043e}\u{0440}\u{043c}\u{0430}\u{0442}: /queue add <\u{0442}\u{0435}\u{043a}\u{0441}\u{0442}>".into(), hint: None });
                } else {
                    self.message_queue.push(text.to_string());
                    self.messages.push(MessageItem::Notice(format!(
                        "Добавлено в очередь (всего: {})",
                        self.message_queue.len()
                    )));
                }
            }
            _ => self.messages.push(MessageItem::Error { text: 
                "\u{041d}\u{0435}\u{0438}\u{0437}\u{0432}\u{0435}\u{0441}\u{0442}\u{043d}\u{0430}\u{044f} \u{043f}\u{043e}\u{0434}\u{043a}\u{043e}\u{043c}\u{0430}\u{043d}\u{0434}\u{0430} \u{043e}\u{0447}\u{0435}\u{0440}\u{0435}\u{0434}\u{0438}. \u{0414}\u{043e}\u{0441}\u{0442}\u{0443}\u{043f}\u{043d}\u{043e}: list | rm <\u{2116}> | edit <\u{2116}> <\u{0442}\u{0435}\u{043a}\u{0441}\u{0442}> | add <\u{0442}\u{0435}\u{043a}\u{0441}\u{0442}> | clear".into(), hint: None }),
        }
        Submit::Handled
    }

    /// Record a TUI-side error (a rejected local command, not an agent error).
    pub fn push_error(&mut self, text: String) {
        self.messages.push(MessageItem::Error { text, hint: None });
    }

    /// Put the newest queued message back into the input line for editing.
    pub fn recall_last_queued(&mut self) -> bool {
        match self.message_queue.pop() {
            Some(text) => {
                self.saved_draft = Some(std::mem::take(&mut self.input));
                self.input = text;
                self.cursor_pos = self.input.chars().count();
                self.check_completions();
                true
            }
            None => false,
        }
    }

    pub fn submit_input(&mut self) -> Option<String> {
        let text = self.input.trim().to_string();
        if text.is_empty() {
            return None;
        }
        self.input_history.push(text.clone());
        self.history_idx = None;
        self.saved_draft = None;
        self.input.clear();
        self.cursor_pos = 0;
        self.show_completions = false;
        self.show_file_completions = false;
        self.scroll_offset = 0;
        self.overscroll = 0;
        self.auto_scroll = true;
        // A new turn starts unpinned: the block focus follows whatever arrives.
        self.focus_pinned = false;

        if self.is_busy {
            // Enqueue message while agent is active. No notice: the queue box
            // under the spinner already shows what is waiting, as in the REPL.
            self.message_queue.push(text);
            None
        } else {
            self.messages.push(MessageItem::User(text.clone()));
            self.begin_turn_if_idle();
            Some(text)
        }
    }

    pub fn insert_char(&mut self, c: char) {
        self.record_undo();
        // Typing over a selection replaces it, as in any editor.
        self.take_selection();
        self.collapse_selection();
        let c = if c == '\r' { '\n' } else { c };
        let byte_pos = self.input.char_indices().nth(self.cursor_pos).map(|(i, _)| i).unwrap_or(self.input.len());
        self.input.insert(byte_pos, c);
        self.cursor_pos += 1;
        self.check_completions();
    }

    pub fn insert_str(&mut self, s: &str) {
        self.record_undo();
        self.take_selection();
        self.collapse_selection();
        let normalized = s.replace("\r\n", "\n").replace('\r', "\n");
        let byte_pos = self.input.char_indices().nth(self.cursor_pos).map(|(i, _)| i).unwrap_or(self.input.len());
        self.input.insert_str(byte_pos, &normalized);
        self.cursor_pos += normalized.chars().count();
        self.check_completions();
    }

    pub fn backspace(&mut self) {
        if self.has_selection() {
            self.delete_selection();
            return;
        }
        if self.cursor_pos > 0 {
            self.record_undo();
            self.collapse_selection();
            self.cursor_pos -= 1;
            let byte_pos = self.input.char_indices().nth(self.cursor_pos).map(|(i, _)| i).unwrap_or(0);
            self.input.remove(byte_pos);
            self.check_completions();
        }
    }

    pub fn delete_char(&mut self) {
        if self.has_selection() {
            self.delete_selection();
            return;
        }
        if self.cursor_pos < self.input.chars().count() {
            self.record_undo();
            self.collapse_selection();
            let byte_pos = self.input.char_indices().nth(self.cursor_pos).map(|(i, _)| i).unwrap_or(0);
            self.input.remove(byte_pos);
            self.check_completions();
        }
    }

    /// Char index of the start of the word before `from` (skipping blanks first).
    fn word_left_of(&self, from: usize) -> usize {
        let chars: Vec<char> = self.input.chars().collect();
        let mut idx = from.min(chars.len());
        while idx > 0 && chars[idx - 1].is_whitespace() {
            idx -= 1;
        }
        while idx > 0 && !chars[idx - 1].is_whitespace() {
            idx -= 1;
        }
        idx
    }

    /// Char index of the end of the word after `from` (skipping blanks first).
    fn word_right_of(&self, from: usize) -> usize {
        let chars: Vec<char> = self.input.chars().collect();
        let total = chars.len();
        let mut idx = from.min(total);
        while idx < total && chars[idx].is_whitespace() {
            idx += 1;
        }
        while idx < total && !chars[idx].is_whitespace() {
            idx += 1;
        }
        idx
    }

    pub fn delete_word_left(&mut self) {
        if self.has_selection() {
            self.delete_selection();
            return;
        }
        let idx = self.word_left_of(self.cursor_pos);
        self.kill_range(idx, self.cursor_pos, "word");
    }

    pub fn delete_word_right(&mut self) {
        if self.has_selection() {
            self.delete_selection();
            return;
        }
        let idx = self.word_right_of(self.cursor_pos);
        self.kill_range(self.cursor_pos, idx, "word");
    }

    pub fn move_word_left(&mut self) {
        self.cursor_pos = self.word_left_of(self.cursor_pos);
    }

    pub fn move_word_right(&mut self) {
        self.cursor_pos = self.word_right_of(self.cursor_pos);
    }

    /// Ctrl+Shift+Left: grow the selection one word to the left.
    pub fn select_word_left(&mut self) {
        self.set_cursor(self.word_left_of(self.cursor_pos), true);
    }

    /// Ctrl+Shift+Right: grow the selection one word to the right.
    pub fn select_word_right(&mut self) {
        self.set_cursor(self.word_right_of(self.cursor_pos), true);
    }

    /// Ctrl+A: select everything — the whole input, or, with nothing typed, the
    /// whole conversation (Ctrl+C then copies it).
    pub fn select_all(&mut self) {
        if !self.select_all_input() {
            self.select_all_transcript();
        }
    }

    /// Tab (`forward`) or Shift+Tab: step through the open suggestions; with none
    /// open, open or close the focused reasoning / tool block instead.
    pub fn tab(&mut self, forward: bool) {
        if !self.cycle_completion(forward, false) {
            self.toggle_focused_block();
        }
    }

    /// Ctrl+A: select the whole input. False when there is nothing to select.
    pub fn select_all_input(&mut self) -> bool {
        let total = self.input.chars().count();
        if total == 0 {
            return false;
        }
        self.sel_anchor = Some(0);
        self.cursor_pos = total;
        self.completion_cycle = None;
        self.show_completions = false;
        self.show_file_completions = false;
        true
    }

    pub fn delete_to_start(&mut self) {
        self.kill_range(0, self.cursor_pos, "line");
    }

    pub fn delete_to_end(&mut self) {
        let total = self.input.chars().count();
        self.kill_range(self.cursor_pos, total, "line");
    }

    pub fn move_cursor_up(&mut self) -> bool {
        let (line, col) = self.cursor_coordinates();
        if line == 0 {
            return false;
        }
        let target_line = line - 1;
        let mut cur_l = 0;
        let mut target_char = 0;
        let mut visual_w = 0;

        for (idx, c) in self.input.chars().enumerate() {
            if c == '\n' {
                cur_l += 1;
                if cur_l == target_line {
                    target_char = idx + 1;
                }
            } else if cur_l == target_line {
                let w = unicode_width::UnicodeWidthChar::width(c).unwrap_or(1);
                visual_w += w;
                if visual_w <= col {
                    target_char = idx + 1;
                }
            } else if cur_l > target_line {
                break;
            }
        }
        self.cursor_pos = target_char;
        self.check_completions();
        true
    }

    pub fn move_cursor_down(&mut self) -> bool {
        let (line, col) = self.cursor_coordinates();
        let total_lines = self.input.split('\n').count();
        if line + 1 >= total_lines {
            return false;
        }
        let target_line = line + 1;
        let mut cur_l = 0;
        let mut target_char = 0;
        let mut visual_w = 0;

        for (idx, c) in self.input.chars().enumerate() {
            if c == '\n' {
                cur_l += 1;
                if cur_l == target_line {
                    target_char = idx + 1;
                }
            } else if cur_l == target_line {
                let w = unicode_width::UnicodeWidthChar::width(c).unwrap_or(1);
                visual_w += w;
                if visual_w <= col {
                    target_char = idx + 1;
                }
            } else if cur_l > target_line {
                break;
            }
        }
        self.cursor_pos = target_char;
        self.check_completions();
        true
    }

    pub fn move_cursor_left(&mut self) {
        if self.cursor_pos > 0 {
            self.cursor_pos -= 1;
            self.check_completions();
        }
    }

    pub fn move_cursor_right(&mut self) {
        if self.cursor_pos < self.input.chars().count() {
            self.cursor_pos += 1;
            self.check_completions();
        }
    }

    pub fn move_cursor_home(&mut self) {
        self.cursor_pos = 0;
        self.check_completions();
    }

    pub fn move_cursor_end(&mut self) {
        self.cursor_pos = self.input.chars().count();
        self.check_completions();
    }

    /// Tools the legacy CLI treats as interactive; a spinner would fight
    /// whatever prompt the tool itself draws.
    pub fn is_interactive_tool(name: &str) -> bool {
        matches!(
            name,
            "ask_user_questions"
                | "request_user_approval"
                | "run_command"
                | "run_admin_command"
                | "start_background_command"
                | "delete_file"
                | "git_rollback"
        )
    }

    pub fn clear_selection(&mut self) {
        self.sel_anchor = None;
    }

    /// Drop the selection anchor.
    ///
    /// The anchor is a char index into `input`, so any edit that rewrites the
    /// buffer without going through `delete_selection` must invalidate it —
    /// otherwise a stale index outlives the text it pointed into and the TUI
    /// routes Backspace / Ctrl+C / Ctrl+X into the selection paths for a
    /// selection that no longer exists.
    fn collapse_selection(&mut self) {
        self.sel_anchor = None;
    }

    pub fn has_selection(&self) -> bool {
        // The `a <= len` guard keeps a stale anchor inert even if a future edit
        // path forgets to collapse it.
        let len = self.input.chars().count();
        matches!(self.sel_anchor, Some(a) if a != self.cursor_pos && a <= len && self.cursor_pos <= len)
    }

    pub fn selected_text(&self) -> Option<String> {
        let a = self.sel_anchor?;
        let b = self.cursor_pos;
        if a == b {
            return None;
        }
        let (lo, hi) = if a < b { (a, b) } else { (b, a) };
        Some(self.input.chars().take(hi).skip(lo).collect())
    }

    /// Extract clean Markdown text of the Assistant response at a specific index.
    pub fn assistant_content_at(&self, idx: usize) -> Option<String> {
        if let Some(MessageItem::Assistant { content, .. }) = self.messages.get(idx) {
            if !content.trim().is_empty() {
                return Some(content.clone());
            }
        }
        None
    }

    /// Extract clean Markdown text of the most recent Assistant response.
    pub fn last_assistant_content(&self) -> Option<String> {
        for msg in self.messages.iter().rev() {
            if let MessageItem::Assistant { content, .. } = msg {
                if !content.trim().is_empty() {
                    return Some(content.clone());
                }
            }
        }
        None
    }

    /// Extract code blocks (```...```) from the most recent Assistant response.
    pub fn last_assistant_code(&self) -> Option<String> {
        let content = self.last_assistant_content()?;
        let mut blocks = Vec::new();
        let mut in_block = false;
        let mut current_block = Vec::new();

        for line in content.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("```") {
                if in_block {
                    in_block = false;
                    let block_text = current_block.join("\n");
                    if !block_text.trim().is_empty() {
                        blocks.push(block_text);
                    }
                    current_block.clear();
                } else {
                    in_block = true;
                    current_block.clear();
                }
            } else if in_block {
                current_block.push(line);
            }
        }

        if !blocks.is_empty() {
            Some(blocks.join("\n\n"))
        } else {
            None
        }
    }

    /// Move the cursor; `extend` keeps a selection anchored where it started.
    ///
    /// The anchor is the position the cursor occupied *before* the move, which
    /// is what makes the first Shift+arrow select rather than collapse.
    pub fn set_cursor(&mut self, to: usize, extend: bool) {
        let total = self.input.chars().count();
        if extend {
            if self.sel_anchor.is_none() {
                self.sel_anchor = Some(self.cursor_pos);
            }
        } else {
            self.sel_anchor = None;
        }
        self.cursor_pos = to.min(total);
        self.check_completions();
    }

    pub fn extend_selection(&mut self, n: isize) {
        let total = self.input.chars().count() as isize;
        let cur = self.cursor_pos as isize;
        self.set_cursor((cur + n).clamp(0, total) as usize, true);
    }

    /// Cut the selected text out of the input without an undo step of its own
    /// (the caller records one), returning it; `None` when nothing is selected.
    fn take_selection(&mut self) -> Option<String> {
        if !self.has_selection() {
            return None;
        }
        let text = self.selected_text()?;
        let a = self.sel_anchor.unwrap_or(0);
        let lo = a.min(self.cursor_pos);
        let s_byte = self.input.char_indices().nth(lo).map(|(i, _)| i).unwrap_or(0);
        let e_byte = self
            .input
            .char_indices()
            .nth(lo + text.chars().count())
            .map(|(i, _)| i)
            .unwrap_or(self.input.len());
        self.input.drain(s_byte..e_byte);
        self.cursor_pos = lo;
        self.sel_anchor = None;
        Some(text)
    }

    /// Remove the selection, returning the removed text.
    pub fn delete_selection(&mut self) -> Option<String> {
        self.selected_text()?;
        self.record_undo();
        let text = self.take_selection();
        self.check_completions();
        text
    }

    /// Swap the characters either side of the cursor. The CLI binds this to
    /// Ctrl+T, which here drives the thinking level, so it lives on Alt+T
    /// alongside `transpose_words`.
    pub fn transpose_chars(&mut self) {
        let chars: Vec<char> = self.input.chars().collect();
        let n = chars.len();
        if n < 2 {
            return;
        }
        // The cursor has to sit *between* the two characters to swap, so it is
        // pulled inside the buffer first. A cursor at the end transposes the
        // final pair instead of indexing one past the last character.
        let c = self.cursor_pos.min(n - 1);
        if c == 0 {
            // Nothing to the left of the cursor: only advance.
            self.record_undo();
            self.cursor_pos = 1;
            self.check_completions();
            return;
        }
        self.record_undo();
        self.collapse_selection();
        let mut s = chars;
        s.swap(c - 1, c);
        self.input = s.into_iter().collect();
        self.cursor_pos = c + 1;
        self.check_completions();
    }

    /// Swap the words around the cursor.
    pub fn transpose_words(&mut self) {
        let chars: Vec<char> = self.input.chars().collect();
        let n = chars.len();
        if n < 2 {
            return;
        }
        self.record_undo();
        self.collapse_selection();
        let c = self.cursor_pos.min(n);
        let is_sep = |ch: char| ch.is_whitespace();
        let mut a = c;
        while a > 0 && !is_sep(chars[a - 1]) {
            a -= 1;
        }
        let mut b = c;
        while b < n && is_sep(chars[b]) {
            b += 1;
        }
        let mut e = b;
        while e < n && !is_sep(chars[e]) {
            e += 1;
        }
        if a == b && b == e {
            return;
        }
        let left: String = chars[a..c].iter().collect();
        let mid: String = chars[c..b].iter().collect();
        let right: String = chars[b..e].iter().collect();
        // Swap the word ending at the cursor with the one after it.
        let mut rebuilt = String::new();
        rebuilt.push_str(&right);
        rebuilt.push_str(&mid);
        rebuilt.push_str(&left);
        let head: String = chars[..a].iter().collect();
        let tail: String = chars[e..].iter().collect();
        self.input = format!("{head}{rebuilt}{tail}");
        self.cursor_pos = a + right.chars().count() + mid.chars().count();
        self.check_completions();
    }

    /// Push the current buffer so the next destructive edit can be undone.
    pub fn record_undo(&mut self) {
        self.undo_stack.push((self.input.clone(), self.cursor_pos));
        if self.undo_stack.len() > 200 {
            self.undo_stack.remove(0);
        }
    }

    /// Undo the last recorded edit. Returns false when there is nothing to undo.
    pub fn undo(&mut self) -> bool {
        match self.undo_stack.pop() {
            Some((text, cursor)) => {
                self.input = text;
                self.cursor_pos = cursor.min(self.input.chars().count());
                self.collapse_selection();
                self.check_completions();
                true
            }
            None => false,
        }
    }

    /// Remove the `[start, end)` character range and push it onto the kill ring.
    ///
    /// Consecutive kills sharing a `unit` append to the same entry, matching
    /// emacs: Ctrl+K twice then Ctrl+Y brings back both chunks.
    pub fn kill_range(&mut self, start: usize, end: usize, unit: &'static str) {
        let total = self.input.chars().count();
        let start = start.min(total);
        let end = end.min(total).max(start);
        if start == end {
            return;
        }
        self.record_undo();
        self.collapse_selection();
        let s_byte = self.input.char_indices().nth(start).map(|(i, _)| i).unwrap_or(self.input.len());
        let e_byte = self.input.char_indices().nth(end).map(|(i, _)| i).unwrap_or(self.input.len());
        let killed: String = self.input.drain(s_byte..e_byte).collect();
        self.cursor_pos = start;
        if self.last_kill_unit == Some(unit) {
            if let Some(last) = self.kill_ring.last_mut() {
                last.push_str(&killed);
            }
        } else {
            self.kill_ring.push(killed);
        }
        self.last_kill_unit = Some(unit);
        self.yanked = None;
        self.check_completions();
    }

    /// Insert the most recent kill at the cursor.
    pub fn yank(&mut self) -> bool {
        let Some(text) = self.kill_ring.last().cloned() else { return false };
        self.record_undo();
        self.last_kill_unit = None;
        self.yanked = Some(text.clone());
        let mut s = text;
        if self.cursor_pos > 0 && !self.input.is_empty() {
            // Emacs does not add a space; keep it literal.
            let _ = &mut s;
        }
        self.insert_str_raw(&s);
        true
    }

    /// Replace the previous yank with the one before it (Alt+Y).
    pub fn yank_pop(&mut self) -> bool {
        if self.kill_ring.len() < 2 || self.yanked.is_none() {
            return false;
        }
        let popped = self.yanked.take().unwrap_or_default();
        if let Some(byte_pos) = self.input.rfind(&popped) {
            // `rfind` yields a BYTE offset; `cursor_pos` is a CHAR index. Slice
            // by bytes (always a char boundary) and convert for the cursor, or
            // multibyte input gets sliced at the wrong offset.
            let pos = self.input[..byte_pos].chars().count();
            self.input = format!(
                "{}{}",
                &self.input[..byte_pos],
                &self.input[byte_pos + popped.len()..]
            );
            self.cursor_pos = pos;
        }
        let prev = self.kill_ring[self.kill_ring.len() - 2].clone();
        self.yanked = Some(prev.clone());
        self.insert_str_raw(&prev);
        true
    }

    /// `insert_str` without pushing another undo frame (used by yank).
    fn insert_str_raw(&mut self, s: &str) {
        let byte_pos = self.input.char_indices().nth(self.cursor_pos).map(|(i, _)| i).unwrap_or(self.input.len());
        self.collapse_selection();
        self.input.insert_str(byte_pos, s);
        self.cursor_pos += s.chars().count();
        self.check_completions();
    }

    // ── incremental reverse history search (Ctrl+R) ───────────────────────

    pub fn history_search_start(&mut self) -> bool {
        if self.input_history.is_empty() {
            return false;
        }
        self.record_undo();
        self.search_active = true;
        self.search_query.clear();
        self.search_idx = None;
        // Snapshotted exactly once, here. Repeating Ctrl+R must not overwrite
        // it, or Esc would restore a history entry instead of the user's line.
        self.search_saved = Some(self.input.clone());
        self.show_match(self.find_match(None));
        true
    }

    pub fn history_search_push(&mut self, c: char) {
        if !self.search_active {
            self.history_search_start();
        }
        self.search_query.push(c);
        self.show_match(self.find_match(None));
    }

    pub fn history_search_backspace(&mut self) {
        if !self.search_active {
            return;
        }
        self.search_query.pop();
        self.show_match(self.find_match(None));
    }

    /// Move to the next older match; Ctrl+R again repeats.
    ///
    /// The current match is the exclusive upper bound, so each press advances
    /// by exactly one entry. At the oldest match the line is left alone.
    pub fn history_search_prev(&mut self) {
        if !self.search_active {
            self.history_search_start();
            return;
        }
        if let Some(next) = self.find_match(self.search_idx) {
            self.show_match(Some(next));
        }
    }

    /// Accept the current match and leave search mode.
    pub fn history_search_accept(&mut self) {
        self.search_active = false;
        self.search_query.clear();
        self.search_idx = None;
        self.search_saved = None;
        self.last_kill_unit = None;
        self.yanked = None;
    }

    /// Restore whatever was in the line before the search started.
    pub fn history_search_cancel(&mut self) {
        if let Some(saved) = self.search_saved.take() {
            self.input = saved;
        }
        self.cursor_pos = self.input.chars().count();
        self.collapse_selection();
        self.search_active = false;
        self.search_query.clear();
        self.search_idx = None;
        self.last_kill_unit = None;
        self.yanked = None;
        self.check_completions();
    }

    /// Newest entry strictly older than `upper` that matches the query.
    ///
    /// `upper` of `None` means "start from the newest entry". An empty query
    /// matches everything, so Ctrl+R with no text still walks the history.
    fn find_match(&self, upper: Option<usize>) -> Option<usize> {
        let start = upper
            .unwrap_or(self.input_history.len())
            .min(self.input_history.len());
        let q = self.search_query.to_lowercase();
        (0..start)
            .rev()
            .find(|&i| q.is_empty() || self.input_history[i].to_lowercase().contains(&q))
    }

    /// Show the entry at `idx`, or leave the line untouched when there is no
    /// match (an empty result keeps whatever was already displayed).
    fn show_match(&mut self, idx: Option<usize>) {
        self.search_idx = idx;
        if let Some(i) = idx {
            self.input = self.input_history[i].clone();
            self.cursor_pos = self.input.chars().count();
            self.collapse_selection();
        }
    }

    /// Step to the next thinking level and return it.
    ///
    /// The caller must persist the result through the bridge; this only decides
    /// what the next value should be, so the frontend stays responsive even if
    /// the config write fails (in which case the level is not updated here).
    pub fn next_thinking_level(&self) -> Option<String> {
        if self.thinking_levels.is_empty() {
            return None;
        }
        let cur = self.status.thinking_level.as_str();
        let idx = self
            .thinking_levels
            .iter()
            .position(|l| l == cur)
            .map(|i| (i + 1) % self.thinking_levels.len())
            .unwrap_or(0);
        self.thinking_levels.get(idx).cloned()
    }

    /// Adopt a level the bridge confirmed, or record why it was rejected.
    pub fn apply_thinking_level(&mut self, level: &str) {
        self.thinking_error = None;
        self.status.thinking_level = level.to_string();
    }

    pub fn set_thinking_error(&mut self, msg: String) {
        self.messages.push(MessageItem::Error { text: msg.clone(), hint: None });
        self.thinking_error = Some(msg);
    }

    /// Replace the level cycle with what the bridge reports it supports.
    pub fn set_thinking_levels(&mut self, levels: Vec<String>) {
        if !levels.is_empty() {
            self.thinking_levels = levels;
        }
    }

    /// Walk back through input history. Returns false when already at the oldest
    /// entry, so the caller can fall back to another action (e.g. recalling the
    /// newest queued message).
    pub fn history_prev(&mut self) -> bool {
        if self.input_history.is_empty() {
            return false;
        }
        if self.history_idx.is_none() {
            self.saved_draft = Some(self.input.clone());
        }
        let new_idx = match self.history_idx {
            None => self.input_history.len().saturating_sub(1),
            Some(i) => i.saturating_sub(1),
        };
        if self.history_idx == Some(new_idx) {
            return false;
        }
        self.history_idx = Some(new_idx);
        if let Some(hist_entry) = self.input_history.get(new_idx) {
            self.input = hist_entry.clone();
            self.cursor_pos = self.input.chars().count();
            self.collapse_selection();
        }
        true
    }

    pub fn history_next(&mut self) {
        if let Some(i) = self.history_idx {
            if i + 1 < self.input_history.len() {
                let next_idx = i + 1;
                self.history_idx = Some(next_idx);
                self.input = self.input_history[next_idx].clone();
                self.cursor_pos = self.input.chars().count();
                self.collapse_selection();
            } else {
                self.history_idx = None;
                self.input = self.saved_draft.take().unwrap_or_default();
                self.cursor_pos = self.input.chars().count();
                self.collapse_selection();
            }
        }
    }

    /// Scroll all the way back to the first message (Ctrl+Home).
    ///
    /// How far that is cannot be known until the whole conversation has been laid
    /// out, and the frame only lays out what it needs to show. So the offset is set
    /// to "as far as it goes"; the next frame measures the real distance and
    /// `sync_scroll` writes it back. Capping it at the last frame's estimate left
    /// Ctrl+Home one screenful short of the top of a long chat.
    pub fn scroll_to_top(&mut self) {
        self.scroll_offset = usize::MAX;
        self.auto_scroll = false;
    }

    /// After a frame: bring `scroll_offset` back inside what the frame measured.
    pub fn sync_scroll(&mut self) {
        if let Some(used) = self.scroll_used.get() {
            // Only a chat that is scrolled back has an offset to take in: the frame
            // may have clamped it to what exists, or moved it to keep the text the
            // user is reading where it was while new output arrived below.
            if self.scroll_offset > 0 && self.scroll_offset != used {
                self.scroll_offset = used;
                // Pinned text that came to rest at the newest output: following again.
                if used == 0 {
                    self.auto_scroll = true;
                }
            }
        }
    }

    pub fn scroll_up(&mut self, amount: usize) {
        if amount == usize::MAX {
            self.overscroll = 0;
            return self.scroll_to_top();
        }
        // Empty space under the chat goes first: back to the newest, then up.
        let closed = amount.min(self.overscroll);
        self.overscroll -= closed;
        let amount = amount - closed;
        if amount == 0 {
            return;
        }
        let max = self.max_scroll.get();
        self.scroll_offset = self.scroll_offset.saturating_add(amount);
        if max > 0 && self.scroll_offset > max {
            self.scroll_offset = max;
        }
        self.auto_scroll = false;
    }

    /// One notch of the mouse wheel: three rows, and more the longer the wheel keeps
    /// turning (notches less than 70 ms apart in the same direction build up to
    /// twelve), so a long history is crossed in a flick and a slow turn stays fine.
    ///
    /// Turning down stops at the newest output. It never carries on into the empty
    /// space under it (`lower`): the prompt sits right under the chat, so that space
    /// pushes the input field up the screen, and a wheel is turned too freely to be
    /// allowed to move it. Only the scroll keys open that space on purpose.
    pub fn wheel(&mut self, up: bool, now: Instant) {
        let streak = match self.last_wheel {
            Some((at, was_up, n)) if was_up == up && now.duration_since(at) <= WHEEL_STREAK_GAP => (n + 1).min(9),
            _ => 0,
        };
        self.last_wheel = Some((now, up, streak));
        let step = 3 + streak as usize;
        if up {
            self.scroll_up(step);
        } else {
            self.scroll_down(step);
        }
    }

    /// Scroll toward the newest output, stopping at it. With `usize::MAX` this is
    /// "jump to the newest" (Ctrl+End), which also closes any empty space opened
    /// below the chat by `lower`.
    pub fn scroll_down(&mut self, amount: usize) {
        let max = self.max_scroll.get();
        if max > 0 && self.scroll_offset > max {
            self.scroll_offset = max;
        }
        self.scroll_offset = self.scroll_offset.saturating_sub(amount);
        if self.scroll_offset == 0 {
            self.auto_scroll = true;
            if amount == usize::MAX {
                self.overscroll = 0;
            }
        }
    }

    /// Scroll down and, once the newest output is reached, carry on into empty
    /// space under it — as a console lets you scroll below the last line. The chat
    /// then ends part-way up the screen with a gap above the prompt (at most half
    /// the chat's height). Scrolling up closes the gap first, and anything new
    /// (output, a sent message) closes it at once.
    pub fn lower(&mut self, amount: usize) {
        let before = self.scroll_offset;
        self.scroll_down(amount);
        let used = before - self.scroll_offset.min(before);
        if self.scroll_offset == 0 {
            self.overscroll = self.overscroll.saturating_add(amount.saturating_sub(used)).min(OVERSCROLL_MAX);
        }
    }

    pub fn check_completions(&mut self) {
        // Any edit or cursor move ends a Tab cycle: the popup follows the text again.
        self.completion_cycle = None;
        let chars_before: String = self.input.chars().take(self.cursor_pos).collect();

        // Command completion stays open while the typed text is still a viable
        // command prefix. Matching on the registry instead of "contains no
        // space" keeps multi-word commands like `/skill import` reachable while
        // still closing the popup once the text can no longer become a command.
        if chars_before.starts_with('/') {
            let prefix = chars_before.to_lowercase();
            let viable = self
                .completions
                .iter()
                .any(|c| c.name.to_lowercase().starts_with(&prefix));
            if viable {
                self.show_completions = true;
                self.show_file_completions = false;
                self.completion_idx = 0;
                return;
            }
        }

        // Check @file mentions (only at start or word boundary, avoiding user@domain false positive)
        if let Some(at_idx) = chars_before.rfind('@') {
            let is_at_boundary = at_idx == 0 || chars_before[..at_idx].ends_with(|c: char| c.is_whitespace());
            if is_at_boundary {
                let file_query = &chars_before[at_idx + 1..];
                if !file_query.contains(' ') {
                    self.show_file_completions = true;
                    self.show_completions = false;
                    self.file_completion_idx = 0;
                    self.find_matching_files(file_query);
                    return;
                }
            }
        }

        self.show_completions = false;
        self.show_file_completions = false;
    }

    /// Collect @-mention candidates: a pruned recursive walk, ranked.
    ///
    /// Ranking follows the CLI: exact basename prefix beats path prefix beats
    /// any substring, and ties break on path so the list is stable.
    pub fn find_matching_files(&mut self, query: &str) {
        const MAX_FILES: usize = 300;
        const MAX_WALK_DEPTH: usize = 6;
        const MAX_RESULTS: usize = 40;

        let query_lower = query.to_lowercase();
        let mut files: Vec<String> = Vec::new();

        fn walk(dir: &std::path::Path, depth: usize, out: &mut Vec<String>) {
            if depth > MAX_WALK_DEPTH || out.len() >= MAX_FILES {
                return;
            }
            let Ok(entries) = std::fs::read_dir(dir) else { return };
            let mut children: Vec<_> = entries.flatten().collect();
            // Deterministic order so the ranking, not the filesystem, decides.
            children.sort_by_key(|e| e.file_name());
            for entry in children {
                if out.len() >= MAX_FILES {
                    return;
                }
                let name = entry.file_name().to_string_lossy().to_string();
                if name.starts_with('.') || PRUNE_DIRS.contains(&name.as_str()) {
                    continue;
                }
                let path = entry.path();
                if path.is_dir() {
                    walk(&path, depth + 1, out);
                } else {
                    let rel = path
                        .strip_prefix(".")
                        .unwrap_or(&path)
                        .to_string_lossy()
                        .replace('\\', "/");
                    out.push(rel);
                }
            }
        }

        walk(std::path::Path::new("."), 0, &mut files);

        let mut ranked: Vec<(u8, String)> = files
            .into_iter()
            .filter_map(|p| {
                let low = p.to_lowercase();
                let base = low.rsplit('/').next().unwrap_or(&low).to_string();
                // An empty query lists everything at the top rank.
                let rank = if query_lower.is_empty() || base.starts_with(&query_lower) {
                    0
                } else if low.starts_with(&query_lower) {
                    1
                } else if base.contains(&query_lower) {
                    2
                } else if low.contains(&query_lower) {
                    3
                } else {
                    return None;
                };
                Some((rank, p))
            })
            .collect();

        ranked.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        ranked.truncate(MAX_RESULTS);
        self.file_completions = ranked.into_iter().map(|(_, p)| p).collect();
    }

    pub fn cursor_coordinates(&self) -> (usize, usize) {
        let chars: Vec<char> = self.input.chars().take(self.cursor_pos).collect();
        let mut line = 0;
        let mut col = 0;
        for c in chars {
            if c == '\n' {
                line += 1;
                col = 0;
            } else {
                col += unicode_width::UnicodeWidthChar::width(c).unwrap_or(1);
            }
        }
        (line, col)
    }

    pub fn input_lines_count(&self) -> usize {
        let count = self.input.matches('\n').count() + 1;
        count.clamp(1, 14)
    }

    pub fn input_visual_lines_count(&self, available_width: usize) -> usize {
        let width = available_width.max(10);
        let mut total = 0;
        for line_str in self.input.split('\n') {
            let line_w = unicode_width::UnicodeWidthStr::width(line_str) + 2; // account for "❯ " or "  "
            let rows = line_w.div_ceil(width);
            total += rows;
        }
        total.max(1)
    }

    /// Every command that matches what was typed. The popup shows a window of
    /// them and scrolls, so the list is not cut: `/` alone reaches all commands.
    pub fn get_matched_completions(&self) -> Vec<&CompletionItem> {
        let typed: String = match &self.completion_cycle {
            Some(cycle) => cycle.typed.clone(),
            None => self.input.chars().take(self.cursor_pos).collect(),
        };
        let filter = typed.to_lowercase();
        self.completions
            .iter()
            .filter(|c| c.name.to_lowercase().starts_with(&filter))
            .collect()
    }

    /// Tab (`forward`) and Shift+Tab with a suggestion popup open: move to the
    /// next / previous suggestion and show it in the input. The popup stays open
    /// and keeps listing what matched before the first Tab, so Tab can go on
    /// cycling instead of the list collapsing onto the one suggestion shown.
    ///
    /// `step_first` is what the arrow keys use: their first press already moves
    /// the highlight, while the first Tab shows the suggestion highlighted now.
    /// Returns false when no popup is open, so the caller can use the key itself.
    pub fn cycle_completion(&mut self, forward: bool, step_first: bool) -> bool {
        let commands = self.show_completions && !self.completions.is_empty();
        let files = !commands && self.show_file_completions && !self.file_completions.is_empty();
        if !commands && !files {
            return false;
        }
        let first = self.completion_cycle.is_none();
        if first {
            let typed: String = self.input.chars().take(self.cursor_pos).collect();
            if files && typed.rfind('@').is_none() {
                return false;
            }
            self.record_undo();
            self.completion_cycle = Some(CompletionCycle {
                original: self.input.clone(),
                original_cursor: self.cursor_pos,
                tail: self.input.chars().skip(self.cursor_pos).collect(),
                typed,
            });
        }
        let count = if commands { self.get_matched_completions().len() } else { self.file_completions.len() };
        if count == 0 {
            return false;
        }
        let idx = if commands { self.completion_idx } else { self.file_completion_idx };
        let idx = idx.min(count - 1);
        let idx = if first && !step_first {
            // The first Tab shows the highlighted suggestion; Shift+Tab goes back.
            if forward { idx } else { (idx + count - 1) % count }
        } else if forward {
            (idx + 1) % count
        } else {
            (idx + count - 1) % count
        };
        if commands {
            self.completion_idx = idx;
        } else {
            self.file_completion_idx = idx;
        }

        let cycle = self.completion_cycle.clone().expect("set above");
        let (text, cursor) = if commands {
            let name = self.get_matched_completions()[idx].name.clone();
            let cursor = name.chars().count();
            (name, cursor)
        } else {
            let at = cycle.typed.rfind('@').expect("checked above");
            let head = format!("{}{}", &cycle.typed[..=at], self.file_completions[idx]);
            let cursor = head.chars().count();
            (format!("{head}{}", cycle.tail), cursor)
        };
        self.input = text;
        self.cursor_pos = cursor;
        self.sel_anchor = None;
        true
    }

    /// Enter with a suggestion popup open: take the highlighted suggestion and
    /// close the popup. Returns true when it changed the input (the caller then
    /// swallows the Enter instead of submitting).
    pub fn accept_completion(&mut self) -> bool {
        let cycle = self.completion_cycle.take();
        let typed: String = match &cycle {
            Some(c) => c.typed.clone(),
            None => self.input.chars().take(self.cursor_pos).collect(),
        };
        let tail: String = match &cycle {
            Some(c) => c.tail.clone(),
            None => self.input.chars().skip(self.cursor_pos).collect(),
        };
        if self.show_completions && !self.completions.is_empty() {
            self.show_completions = false;
            let name = self.get_matched_completions_for(&typed).get(self.completion_idx).map(|c| c.name.clone());
            let Some(name) = name else { return false };
            self.cursor_pos = name.chars().count();
            self.input = name;
            return true;
        }
        if self.show_file_completions && !self.file_completions.is_empty() {
            self.show_file_completions = false;
            let (Some(path), Some(at)) = (self.file_completions.get(self.file_completion_idx), typed.rfind('@'))
            else {
                return false;
            };
            let mut head = format!("{}{} ", &typed[..=at], path);
            let cursor = head.chars().count();
            head.push_str(tail.trim_start_matches(' '));
            self.input = head;
            self.cursor_pos = cursor;
            return true;
        }
        false
    }

    /// Esc with a suggestion popup open: close it; if Tab had been cycling, the
    /// text goes back to what the user typed.
    pub fn cancel_completion(&mut self) -> bool {
        if !(self.show_completions || self.show_file_completions) {
            return false;
        }
        if let Some(cycle) = self.completion_cycle.take() {
            self.input = cycle.original;
            self.cursor_pos = cycle.original_cursor;
        }
        self.show_completions = false;
        self.show_file_completions = false;
        true
    }

    /// The commands matching `typed` (not the live input), for acceptance after a cycle.
    fn get_matched_completions_for(&self, typed: &str) -> Vec<&CompletionItem> {
        let filter = typed.to_lowercase();
        self.completions.iter().filter(|c| c.name.to_lowercase().starts_with(&filter)).collect()
    }

    pub fn completion_next(&mut self) {
        let count = self.get_matched_completions().len();
        if count > 0 {
            self.completion_idx = (self.completion_idx + 1) % count;
        }
    }

    pub fn completion_prev(&mut self) {
        let count = self.get_matched_completions().len();
        if count > 0 {
            self.completion_idx = if self.completion_idx == 0 { count - 1 } else { self.completion_idx - 1 };
        }
    }

    pub fn file_completion_next(&mut self) {
        let count = self.file_completions.len();
        if count > 0 {
            self.file_completion_idx = (self.file_completion_idx + 1) % count;
        }
    }

    pub fn file_completion_prev(&mut self) {
        let count = self.file_completions.len();
        if count > 0 {
            self.file_completion_idx = if self.file_completion_idx == 0 { count - 1 } else { self.file_completion_idx - 1 };
        }
    }
}

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}
