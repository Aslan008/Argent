use pyo3::prelude::{Bound, PyAny, PyResult};
use pyo3::types::{
    PyAnyMethods, PyBool, PyBoolMethods, PyDict, PyDictMethods, PyList, PyListMethods, PyTuple,
    PyTupleMethods,
};
use std::collections::HashMap;

/// Convert a Python value into JSON, keeping its type.
///
/// Tool arguments used to be flattened with `str()`, so a numeric `offset=10`
/// reached the renderer as the STRING "10" (and a bool as "True"): anything that
/// branched on the value's type — the `file:10-50` line range in a tool header —
/// silently never fired. Only objects JSON cannot express fall back to `str()`.
pub fn py_to_json(obj: &Bound<'_, PyAny>) -> serde_json::Value {
    use serde_json::Value;
    if obj.is_none() {
        return Value::Null;
    }
    // bool before int: Python's True is an int subclass.
    if let Ok(b) = obj.downcast::<PyBool>() {
        return Value::Bool(b.is_true());
    }
    if let Ok(i) = obj.extract::<i64>() {
        return Value::from(i);
    }
    if let Ok(f) = obj.extract::<f64>() {
        return serde_json::Number::from_f64(f).map(Value::Number).unwrap_or(Value::Null);
    }
    if let Ok(s) = obj.extract::<String>() {
        return Value::String(s);
    }
    if let Ok(d) = obj.downcast::<PyDict>() {
        let mut m = serde_json::Map::new();
        for (k, v) in d.iter() {
            let key = k.extract::<String>().unwrap_or_else(|_| k.to_string());
            m.insert(key, py_to_json(&v));
        }
        return Value::Object(m);
    }
    if let Ok(l) = obj.downcast::<PyList>() {
        return Value::Array(l.iter().map(|v| py_to_json(&v)).collect());
    }
    if let Ok(t) = obj.downcast::<PyTuple>() {
        return Value::Array(t.iter().map(|v| py_to_json(&v)).collect());
    }
    Value::String(obj.to_string())
}

/// How the REPL words an approval question (`approval.approval_prompt_spec`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ApprovalPrompt {
    pub question: String,
    pub choices: Vec<String>,
    /// `decisions[i]` is what `choices[i]` stands for: `once` / `always` / `deny`.
    pub decisions: Vec<String>,
    /// `confirm` (a y/N question) as opposed to a `select` list.
    pub confirm: bool,
}

/// One block of a conversation that is being shown again (a restored session).
///
/// Not a live event: replaying a saved dialogue through `Content` / `ToolStart` would mark the UI
/// busy, start a turn clock and arm the queue. These blocks are only put on the screen.
#[derive(Debug, Clone, PartialEq)]
pub enum HistoryItem {
    User(String),
    Assistant { text: String, thinking: String },
    Tool {
        name: String,
        args: HashMap<String, serde_json::Value>,
        result: String,
    },
    /// A muted line (a separator, "N older messages are not shown").
    Note(String),
}

#[derive(Debug, Clone)]
pub enum ArgentEvent {
    Content(String),
    ContentReplace(String),
    Thinking(String),
    /// Second-pass verification stream (`◈ Перепроверка`).
    Rethink(String),
    /// A line styled like the REPL's `print_system`.
    System(String),
    /// The start-screen rule; carries the version string.
    Banner(String),
    ToolStart {
        name: String,
        args: HashMap<String, serde_json::Value>,
    },
    ToolEnd {
        name: String,
        result: String,
        duration_secs: Option<f64>,
    },
    ApprovalRequest {
        id: usize,
        action: String,
        destructive: bool,
        grant_key: Option<String>,
        /// The terminal's own wording of the question, when the bridge sent it.
        prompt: Option<ApprovalPrompt>,
    },
    Checkpoint {
        sha: String,
        label: String,
    },
    /// A saved conversation put back on the screen. With `replace` the transcript is cleared
    /// first (a session loaded in the middle of another one).
    History {
        items: Vec<HistoryItem>,
        replace: bool,
    },
    /// "Back to here" went through: drop the last message (and what followed) from the
    /// transcript and put its text back in the prompt. `detail` says what happened to the files.
    Rewound {
        text: String,
        detail: String,
    },
    Diff {
        file: String,
        diff: String,
    },
    Notice(String),
    /// The model is still streaming a tool call's raw JSON arguments.
    ToolGenerating { name: String, bytes: usize, delta: String },
    Error {
        text: String,
        /// Actionable explanation from `error_help.explain_error`, when known.
        hint: Option<String>,
    },
    /// Two kinds of report share this event. The context window's fill (`tokens`,
    /// `max`, `percent`), and a provider's count of the tokens a finished request
    /// generated (`completion`) — what the generation speed is checked against.
    /// The fields a report does not carry stay at 0 / `None`.
    Usage {
        tokens: usize,
        max: usize,
        percent: f64,
        completion: Option<usize>,
    },
    /// Per-turn metrics emitted once, just before `Done`.
    TurnSnapshot {
        total_s: f64,
        ttft_s: f64,
        tool_s: f64,
        tool_calls: usize,
        prompt_tokens: Option<f64>,
        completion_tokens: Option<f64>,
        cached_tokens: Option<f64>,
        cost: Option<f64>,
        ctx_tokens: Option<f64>,
        ctx_max: Option<f64>,
        ctx_percent: Option<f64>,
        /// The REPL's own snapshot text (Rich markup), built on the Python side.
        lines: Vec<String>,
        /// The `Context: …` meter printed under it.
        context_line: Option<String>,
    },
    SelectRequest {
        id: String,
        title: String,
        options: Vec<String>,
        default_value: Option<String>,
        multi: bool,
        initial_checked: Option<Vec<bool>>,
    },
    Done,
    /// The bridge acknowledged a stop request but the turn is still unwinding.
    /// Distinct from `Done` so the frontend does not treat the turn as finished
    /// (which would let a queued message start a concurrent second turn).
    Cancelling,
    /// A command handler asked the frontend to quit (e.g. `/exit`). Its own side
    /// effects, such as exporting the chat, have already run.
    Exit,
    /// `/ui premium|classic`: the saved look changed, apply it now.
    UiStyle(String),
}

pub fn parse_py_event(dict: &Bound<'_, PyDict>) -> PyResult<Option<ArgentEvent>> {
    let event_type_obj = match dict.get_item("type")? {
        Some(val) => val,
        None => return Ok(None),
    };
    let event_type: String = event_type_obj.extract()?;

    match event_type.as_str() {
        "content" => {
            let text: String = dict.get_item("text")?.map(|v| v.extract()).unwrap_or(Ok(String::new()))?;
            Ok(Some(ArgentEvent::Content(text)))
        }
        "content_replace" => {
            let text: String = dict.get_item("text")?.map(|v| v.extract()).unwrap_or(Ok(String::new()))?;
            Ok(Some(ArgentEvent::ContentReplace(text)))
        }
        "thinking" => {
            let text: String = dict.get_item("text")?.map(|v| v.extract()).unwrap_or(Ok(String::new()))?;
            let rethink = dict
                .get_item("rethink")?
                .and_then(|v| v.extract::<bool>().ok())
                .unwrap_or(false);
            Ok(Some(if rethink {
                ArgentEvent::Rethink(text)
            } else {
                ArgentEvent::Thinking(text)
            }))
        }
        "tool_start" => {
            let name: String = dict.get_item("name")?.map(|v| v.extract()).unwrap_or(Ok(String::new()))?;
            let mut args = HashMap::new();
            if let Some(args_dict) = dict.get_item("args")? {
                if let Ok(d) = args_dict.downcast::<PyDict>() {
                    for (k, v) in d.iter() {
                        let k_str: String = k.extract().unwrap_or_default();
                        args.insert(k_str, py_to_json(&v));
                    }
                }
            }
            Ok(Some(ArgentEvent::ToolStart { name, args }))
        }
        "tool_end" => {
            let name: String = dict.get_item("name")?.map(|v| v.extract()).unwrap_or(Ok(String::new()))?;
            let result: String = dict.get_item("result")?.map(|v| v.extract()).unwrap_or(Ok(String::new()))?;
            let duration_secs: Option<f64> = dict.get_item("duration_secs")?.and_then(|v| v.extract().ok());
            Ok(Some(ArgentEvent::ToolEnd {
                name,
                result,
                duration_secs,
            }))
        }
        "approval_request" => {
            let id: usize = dict.get_item("id")?.map(|v| v.extract()).unwrap_or(Ok(0))?;
            let action: String = dict.get_item("action")?.map(|v| v.extract()).unwrap_or(Ok(String::new()))?;
            let destructive: bool = dict.get_item("destructive")?.map(|v| v.extract()).unwrap_or(Ok(false))?;
            let grant_key: Option<String> = dict.get_item("grant_key")?.and_then(|v| v.extract().ok());
            let choices: Vec<String> = dict
                .get_item("choices")?
                .and_then(|v| v.extract().ok())
                .unwrap_or_default();
            let decisions: Vec<String> = dict
                .get_item("decisions")?
                .and_then(|v| v.extract().ok())
                .unwrap_or_default();
            let prompt = if choices.is_empty() || choices.len() != decisions.len() {
                None
            } else {
                Some(ApprovalPrompt {
                    question: dict
                        .get_item("question")?
                        .and_then(|v| v.extract().ok())
                        .unwrap_or_default(),
                    choices,
                    decisions,
                    confirm: dict
                        .get_item("prompt_kind")?
                        .and_then(|v| v.extract::<String>().ok())
                        .as_deref()
                        == Some("confirm"),
                })
            };
            Ok(Some(ArgentEvent::ApprovalRequest {
                id,
                action,
                destructive,
                grant_key,
                prompt,
            }))
        }
        "checkpoint" => {
            let sha: String = dict.get_item("sha")?.map(|v| v.extract()).unwrap_or(Ok(String::new()))?;
            let label: String = dict.get_item("label")?.map(|v| v.extract()).unwrap_or(Ok(String::new()))?;
            Ok(Some(ArgentEvent::Checkpoint { sha, label }))
        }
        "history" => {
            let mut items = Vec::new();
            if let Some(list) = dict.get_item("items")? {
                if let Ok(list) = list.downcast::<PyList>() {
                    for entry in list.iter() {
                        let Ok(d) = entry.downcast::<PyDict>() else { continue };
                        let text = |key: &str| -> String {
                            d.get_item(key)
                                .ok()
                                .flatten()
                                .and_then(|v| v.extract::<String>().ok())
                                .unwrap_or_default()
                        };
                        let role = text("role");
                        match role.as_str() {
                            "user" => items.push(HistoryItem::User(text("text"))),
                            "assistant" => items.push(HistoryItem::Assistant {
                                text: text("text"),
                                thinking: text("thinking"),
                            }),
                            "tool" => {
                                let mut args = HashMap::new();
                                if let Ok(Some(a)) = d.get_item("args") {
                                    if let Ok(ad) = a.downcast::<PyDict>() {
                                        for (k, v) in ad.iter() {
                                            let k_str: String = k.extract().unwrap_or_default();
                                            args.insert(k_str, py_to_json(&v));
                                        }
                                    }
                                }
                                items.push(HistoryItem::Tool {
                                    name: text("name"),
                                    args,
                                    result: text("result"),
                                });
                            }
                            "note" => items.push(HistoryItem::Note(text("text"))),
                            _ => {}
                        }
                    }
                }
            }
            let replace = dict
                .get_item("replace")?
                .and_then(|v| v.extract::<bool>().ok())
                .unwrap_or(false);
            Ok(Some(ArgentEvent::History { items, replace }))
        }
        "rewound" => {
            let text: String = dict.get_item("text")?.map(|v| v.extract()).unwrap_or(Ok(String::new()))?;
            let detail: String = dict.get_item("detail")?.map(|v| v.extract()).unwrap_or(Ok(String::new()))?;
            Ok(Some(ArgentEvent::Rewound { text, detail }))
        }
        "diff" => {
            let file: String = dict.get_item("file")?.map(|v| v.extract()).unwrap_or(Ok(String::new()))?;
            let diff: String = dict.get_item("diff")?.map(|v| v.extract()).unwrap_or(Ok(String::new()))?;
            Ok(Some(ArgentEvent::Diff { file, diff }))
        }
        "notice" => {
            let text: String = dict.get_item("text")?.map(|v| v.extract()).unwrap_or(Ok(String::new()))?;
            Ok(Some(ArgentEvent::Notice(text)))
        }
        "error" => {
            let text: String = dict.get_item("text")?.map(|v| v.extract()).unwrap_or(Ok(String::new()))?;
            let hint: Option<String> = dict.get_item("hint")?.and_then(|v| v.extract().ok());
            Ok(Some(ArgentEvent::Error { text, hint }))
        }
        "usage" => {
            let mut tokens = 0;
            let mut max = 0;
            let mut percent = 0.0;
            let mut completion = None;
            if let Some(data) = dict.get_item("data")? {
                if let Ok(d) = data.downcast::<PyDict>() {
                    tokens = d.get_item("tokens")?.and_then(|v| v.extract().ok()).unwrap_or(0);
                    max = d.get_item("max")?.and_then(|v| v.extract().ok()).unwrap_or(0);
                    percent = d.get_item("percent")?.and_then(|v| v.extract().ok()).unwrap_or(0.0);
                    // `providers._parse_openai_usage` and the Anthropic stream both say `completion`.
                    completion = ["completion", "completion_tokens"]
                        .iter()
                        .find_map(|k| d.get_item(*k).ok().flatten().and_then(|v| v.extract::<usize>().ok()))
                        .filter(|n| *n > 0);
                }
            }
            Ok(Some(ArgentEvent::Usage { tokens, max, percent, completion }))
        }
        "turn_snapshot" => {
            let num = |k: &str| -> Option<f64> {
                let data = dict.get_item("data").ok().flatten()?;
                let d = data.downcast::<PyDict>().ok()?;
                let v = d.get_item(k).ok().flatten()?;
                v.extract::<f64>().ok()
            };
            Ok(Some(ArgentEvent::TurnSnapshot {
                total_s: num("total_s").unwrap_or(0.0),
                ttft_s: num("ttft_s").unwrap_or(0.0),
                tool_s: num("tool_s").unwrap_or(0.0),
                tool_calls: num("tool_calls").unwrap_or(0.0) as usize,
                prompt_tokens: num("prompt_tokens"),
                completion_tokens: num("completion_tokens"),
                cached_tokens: num("cached_tokens"),
                cost: num("cost"),
                ctx_tokens: num("ctx_tokens"),
                ctx_max: num("ctx_max"),
                ctx_percent: num("ctx_percent"),
                lines: dict
                    .get_item("lines")?
                    .and_then(|v| v.extract::<Vec<String>>().ok())
                    .unwrap_or_default(),
                context_line: dict.get_item("context_line")?.and_then(|v| v.extract::<String>().ok()),
            }))
        }
        "rethink" => {
            let text: String = dict.get_item("text")?.map(|v| v.extract()).unwrap_or(Ok(String::new()))?;
            Ok(Some(ArgentEvent::Rethink(text)))
        }
        "system" => {
            let text: String = dict.get_item("text")?.map(|v| v.extract()).unwrap_or(Ok(String::new()))?;
            Ok(Some(ArgentEvent::System(text)))
        }
        "banner" => {
            let version: String = dict.get_item("version")?.map(|v| v.extract()).unwrap_or(Ok(String::new()))?;
            Ok(Some(ArgentEvent::Banner(version)))
        }
        "select_request" => {
            let id: String = dict.get_item("id")?.map(|v| v.extract()).unwrap_or(Ok(String::new()))?;
            let title: String = dict.get_item("title")?.map(|v| v.extract()).unwrap_or(Ok(String::new()))?;
            let mut options = Vec::new();
            if let Some(opts_obj) = dict.get_item("options")? {
                if let Ok(list) = opts_obj.extract::<Vec<String>>() {
                    options = list;
                }
            }
            let default_value: Option<String> = dict.get_item("default")?.and_then(|v| v.extract().ok());
            let multi: bool = dict.get_item("multi")?.and_then(|v| v.extract().ok()).unwrap_or(false);
            let initial_checked: Option<Vec<bool>> = dict.get_item("initial_checked")?.and_then(|v| v.extract().ok());
            Ok(Some(ArgentEvent::SelectRequest {
                id,
                title,
                options,
                default_value,
                multi,
                initial_checked,
            }))
        }
        "tool_generating" => {
            let name: String = dict.get_item("name")?.map(|v| v.extract()).unwrap_or(Ok(String::new()))?;
            let bytes: usize = dict.get_item("bytes")?.and_then(|v| v.extract().ok()).unwrap_or(0);
            let delta: String = dict
                .get_item("delta")?
                .and_then(|v| v.extract().ok())
                .unwrap_or_default();
            Ok(Some(ArgentEvent::ToolGenerating { name, bytes, delta }))
        }
        "done" => Ok(Some(ArgentEvent::Done)),
        "cancelling" => Ok(Some(ArgentEvent::Cancelling)),
        "exit" => Ok(Some(ArgentEvent::Exit)),
        "ui_style" => {
            let style: String = dict.get_item("style")?.map(|v| v.extract()).unwrap_or(Ok(String::new()))?;
            Ok(Some(ArgentEvent::UiStyle(style)))
        }
        _ => Ok(None),
    }
}
