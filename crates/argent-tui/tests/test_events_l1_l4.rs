use pyo3::prelude::*;
use pyo3::types::PyDict;
use argent_tui::events::{parse_py_event, ArgentEvent};

#[test]
fn test_events_all_valid_types() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        // 1. Content
        let d = PyDict::new_bound(py);
        d.set_item("type", "content").unwrap();
        d.set_item("text", "hello world").unwrap();
        let ev = parse_py_event(&d).unwrap().unwrap();
        if let ArgentEvent::Content(t) = ev {
            assert_eq!(t, "hello world");
        } else {
            panic!("Expected Content event");
        }

        // 2. Thinking
        let d = PyDict::new_bound(py);
        d.set_item("type", "thinking").unwrap();
        d.set_item("text", "deep thought").unwrap();
        let ev = parse_py_event(&d).unwrap().unwrap();
        if let ArgentEvent::Thinking(t) = ev {
            assert_eq!(t, "deep thought");
        } else {
            panic!("Expected Thinking event");
        }

        // 3. ToolStart
        let d = PyDict::new_bound(py);
        d.set_item("type", "tool_start").unwrap();
        d.set_item("name", "read_file").unwrap();
        let args = PyDict::new_bound(py);
        args.set_item("path", "foo.txt").unwrap();
        d.set_item("args", args).unwrap();
        let ev = parse_py_event(&d).unwrap().unwrap();
        if let ArgentEvent::ToolStart { name, args } = ev {
            assert_eq!(name, "read_file");
            assert_eq!(args.get("path").and_then(|v| v.as_str()), Some("foo.txt"));
        } else {
            panic!("Expected ToolStart event");
        }

        // 4. ToolEnd
        let d = PyDict::new_bound(py);
        d.set_item("type", "tool_end").unwrap();
        d.set_item("name", "read_file").unwrap();
        d.set_item("result", "contents").unwrap();
        d.set_item("duration_secs", 0.42).unwrap();
        let ev = parse_py_event(&d).unwrap().unwrap();
        if let ArgentEvent::ToolEnd { name, result, duration_secs } = ev {
            assert_eq!(name, "read_file");
            assert_eq!(result, "contents");
            assert_eq!(duration_secs, Some(0.42));
        } else {
            panic!("Expected ToolEnd event");
        }

        // 5. ApprovalRequest
        let d = PyDict::new_bound(py);
        d.set_item("type", "approval_request").unwrap();
        d.set_item("id", 101).unwrap();
        d.set_item("action", "delete repo").unwrap();
        d.set_item("destructive", true).unwrap();
        d.set_item("grant_key", "repo:del").unwrap();
        let ev = parse_py_event(&d).unwrap().unwrap();
        if let ArgentEvent::ApprovalRequest { id, action, destructive, grant_key, .. } = ev {
            assert_eq!(id, 101);
            assert_eq!(action, "delete repo");
            assert!(destructive);
            assert_eq!(grant_key.as_deref(), Some("repo:del"));
        } else {
            panic!("Expected ApprovalRequest event");
        }

        // 6. Usage
        let d = PyDict::new_bound(py);
        d.set_item("type", "usage").unwrap();
        let data = PyDict::new_bound(py);
        data.set_item("tokens", 5000).unwrap();
        data.set_item("max", 100000).unwrap();
        data.set_item("percent", 5.0).unwrap();
        d.set_item("data", data).unwrap();
        let ev = parse_py_event(&d).unwrap().unwrap();
        if let ArgentEvent::Usage { tokens, max, percent, completion } = ev {
            assert_eq!(tokens, 5000);
            assert_eq!(max, 100000);
            assert_eq!(percent, 5.0);
            assert_eq!(completion, None, "a context report carries no generated-token count");
        } else {
            panic!("Expected Usage event");
        }

        // 7. SelectRequest
        let d = PyDict::new_bound(py);
        d.set_item("type", "select_request").unwrap();
        d.set_item("id", "sel_1").unwrap();
        d.set_item("title", "Select Provider:").unwrap();
        d.set_item("options", vec!["ollama", "zai", "openrouter"]).unwrap();
        d.set_item("default", "zai").unwrap();
        let ev = parse_py_event(&d).unwrap().unwrap();
        if let ArgentEvent::SelectRequest { id, title, options, default_value, multi, initial_checked } = ev {
            assert_eq!(id, "sel_1");
            assert_eq!(title, "Select Provider:");
            assert_eq!(options, vec!["ollama", "zai", "openrouter"]);
            assert_eq!(default_value.as_deref(), Some("zai"));
            assert!(!multi);
            assert_eq!(initial_checked, None);
        } else {
            panic!("Expected SelectRequest event");
        }

        // 8. Done
        let d = PyDict::new_bound(py);
        d.set_item("type", "done").unwrap();
        let ev = parse_py_event(&d).unwrap().unwrap();
        assert!(matches!(ev, ArgentEvent::Done));
    });
}

#[test]
fn test_events_adversarial_malformed_dicts() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        // 1. Missing type field -> returns Ok(None)
        let d = PyDict::new_bound(py);
        let ev = parse_py_event(&d).unwrap();
        assert!(ev.is_none());

        // 2. Unknown type field -> returns Ok(None)
        let d = PyDict::new_bound(py);
        d.set_item("type", "unknown_custom_event").unwrap();
        let ev = parse_py_event(&d).unwrap();
        assert!(ev.is_none());

        // 3. ToolStart with missing args -> defaults to empty HashMap, does not panic
        let d = PyDict::new_bound(py);
        d.set_item("type", "tool_start").unwrap();
        d.set_item("name", "bash").unwrap();
        let ev = parse_py_event(&d).unwrap().unwrap();
        if let ArgentEvent::ToolStart { name, args } = ev {
            assert_eq!(name, "bash");
            assert!(args.is_empty());
        }

        // 4. ApprovalRequest with missing id -> defaults to 0
        let d = PyDict::new_bound(py);
        d.set_item("type", "approval_request").unwrap();
        d.set_item("action", "test").unwrap();
        let ev = parse_py_event(&d).unwrap().unwrap();
        if let ArgentEvent::ApprovalRequest { id, .. } = ev {
            assert_eq!(id, 0);
        }

        // 5. Usage with missing data subdict -> defaults to 0
        let d = PyDict::new_bound(py);
        d.set_item("type", "usage").unwrap();
        let ev = parse_py_event(&d).unwrap().unwrap();
        if let ArgentEvent::Usage { tokens, max, percent, completion } = ev {
            assert_eq!(tokens, 0);
            assert_eq!(max, 0);
            assert_eq!(percent, 0.0);
            assert_eq!(completion, None);
        }
    });
}

/// What the providers actually send (`providers._parse_openai_usage`, the Anthropic
/// stream): a finished request's token counts, with none of the context fields.
#[test]
fn a_providers_usage_report_carries_the_generated_token_count() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let usage = |pairs: &[(&str, i64)]| {
            let d = PyDict::new_bound(py);
            d.set_item("type", "usage").unwrap();
            let data = PyDict::new_bound(py);
            for (k, v) in pairs {
                data.set_item(*k, *v).unwrap();
            }
            d.set_item("data", data).unwrap();
            parse_py_event(&d).unwrap().unwrap()
        };

        match usage(&[("prompt", 1200), ("completion", 340), ("total", 1540)]) {
            ArgentEvent::Usage { tokens, max, completion, .. } => {
                assert_eq!(completion, Some(340));
                assert_eq!((tokens, max), (0, 0), "it says nothing about the context window");
            }
            other => panic!("Expected Usage, got {other:?}"),
        }
        match usage(&[("completion_tokens", 77)]) {
            ArgentEvent::Usage { completion, .. } => assert_eq!(completion, Some(77)),
            other => panic!("Expected Usage, got {other:?}"),
        }
        // A provider that sends no count (or a zero one) leaves nothing to check against.
        match usage(&[("prompt", 1200), ("completion", 0)]) {
            ArgentEvent::Usage { completion, .. } => assert_eq!(completion, None),
            other => panic!("Expected Usage, got {other:?}"),
        }
    });
}
