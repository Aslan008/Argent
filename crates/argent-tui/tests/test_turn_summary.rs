//! Итог хода: одна строка чисел, подробности — по клику.
//!
//! Сводка приходила четырьмя строками (`/snapshot verbose`), и после каждого ответа
//! между репликами стояла таблица. Теперь на экране одна строка «▸ время · токены ·
//! контекст · клик — подробности», а развёрнутый отчёт (те же строки, что печатает
//! REPL) открывается нажатием по ней.

use argent_tui::app::{App, ClickAction, MessageItem};
use argent_tui::events::ArgentEvent;
use argent_tui::theme::UiStyle;
use argent_tui::ui::draw_ui;
use ratatui::backend::TestBackend;
use ratatui::Terminal;

fn frame(app: &App, w: u16, h: u16) -> Vec<String> {
    let mut terminal = Terminal::new(TestBackend::new(w, h)).expect("terminal");
    terminal.draw(|f| draw_ui(f, app)).expect("draw");
    let buf = terminal.backend().buffer().clone();
    let area = buf.area;
    let mut out = Vec::new();
    for y in 0..area.height {
        let mut row = String::new();
        for x in 0..area.width {
            row.push_str(buf.content[(y * area.width + x) as usize].symbol());
        }
        out.push(row);
    }
    out
}

/// Шестнадцать минут работы, сто восемь инструментов и четыре строки verbose-отчёта.
fn snapshot_event() -> ArgentEvent {
    ArgentEvent::TurnSnapshot {
        total_s: 1060.0,
        ttft_s: 6.2,
        tool_s: 211.0,
        tool_calls: 108,
        prompt_tokens: Some(188_300.0),
        completion_tokens: Some(1400.0),
        cached_tokens: Some(186_600.0),
        cost: Some(0.01),
        ctx_tokens: Some(164_600.0),
        ctx_max: Some(1_000_000.0),
        ctx_percent: Some(16.4),
        lines: vec![
            "  [dim]┄ 17m 40s (ожидание: 6.2s · модель: 14m 7s · инструменты: 3m 31s)[/dim]".into(),
            "  [dim]┄ 188.3k→1.4k tok · cached: 186.6k (99%) · sum 189.7k tok · last~ $0.0100[/dim]".into(),
            "  [dim]┄ ctx 164.6k/1M (16%) · sys 3.8k · tools 13.1k · history 160.9k[/dim]".into(),
            "  [dim]┄ 108 tools[/dim]".into(),
        ],
        context_line: Some("[dim]Context: 16.4% (164634/1000000 tokens)[/dim]".into()),
        compaction: None,
    }
}

fn app_with_snapshot(style: UiStyle) -> App {
    let mut app = App::new();
    app.ui.set_style(style);
    app.handle_event(snapshot_event());
    app
}

#[test]
fn the_summary_is_one_row_until_the_person_asks_for_more() {
    for style in [UiStyle::Classic, UiStyle::Premium] {
        let app = app_with_snapshot(style);
        let rows = frame(&app, 120, 30);
        let folded: Vec<&String> = rows.iter().filter(|r| r.contains("клик — подробности")).collect();
        assert_eq!(folded.len(), 1, "{style:?}: одна свёрнутая строка:\n{}", rows.join("\n"));

        let row = folded[0];
        assert!(row.contains('▸'), "{style:?}: видно, что строка раскрывается: {row}");
        // The numbers the person needs between answers, all on that one row.
        assert!(row.contains("17m 40s"), "{style:?}: {row}");
        assert!(row.contains("188.3k→1.4k tok"), "{style:?}: {row}");
        assert!(row.contains("108 tools"), "{style:?}: {row}");
        assert!(row.contains("ctx 164.6k/1M"), "{style:?}: {row}");
        // The breakdown is hidden, not lost.
        assert!(
            !rows.iter().any(|r| r.contains("history 160.9k")),
            "{style:?}: подробности свёрнуты:\n{}",
            rows.join("\n")
        );
    }
}

#[test]
fn a_click_opens_the_repls_breakdown() {
    for style in [UiStyle::Classic, UiStyle::Premium] {
        let mut app = app_with_snapshot(style);
        assert!(app.toggle_block_by_idx(0), "{style:?}: строка — кнопка");

        let rows = frame(&app, 120, 30);
        let row = rows
            .iter()
            .find(|r| r.contains("клик — свернуть"))
            .unwrap_or_else(|| panic!("{style:?}: нет подписи раскрытой строки:\n{}", rows.join("\n")));
        assert!(row.contains('▾'), "{style:?}: {row}");
        assert!(
            rows.iter().any(|r| r.contains("history 160.9k")),
            "{style:?}: подробности раскрыты:\n{}",
            rows.join("\n")
        );
        assert!(
            rows.iter().any(|r| r.contains("Context: 16.4%")),
            "{style:?}: счётчик контекста тоже под строкой:\n{}",
            rows.join("\n")
        );
    }
}

#[test]
fn the_folded_row_is_built_from_the_numbers() {
    // The bridge sends four typed rows in verbose mode; the folded row must stand on
    // its own and read the same whichever mode built `lines`.
    let app = app_with_snapshot(UiStyle::Premium);
    match app.messages.last() {
        Some(MessageItem::Snapshot { summary, expanded, lines, .. }) => {
            assert_eq!(
                summary,
                "17m 40s · 188.3k→1.4k tok · cached 186.6k (99%) · 108 tools · ctx 164.6k/1M (16%) · $0.0100"
            );
            assert!(!expanded, "свёрнута по умолчанию");
            assert_eq!(lines.len(), 4, "подробности на месте, их просто не видно");
        }
        other => panic!("expected a snapshot, got {other:?}"),
    }
}

#[test]
fn a_turn_that_compacted_the_history_says_so_on_the_row() {
    // The context does get compacted mid-turn. Until now that fact only existed as a
    // separate line in the chat; the summary is where a person looks back at the turn.
    let mut app = App::new();
    let mut event = snapshot_event();
    if let ArgentEvent::TurnSnapshot { compaction, .. } = &mut event {
        *compaction = Some("2 сжатия · 128k→12k".to_string());
    }
    app.handle_event(event);

    let rows = frame(&app, 140, 30);
    let row = rows
        .iter()
        .find(|r| r.contains("клик — подробности"))
        .unwrap_or_else(|| panic!("нет сводки:\n{}", rows.join("\n")));
    assert!(
        row.contains("2 сжатия · 128k→12k"),
        "сжатия должны быть видны в свёрнутой строке: {row}"
    );
}

#[test]
fn an_ordinary_turn_says_nothing_about_compaction() {
    let app = app_with_snapshot(UiStyle::Premium);
    let rows = frame(&app, 140, 30);
    let row = rows
        .iter()
        .find(|r| r.contains("клик — подробности"))
        .unwrap_or_else(|| panic!("нет сводки:\n{}", rows.join("\n")));
    assert!(
        !row.contains("сжати"),
        "без сжатий — без упоминания о них: {row}"
    );
}

#[test]
fn a_summary_with_nothing_to_open_is_not_a_button() {
    // `compact`/`off` send one line without a context meter. Folding it would draw a
    // button whose only content is itself — the REPL's line is shown as it is.
    let mut app = App::new();
    app.handle_event(ArgentEvent::TurnSnapshot {
        total_s: 2.0,
        ttft_s: 0.0,
        tool_s: 0.0,
        tool_calls: 0,
        prompt_tokens: Some(10.0),
        completion_tokens: Some(5.0),
        cached_tokens: None,
        cost: None,
        ctx_tokens: None,
        ctx_max: None,
        ctx_percent: None,
        lines: vec!["  [dim]┄ 2.0s · 10→5 tok[/dim]".into()],
        context_line: None,
        compaction: None,
    });

    let rows = frame(&app, 100, 20);
    assert!(
        !rows.iter().any(|r| r.contains("клик — подробности")),
        "нечего раскрывать — нет и кнопки:\n{}",
        rows.join("\n")
    );
    assert!(
        rows.iter().any(|r| r.contains("2.0s · 10→5 tok")),
        "строка REPL на экране:\n{}",
        rows.join("\n")
    );
}

#[test]
fn the_row_says_out_loud_that_it_can_be_clicked() {
    assert_eq!(
        ClickAction::ToggleSnapshot(0).hint(),
        Some("Клик — подробности хода: время, токены, контекст")
    );

    let mut app = app_with_snapshot(UiStyle::Premium);
    assert!(app.toggle_block_by_idx(0));
    assert!(matches!(app.messages.last(), Some(MessageItem::Snapshot { expanded: true, .. })));
    assert!(app.toggle_block_by_idx(0));
    assert!(matches!(app.messages.last(), Some(MessageItem::Snapshot { expanded: false, .. })));
}
