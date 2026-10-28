//! Очередь и вложения: то, что раньше было картинкой без управления.
//!
//! Очередь рисовалась внутри чата — на неё нельзя было нажать, и о том, что
//! сообщение можно править, приходилось узнавать из `/queue`. Теперь это панель
//! над вводом, где каждая строка — кнопка. Вложения раньше видны не были вовсе.

use argent_tui::app::{parse_attachments, App, Attachment, ClickAction, QueuedMessage, RgbGrid};
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

fn busy_with_queue(messages: &[&str]) -> App {
    let mut app = App::new();
    app.is_busy = true;
    for m in messages {
        app.message_queue.push(QueuedMessage::plain(*m));
    }
    app
}

fn row_with(app: &App, needle: &str) -> usize {
    frame(app, 100, 30)
        .iter()
        .position(|r| r.contains(needle))
        .unwrap_or_else(|| panic!("нет строки {needle:?} в кадре:\n{}", frame(app, 100, 30).join("\n")))
}

// ── очередь ────────────────────────────────────────────────────────────────

#[test]
fn the_queue_is_a_panel_above_the_prompt_and_every_row_is_a_button() {
    let mut app = busy_with_queue(&["первое сообщение", "второе", "третье"]);
    // Занятый агент рисует поле ввода только когда в нём что-то есть.
    app.input = "черновик".into();
    app.cursor_pos = app.input.chars().count();
    let rows = frame(&app, 100, 30);
    let head = row_with(&app, "Очередь сообщений (3)");
    let prompt = rows.iter().position(|r| r.contains('❯')).expect("поле ввода");
    assert!(head < prompt, "панель стоит над полем ввода:\n{}", rows.join("\n"));

    // Три сообщения: все видны, пронумерованы, и в конце каждой строки ✕.
    for (i, text) in ["первое сообщение", "второе", "третье"].iter().enumerate() {
        let row = head + 1 + i;
        assert!(rows[row].contains(text), "строка {row} потеряла текст: {:?}", rows[row]);
        assert!(rows[row].contains(&format!("{}. ", i + 1)), "нет номера: {:?}", rows[row]);
        assert!(rows[row].trim_end().ends_with('✕'), "нет ✕ в конце: {:?}", rows[row]);
    }
}

#[test]
fn clicking_a_queued_row_edits_it_and_the_cross_drops_it() {
    let app = busy_with_queue(&["старое", "второе"]);
    let head = row_with(&app, "Очередь сообщений (2)");
    let row = head + 1;

    assert_eq!(app.get_clicked_action(5, row as u16), Some(ClickAction::EditQueue(0)));
    assert_eq!(app.get_clicked_action(99, row as u16), Some(ClickAction::DropQueue(0)));
}

#[test]
fn editing_a_queued_message_keeps_its_place_in_the_turn_order() {
    let mut app = busy_with_queue(&["первое", "второе", "третье"]);

    assert!(app.edit_queued(1));
    assert_eq!(app.input, "второе", "сообщение пришло в поле ввода");
    assert_eq!(app.message_queue.len(), 3, "слот остался на месте");
    assert_eq!(app.queue_edit, Some(1));

    app.input = "второе, но понятнее".to_string();
    assert!(app.submit_input().is_none(), "ход не начинается, пока агент занят");
    assert_eq!(app.message_queue, vec!["первое", "второе, но понятнее", "третье"]);
    assert_eq!(app.queue_edit, None, "правка закончена");
}

#[test]
fn esc_leaves_the_edit_and_puts_the_interrupted_draft_back() {
    let mut app = busy_with_queue(&["в очереди"]);
    app.input = "мой черновик".to_string();
    app.cursor_pos = app.input.chars().count();

    app.edit_queued(0);
    assert_eq!(app.input, "в очереди");

    app.cancel_queue_edit();
    assert_eq!(app.input, "мой черновик");
    assert_eq!(app.queue_edit, None);
    assert_eq!(app.message_queue, vec!["в очереди"], "сообщение не тронуто");
}

#[test]
fn dropping_the_message_being_edited_ends_the_edit() {
    let mut app = busy_with_queue(&["первое", "второе"]);
    app.edit_queued(0);

    let removed = app.drop_queued(0);

    assert_eq!(removed.as_deref(), Some("первое"));
    assert_eq!(app.queue_edit, None, "слота больше нет — правка закончена");
    assert_eq!(app.message_queue, vec!["второе"]);
}

#[test]
fn the_edit_marker_follows_the_list_when_an_earlier_message_goes() {
    let mut app = busy_with_queue(&["a", "b", "c"]);
    app.edit_queued(2);

    app.drop_queued(0);

    assert_eq!(app.queue_edit, Some(1), "правка едет вместе со своей строкой");
}

#[test]
fn a_long_queue_still_fits_and_says_how_much_is_left() {
    let long: Vec<String> = (1..=9).map(|i| format!("сообщение {i}")).collect();
    let mut app = busy_with_queue(&[]);
    app.message_queue = long.into_iter().map(QueuedMessage::plain).collect();

    let rows = frame(&app, 100, 30);
    let head = row_with(&app, "Очередь сообщений (9)");

    assert!(rows[head + 1].contains("сообщение 1"));
    assert!(rows[head + 6].contains("сообщение 6"), "в кадр попало шесть строк");
    assert!(
        rows.iter().any(|r| r.contains("…и ещё 3")),
        "остаток назван числом:\n{}",
        rows.join("\n")
    );
}

// ── вложения ───────────────────────────────────────────────────────────────

fn tiny_grid(cols: usize, rows: usize) -> RgbGrid {
    let data: Vec<u8> = (0..cols * rows * 2 * 3).map(|i| (i % 256) as u8).collect();
    RgbGrid { cols, rows, data }
}

#[test]
fn a_pasted_picture_gets_a_preview_above_the_prompt() {
    let mut app = App::new();
    app.attachments.push(Attachment { label: "скриншот из буфера (10×10)".into(), thumb: tiny_grid(6, 2), full: None });

    let rows = frame(&app, 100, 30);
    let prompt = rows.iter().position(|r| r.contains('❯')).expect("поле ввода");

    // Полоса стоит прямо над вводом.
    assert!(prompt > 0);
    assert!(rows[prompt - 1].contains('▀') || rows[prompt - 2].contains('▀'), "нет картинки:\n{}", rows.join("\n"));
}

#[test]
fn the_cross_on_a_preview_removes_it_and_clicking_the_picture_opens_it() {
    let app = {
        let mut a = App::new();
        a.attachments.push(Attachment { label: "кадр".into(), thumb: tiny_grid(6, 2), full: None });
        a
    };
    let rows = frame(&app, 100, 30);

    // Ищем клетку ✕ на экране: она одна.
    let mut found = None;
    for (y, row) in rows.iter().enumerate() {
        if let Some(x) = row.chars().position(|c| c == '✕') {
            found = Some((x as u16, y as u16));
            break;
        }
    }
    let (x, y) = found.expect("на превью нет ✕");

    assert_eq!(app.get_clicked_action(x, y), Some(ClickAction::DropAttachment(0)));
    assert_eq!(
        app.get_clicked_action(x.saturating_sub(3), y),
        Some(ClickAction::ShowAttachment(0)),
        "клик по самой картинке открывает её"
    );
}

#[test]
fn pending_images_from_the_bridge_become_attachments() {
    // Тот вид, что отдаёт Python: подпись и сетка в base64.
    let raw = format!(
        r#"[{{"index":0,"label":"скриншот из буфера (1920×1080)","thumb":{{"cols":2,"rows":1,"rgb":"{}"}}}}]"#,
        b64_encode(&[1u8; 12])
    );

    let items = parse_attachments(&raw);

    assert_eq!(items.len(), 1);
    assert_eq!(items[0].label, "скриншот из буфера (1920×1080)");
    assert_eq!(items[0].thumb.cols, 2);
    assert_eq!(items[0].thumb.rows, 1);
}

#[test]
fn a_broken_payload_never_costs_the_message() {
    assert!(parse_attachments("").is_empty());
    assert!(parse_attachments("[]").is_empty());
    assert!(parse_attachments("не json").is_empty());
    // Сетка короче, чем обещает заголовок: мусор, а не паника.
    assert!(parse_attachments(r#"[{"label":"x","thumb":{"cols":4,"rows":4,"rgb":"AAAA"}}]"#).is_empty());
}

#[test]
fn two_queued_messages_do_not_hide_what_the_person_is_typing() {
    // The queue panel was drawn without reserving its rows, so with two messages
    // the prompt slid below the bottom edge: the keys went into the draft and the
    // draft was not on screen.
    let mut app = busy_with_queue(&["первое", "второе"]);
    app.input = "печатаю следующее сообщение".into();
    app.cursor_pos = app.input.chars().count();

    for height in [16u16, 20, 24, 30, 40] {
        let rows = frame(&app, 100, height);
        assert!(
            rows.iter().any(|r| r.contains("печатаю следующее")),
            "на высоте {height} очередь съела ввод:\n{}",
            rows.join("\n")
        );
    }
}

#[test]
fn the_draft_outlives_every_panel_that_could_crowd_it_out() {
    // The stack may not fit on a short terminal: the panels give way, the line being
    // typed does not. After the last frame the caret still has a row it sits on.
    let mut app = busy_with_queue(&["раз", "два", "три", "четыре"]);
    app.input = "важный черновик".into();
    app.cursor_pos = app.input.chars().count();
    app.status.todos = (0..6)
        .map(|i| argent_tui::app::TodoItem {
            content: format!("шаг {i}"),
            status: if i == 0 { "in_progress".into() } else { "pending".into() },
            since: None,
        })
        .collect();

    let rows = frame(&app, 100, 14);
    assert!(
        rows.iter().any(|r| r.contains("важный черновик")),
        "черновик вытеснен панелями:\n{}",
        rows.join("\n")
    );
}

#[test]
fn a_draft_survives_the_picture_being_attached() {
    // The strip used to be drawn without reserving its rows: the prompt was placed
    // below the visible area and the text vanished while the composer still held it.
    let mut app = App::new();
    app.attachments.push(Attachment {
        label: "скриншот".into(),
        thumb: tiny_grid(18, 6),
        full: Some(tiny_grid(120, 34)),
    });
    app.input = "черновик, который не должен исчезать".into();
    app.cursor_pos = app.input.chars().count();

    for height in [12u16, 18, 24, 40] {
        let rows = frame(&app, 100, height);
        assert!(
            rows.iter().any(|r| r.contains("черновик")),
            "на высоте {height} ввод пропал:\n{}",
            rows.join("\n")
        );
    }
}

#[test]
fn the_preview_is_drawn_from_the_wide_frame_not_a_thumb() {
    // An 18-column thumb blown up to fill the panel is what looked like a mosaic.
    // The strip draws the wide frame, scaled down — its width is now set by the
    // row cap, not by the 56-column figure that used to crop the picture.
    let mut app = App::new();
    app.attachments.push(Attachment {
        label: "кадр".into(),
        thumb: tiny_grid(18, 6),
        full: Some(tiny_grid(120, 34)),
    });

    let rows = frame(&app, 120, 40);
    let widest = rows.iter().map(|r| r.matches('▀').count()).max().unwrap_or(0);
    assert!(
        widest > 30,
        "превью шириной {widest} знаков ближе к миниатюре (18), чем к широкому кадру:\n{}",
        rows.join("\n")
    );
}

#[test]
fn the_strip_shows_the_whole_frame_instead_of_its_top_half() {
    // 16:9 at 320 columns is 90 rows of cells. A strip capped at 56 columns and 8
    // rows drew the top of the picture and called it a preview; the limit is now
    // on rows, so the whole frame fits however wide it is.
    let mut app = App::new();
    app.attachments.push(Attachment {
        label: "кадр".into(),
        thumb: tiny_grid(56, 32),
        full: Some(tiny_grid(320, 90)),
    });

    let rows = frame(&app, 120, 40);
    let image_rows: Vec<&String> = rows.iter().filter(|r| r.contains('▀')).collect();
    assert_eq!(
        image_rows.len(),
        10,
        "весь кадр — десять строк, а не обрезок:\n{}",
        rows.join("\n")
    );
    let width = image_rows[1].matches('▀').count();
    assert!(
        (30..=42).contains(&width),
        "16:9 при десяти строках — это ~35 знакомест, а не {width}"
    );
    assert_eq!(
        image_rows[0].matches('▀').count() + 1,
        width,
        "в первой строке одна клетка занята ✕, остальное — та же ширина"
    );
    for r in &image_rows {
        assert!(
            r.matches('▀').count().abs_diff(width) <= 1,
            "кадр обрезан по строке: {:?}",
            r
        );
    }
}

fn b64_encode(bytes: &[u8]) -> String {
    const T: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::new();
    for chunk in bytes.chunks(3) {
        let b = [chunk[0], *chunk.get(1).unwrap_or(&0), *chunk.get(2).unwrap_or(&0)];
        let n = ((b[0] as u32) << 16) | ((b[1] as u32) << 8) | b[2] as u32;
        out.push(T[(n >> 18) as usize & 63] as char);
        out.push(T[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 { T[(n >> 6) as usize & 63] as char } else { '=' });
        out.push(if chunk.len() > 2 { T[n as usize & 63] as char } else { '=' });
    }
    out
}
