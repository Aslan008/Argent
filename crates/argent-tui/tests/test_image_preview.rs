//! Превью изображения: из чего терминал рисует картинку и как её открыть целиком.
//!
//! Сетку готовит Python (PIL), фронтенд только раскладывает её полублоками, поэтому
//! всё, что здесь проверяется, — разбор полезной нагрузки, цвета клеток и адресация
//! превью к своей карточке.

use std::collections::HashMap;

use argent_tui::app::{b64_decode, parse_image_preview, App, ClickAction, MessageItem};
use argent_tui::ui::draw_ui;
use ratatui::backend::TestBackend;
use ratatui::style::Color;
use ratatui::Terminal;

/// `"hello world"` в base64: 11 байт, хватает сетке 1×1 (нужно 6).
const RGB: &str = "aGVsbG8gd29ybGQ=";

fn payload(seq: u64) -> String {
    format!(
        r#"{{"seq":{seq},"path":"shot.png","caption":"Скриншот",
            "thumb":{{"cols":1,"rows":1,"rgb":"{RGB}"}},
            "full":{{"cols":1,"rows":1,"rgb":"{RGB}"}}}}"#
    )
}

/// Standard base64, the same twenty lines the crate uses: a grid payload needs
/// a few kilobytes of bytes and nothing else here decodes them.
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

fn grid(cols: usize, rows: usize) -> String {
    let data = vec![0x40u8; cols * rows * 2 * 3];
    format!(r#"{{"cols":{cols},"rows":{rows},"rgb":"{}"}}"#, b64_encode(&data))
}

fn image_tool_app() -> App {
    let mut app = App::new();
    // The shipped look: the drawn card lives in `cards.rs`, and the tests that
    // check what the person actually sees have to ask for it.
    app.ui.set_style(argent_tui::theme::UiStyle::Premium);
    app.messages.push(MessageItem::Tool {
        name: "view_image".into(),
        args: HashMap::new(),
        result: Some("ok".into()),
        duration_secs: None,
        expanded: false,
        is_running: false,
    });
    app
}

fn render(app: &App, w: u16, h: u16) -> Vec<String> {
    let mut terminal = Terminal::new(TestBackend::new(w, h)).expect("terminal");
    terminal.draw(|f| draw_ui(f, app)).expect("draw");
    let buf = terminal.backend().buffer().clone();
    (0..buf.area.height)
        .map(|y| {
            (0..buf.area.width)
                .map(|x| buf.content[(y * buf.area.width + x) as usize].symbol())
                .collect::<String>()
        })
        .collect()
}

#[test]
fn base64_is_decoded_without_a_dependency() {
    assert_eq!(b64_decode("aGVsbG8="), b"hello");
    assert_eq!(b64_decode(""), Vec::<u8>::new());
    // '=' и переводы строк — не данные.
    assert_eq!(b64_decode("aGVs\nbG8="), b"hello");
}

#[test]
fn the_payload_carries_both_grids() {
    let p = parse_image_preview(&payload(7)).expect("payload разобран");

    assert_eq!(p.seq, 7);
    assert_eq!(p.path, "shot.png");
    assert_eq!(p.caption, "Скриншот");
    assert_eq!((p.thumb.cols, p.thumb.rows), (1, 1));
    assert_eq!((p.full.cols, p.full.rows), (1, 1));
    // Первый пиксель — начало строки "hello".
    assert_eq!(p.thumb.pixel(0, 0), Color::Rgb(b'h', b'e', b'l'));
}

#[test]
fn a_broken_payload_is_ignored_instead_of_breaking_the_screen() {
    assert!(parse_image_preview("").is_none());
    assert!(parse_image_preview("не json").is_none());
    // Сетки нет вовсе.
    assert!(parse_image_preview(r#"{"seq":1,"path":"x"}"#).is_none());
    // Данных меньше, чем обещает размер сетки.
    assert!(parse_image_preview(&format!(
        r#"{{"seq":1,"thumb":{{"cols":9,"rows":9,"rgb":"{RGB}"}},"full":{{"cols":9,"rows":9,"rgb":"{RGB}"}}}}"#
    ))
    .is_none());
}

#[test]
fn a_cell_shows_two_pixels_one_above_the_other() {
    let p = parse_image_preview(&payload(1)).expect("payload");
    let lines = argent_tui::cards::image_lines(&p.thumb, 10);

    assert_eq!(lines.len(), 1, "одна строка клеток — два пикселя по высоте");
    let span = &lines[0].spans[0];
    assert_eq!(span.content.as_ref(), "▀", "верхний пиксель цветом, нижний фоном");
    assert_eq!(span.style.fg, Some(Color::Rgb(b'h', b'e', b'l')));
    assert_eq!(span.style.bg, Some(Color::Rgb(b'l', b'o', b' ')), "нижняя половина клетки");
}

#[test]
fn a_grid_wider_than_the_room_is_clipped_not_wrapped() {
    let p = parse_image_preview(&payload(1)).expect("payload");
    let lines = argent_tui::cards::image_lines(&p.full, 0);
    assert!(lines[0].spans.is_empty(), "нулевая ширина — пустая строка, а не паника");

    let lines = argent_tui::cards::image_lines(&p.full, 1);
    assert_eq!(lines[0].spans.len(), 1);
}

#[test]
fn a_fresh_picture_lands_on_the_image_tool_card() {
    let mut app = App::new();
    app.messages.push(MessageItem::User("посмотри скриншот".into()));
    app.messages.push(MessageItem::Tool {
        name: "view_image".into(),
        args: HashMap::new(),
        result: Some("ok".into()),
        duration_secs: Some(0.2),
        expanded: false,
        is_running: false,
    });

    app.update_status(HashMap::from([("image_preview".to_string(), payload(1))]));

    assert_eq!(app.previews.len(), 1, "превью привязано к карточке инструмента");
    assert!(app.previews.contains_key(&1), "индекс карточки — второй элемент");
}

#[test]
fn the_same_payload_is_not_attached_twice() {
    let mut app = App::new();
    app.messages.push(MessageItem::Tool {
        name: "capture_desktop_window".into(),
        args: HashMap::new(),
        result: Some("ok".into()),
        duration_secs: None,
        expanded: false,
        is_running: false,
    });

    app.update_status(HashMap::from([("image_preview".to_string(), payload(1))]));
    app.update_status(HashMap::from([("image_preview".to_string(), payload(1))]));
    assert_eq!(app.previews[&0].seq, 1);

    app.update_status(HashMap::from([("image_preview".to_string(), payload(2))]));
    assert_eq!(app.previews[&0].seq, 2, "новый кадр заменяет старый");
}

#[test]
fn a_click_on_the_thumbnail_opens_the_whole_screen() {
    let mut app = App::new();
    app.messages.push(MessageItem::Tool {
        name: "view_image".into(),
        args: HashMap::new(),
        result: Some("ok".into()),
        duration_secs: None,
        expanded: false,
        is_running: false,
    });
    app.update_status(HashMap::from([("image_preview".to_string(), payload(1))]));
    assert_eq!(ClickAction::ShowImage(0).hint(), Some("Клик — открыть изображение на весь экран · Esc закроет"));

    // The overlay replaces the screen: the prompt has no business showing through.
    app.image_view = app.previews.get(&0).cloned();
    let screen = render(&app, 60, 12);
    assert!(screen.iter().any(|r| r.contains("Esc — закрыть")), "есть путь назад: {screen:?}");
    assert!(
        screen.iter().any(|r| r.contains("открыть в системе")),
        "терминал не покажет скриншот читаемо — нужен путь к оригиналу: {screen:?}"
    );
    assert!(screen.iter().any(|r| r.contains('▀')), "картинка видна: {screen:?}");
    assert!(!screen.iter().any(|r| r.contains('❯')), "промпт перекрыт");
}

#[test]
fn the_card_preview_is_drawn_from_the_wide_frame_not_the_thumb() {
    // A 4-column thumb spread over a card was the mosaic: the folded card must
    // draw the wide frame, scaled down, never the tiny grid.
    let mut app = image_tool_app();
    let raw = format!(
        r#"{{"seq":1,"path":"shot.png","caption":"Скриншот","thumb":{},"full":{}}}"#,
        grid(4, 2),
        grid(50, 16)
    );
    app.update_status(HashMap::from([("image_preview".to_string(), raw)]));
    assert_eq!(app.previews.len(), 1, "кадр не попал в карту: {:?}", app.previews.keys());

    let screen = render(&app, 120, 40);
    let widest = screen.iter().map(|r| r.matches('▀').count()).max().unwrap_or(0);
    assert!(
        widest >= 40,
        "превью шириной {widest} знаков — та самая мозаика из миниатюры: {screen:?}"
    );
}

#[test]
fn a_wider_card_draws_a_wider_picture() {
    // The point of scaling: the picture follows the room it is given instead of
    // stopping at whatever width Python happened to send.
    let mut app = image_tool_app();
    let raw = format!(
        r#"{{"seq":1,"path":"shot.png","caption":"","thumb":{},"full":{}}}"#,
        grid(4, 2),
        grid(50, 16)
    );
    app.update_status(HashMap::from([("image_preview".to_string(), raw)]));
    app.toggle_block_by_idx(0); // раскрытая карточка — на всю ширину

    let narrow = render(&app, 90, 40).iter().map(|r| r.matches('▀').count()).max().unwrap_or(0);
    let wide = render(&app, 200, 40).iter().map(|r| r.matches('▀').count()).max().unwrap_or(0);
    assert!(wide > narrow, "ширина {narrow} → {wide}: превью не растёт вместе с карточкой");
}
