//! The look-and-feel settings the TUI's NATIVE widgets need.
//!
//! Static blocks (answers, tool results, diffs, ...) are rendered by the Python
//! side with the user's theme already applied. What remains in Rust — the prompt,
//! the live spinner lines, the queue box — has to follow the same `theme.yaml`,
//! so the bridge hands the colour table over (`get_ui_config`) and it is parsed
//! here instead of being hard-coded.

use ratatui::style::{Color, Modifier, Style};
use serde_json::Value;

use crate::ansi::style_from_rich;

/// Colours of the native widgets. Defaults equal the shipped `theme.yaml`, so a
/// bridge that cannot answer still gives the standard look.
#[derive(Debug, Clone)]
pub struct Theme {
    pub user_prompt: Style,
    pub user_prompt_symbol: Style,
    pub assistant_header: Style,
    pub assistant_text: Style,
    pub system_msg: Style,
    pub error_msg: Style,
    pub reasoning_label: Style,
    pub reasoning_text: Style,
}

impl Default for Theme {
    fn default() -> Self {
        Self {
            user_prompt: style_from_rich("#ffffff"),
            user_prompt_symbol: style_from_rich("#58a6ff"),
            assistant_header: style_from_rich("bold #58a6ff"),
            assistant_text: Style::default(),
            system_msg: style_from_rich("dim #8b949e"),
            error_msg: style_from_rich("bold #f85149"),
            reasoning_label: style_from_rich("bold #8b949e"),
            reasoning_text: style_from_rich("dim #8b949e"),
        }
    }
}

/// How the interface looks.
///
/// `Classic` is the terminal REPL's look, cell for cell: the parity tests hold the
/// TUI to it. `Premium` keeps the same layout and keys but dresses it: tinted
/// panels, an accent bar on what you wrote, chips in the toolbar, a highlighted
/// suggestion list, and headings / quotes / code in answers drawn as designed
/// blocks. It costs nothing at run time — the colours are fixed RGB values and
/// rows are cached like any other.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum UiStyle {
    #[default]
    Classic,
    Premium,
}

impl UiStyle {
    pub fn parse(name: &str) -> Option<Self> {
        match name.trim().to_lowercase().as_str() {
            "classic" | "plain" | "repl" => Some(Self::Classic),
            "premium" | "rich" => Some(Self::Premium),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Classic => "classic",
            Self::Premium => "premium",
        }
    }
}

/// The premium palette (GitHub-dark, like the shipped `theme.yaml`).
///
/// A terminal has no alpha channel, so "translucent" panels are colours mixed
/// ahead of time: `surface` is a hair above `base`, `accent_soft` is the accent at
/// about a fifth of its strength over `surface`. Only panels get a background; the
/// page itself keeps the terminal's own, so a translucent or blurred terminal
/// window shows through everywhere else.
#[derive(Debug, Clone)]
pub struct Palette {
    /// What the terminal background is assumed to be (the tints are mixed into it).
    pub base: Color,
    /// A raised panel: the prompt, tool cards, the suggestion list.
    pub surface: Color,
    /// A panel one step higher: the pointed-at item, chips.
    pub surface_hi: Color,
    pub accent: Color,
    /// The accent, dimmed into `surface`: selection and active-item backgrounds.
    pub accent_soft: Color,
    /// Background of selected text.
    pub selection: Color,
    pub border: Color,
    pub text: Color,
    pub muted: Color,
    pub ok: Color,
    pub warn: Color,
    pub danger: Color,
    /// Your message bubble: the accent a little over the page.
    pub user_bg: Color,
    /// A tool call's rail (running and finished alike); its result says ok or failed.
    pub tool: Color,
    /// The model's reasoning, drawn quietly, and its second-pass check.
    pub think: Color,
    pub rethink: Color,
    /// The body of a card (a tool's output, a diff's context): darker than `surface`.
    pub code: Color,
    /// Tints of the status colours for a whole row: a failed call, an added line.
    pub ok_soft: Color,
    pub danger_soft: Color,
    pub warn_soft: Color,
}

impl Default for Palette {
    fn default() -> Self {
        let base = Color::Rgb(0x0d, 0x11, 0x17);
        let surface = Color::Rgb(0x16, 0x1b, 0x22);
        let ok = Color::Rgb(0x3f, 0xb9, 0x50);
        let warn = Color::Rgb(0xd2, 0x99, 0x22);
        let danger = Color::Rgb(0xf8, 0x51, 0x49);
        let mut p = Self {
            base,
            surface,
            surface_hi: Color::Rgb(0x21, 0x26, 0x2d),
            accent: Color::Rgb(0x58, 0xa6, 0xff),
            accent_soft: base,
            selection: base,
            border: Color::Rgb(0x30, 0x36, 0x3d),
            text: Color::Rgb(0xe6, 0xed, 0xf3),
            muted: Color::Rgb(0x8b, 0x94, 0x9e),
            ok,
            warn,
            danger,
            user_bg: base,
            // One accent (blue) for you and the answer; the rest of the roles each get a quiet hue of their
            // own, so a call, a thought and a system line differ by colour and not by how grey they are.
            tool: Color::Rgb(0x3f, 0xa8, 0x9c),
            think: Color::Rgb(0x9b, 0x8c, 0xd0),
            rethink: Color::Rgb(0xd1, 0x8f, 0x52),
            code: Color::Rgb(0x11, 0x15, 0x1c),
            ok_soft: blend(ok, surface, 0.16),
            danger_soft: blend(danger, surface, 0.16),
            warn_soft: blend(warn, surface, 0.16),
        };
        p.set_accent(p.accent);
        p
    }
}

impl Palette {
    /// Take `accent` as the accent and re-mix everything that is derived from it.
    pub fn set_accent(&mut self, accent: Color) {
        self.accent = accent;
        self.accent_soft = blend(accent, self.surface, 0.22);
        self.selection = blend(accent, self.base, 0.42);
        self.user_bg = blend(accent, self.base, 0.20);
    }
}

fn rgb_of(c: Color) -> Option<(u8, u8, u8)> {
    match c {
        Color::Rgb(r, g, b) => Some((r, g, b)),
        _ => None,
    }
}

/// `fg` laid over `bg` at `alpha` (0 = all `bg`, 1 = all `fg`). Colours that are
/// not RGB cannot be mixed and come back as `fg`.
pub fn blend(fg: Color, bg: Color, alpha: f32) -> Color {
    let (Some((fr, fg_, fb)), Some((br, bg_, bb))) = (rgb_of(fg), rgb_of(bg)) else {
        return fg;
    };
    let a = alpha.clamp(0.0, 1.0);
    let mix = |f: u8, b: u8| ((f as f32) * a + (b as f32) * (1.0 - a)).round() as u8;
    Color::Rgb(mix(fr, br), mix(fg_, bg_), mix(fb, bb))
}

/// Settings fetched from the bridge once at start-up.
#[derive(Debug, Clone)]
pub struct UiConfig {
    pub theme: Theme,
    /// `"summary"` (collapsed reasoning line) or `"full"`.
    pub thinking_mode: String,
    pub show_tool_results: bool,
    pub style: UiStyle,
    pub palette: Palette,
}

impl Default for UiConfig {
    fn default() -> Self {
        Self {
            theme: Theme::default(),
            thinking_mode: "summary".to_string(),
            show_tool_results: false,
            style: UiStyle::Classic,
            palette: Palette::default(),
        }
    }
}

impl UiConfig {
    pub fn is_premium(&self) -> bool {
        self.style == UiStyle::Premium
    }

    /// Switch the look; returns true when it changed.
    pub fn set_style(&mut self, style: UiStyle) -> bool {
        let changed = self.style != style;
        self.style = style;
        changed
    }
}

impl UiConfig {
    /// Parse the JSON produced by `src.bridge.render.ui_config`. Anything missing
    /// or malformed keeps its default — a broken theme must not break the UI.
    pub fn from_json(raw: &str) -> Self {
        let mut cfg = Self::default();
        let Ok(v) = serde_json::from_str::<Value>(raw) else {
            return cfg;
        };
        if let Some(colors) = v.get("colors").and_then(Value::as_object) {
            let pick = |key: &str, slot: &mut Style| {
                if let Some(s) = colors.get(key).and_then(Value::as_str) {
                    *slot = style_from_rich(s);
                }
            };
            pick("user_prompt", &mut cfg.theme.user_prompt);
            pick("user_prompt_symbol", &mut cfg.theme.user_prompt_symbol);
            pick("assistant_header", &mut cfg.theme.assistant_header);
            pick("assistant_text", &mut cfg.theme.assistant_text);
            pick("system_msg", &mut cfg.theme.system_msg);
            pick("error_msg", &mut cfg.theme.error_msg);
            pick("reasoning_label", &mut cfg.theme.reasoning_label);
            pick("reasoning_text", &mut cfg.theme.reasoning_text);
        }
        if let Some(m) = v.get("thinking_mode").and_then(Value::as_str) {
            if m == "summary" || m == "full" {
                cfg.thinking_mode = m.to_string();
            }
        }
        if let Some(b) = v.get("show_tool_results").and_then(Value::as_bool) {
            cfg.show_tool_results = b;
        }
        if let Some(style) = v.get("style").and_then(Value::as_str).and_then(UiStyle::parse) {
            cfg.style = style;
        }
        // The accent follows the theme's own prompt colour, so a user theme still
        // recolours the premium look.
        if let Some(Color::Rgb(r, g, b)) = cfg.theme.user_prompt_symbol.fg {
            cfg.palette.set_accent(Color::Rgb(r, g, b));
        }
        cfg
    }
}

/// prompt_toolkit's default `bottom-toolbar` style is `reverse`; the REPL's
/// style sheet sets the default foreground to `#ffffff`, so the bar is a
/// reversed white-on-default strip.
pub fn toolbar_style() -> Style {
    Style::default().fg(Color::Rgb(255, 255, 255)).add_modifier(Modifier::REVERSED)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_the_shipped_theme() {
        let t = Theme::default();
        assert_eq!(t.user_prompt_symbol.fg, Some(Color::Rgb(0x58, 0xa6, 0xff)));
        assert!(t.assistant_header.add_modifier.contains(Modifier::BOLD));
        assert!(t.system_msg.add_modifier.contains(Modifier::DIM));
    }

    #[test]
    fn json_overrides_only_what_it_names() {
        let c = UiConfig::from_json(
            r##"{"colors":{"user_prompt_symbol":"#ff0000"},"thinking_mode":"full","show_tool_results":true}"##,
        );
        assert_eq!(c.theme.user_prompt_symbol.fg, Some(Color::Rgb(255, 0, 0)));
        assert_eq!(c.theme.error_msg, Theme::default().error_msg);
        assert_eq!(c.thinking_mode, "full");
        assert!(c.show_tool_results);
    }

    #[test]
    fn the_default_look_is_the_repls() {
        assert_eq!(UiConfig::default().style, UiStyle::Classic);
        assert_eq!(UiConfig::from_json("{}").style, UiStyle::Classic, "no setting: classic");
    }

    #[test]
    fn the_style_comes_from_the_bridge() {
        assert_eq!(UiConfig::from_json(r#"{"style":"premium"}"#).style, UiStyle::Premium);
        assert_eq!(UiConfig::from_json(r#"{"style":"Classic"}"#).style, UiStyle::Classic);
        assert_eq!(UiConfig::from_json(r#"{"style":"nonsense"}"#).style, UiStyle::Classic);
        assert_eq!(UiStyle::parse(" PREMIUM "), Some(UiStyle::Premium));
    }

    #[test]
    fn blending_mixes_toward_the_background() {
        let white = Color::Rgb(255, 255, 255);
        let black = Color::Rgb(0, 0, 0);
        assert_eq!(blend(white, black, 0.0), black);
        assert_eq!(blend(white, black, 1.0), white);
        assert_eq!(blend(white, black, 0.5), Color::Rgb(128, 128, 128));
        assert_eq!(blend(Color::Indexed(4), black, 0.5), Color::Indexed(4), "non-RGB colours pass through");
    }

    #[test]
    fn the_soft_accent_sits_between_the_panel_and_the_accent() {
        let p = Palette::default();
        let (Color::Rgb(sr, _, sb), Color::Rgb(ar, _, ab), Color::Rgb(pr, _, pb)) =
            (p.surface, p.accent, p.accent_soft)
        else {
            panic!("the palette is RGB");
        };
        assert!(pb > sb && pb < ab, "bluer than the panel, dimmer than the accent");
        assert!(pr >= sr.min(ar) && pr <= sr.max(ar));
    }

    #[test]
    fn a_theme_recolours_the_accent() {
        let c = UiConfig::from_json(r##"{"style":"premium","colors":{"user_prompt_symbol":"#ff8800"}}"##);
        assert_eq!(c.palette.accent, Color::Rgb(0xff, 0x88, 0x00));
        assert_ne!(c.palette.accent_soft, Palette::default().accent_soft);
    }

    #[test]
    fn garbage_keeps_the_defaults() {
        let c = UiConfig::from_json("not json");
        assert_eq!(c.thinking_mode, "summary");
        let c = UiConfig::from_json(r#"{"thinking_mode":"bogus"}"#);
        assert_eq!(c.thinking_mode, "summary");
    }
}
