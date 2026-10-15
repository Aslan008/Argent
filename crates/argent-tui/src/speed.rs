//! How fast the model is generating, measured from the stream as it arrives.
//!
//! A provider reports its exact token count only once a request is over, so while
//! an answer is still streaming the speed is built from what has been received:
//! every piece of text is worth an estimated number of tokens, and the speed is
//! those tokens over a short sliding window of arrival times. When a request ends,
//! the provider's exact `completion` count corrects the estimate (tokens per
//! character differ by language, by code against prose, by tokenizer), so from the
//! second request of a session the figure follows the model's real tokenizer. Until
//! the first correction it is marked as approximate.
//!
//! Time is the moment a piece reached the event sink (on the Python worker
//! thread), not the moment the UI got round to it: a slow frame must not bunch the
//! stream up and then spread it out again.
//!
//! The first piece of a stream only says WHEN generation began (the wait before it
//! is the prompt being read, not the model writing), so its tokens are not counted:
//! the speed is that of decoding, the figure inference servers quote.

use std::cell::Cell;
use std::collections::VecDeque;
use std::time::{Duration, Instant};

use unicode_width::UnicodeWidthChar;

/// How far back the speed looks: long enough to smooth out chunky delivery, short
/// enough to follow a model that slows down or speeds up.
const WINDOW: Duration = Duration::from_secs(3);
/// No piece for this long: the stream is over or stuck, so there is no speed to show.
/// It is also what separates one request from the next when no token count ever came.
const STALL: Duration = Duration::from_millis(2500);
/// Less than this much stream, or fewer tokens, is too little to divide.
const MIN_SPAN: Duration = Duration::from_millis(500);
const MIN_TOKENS: f64 = 3.0;
/// A figure stays on screen at least this long, so the digits do not flicker with
/// every frame.
const HOLD: Duration = Duration::from_millis(250);
/// A request whose exact count is further than this from the estimate is not one the
/// estimate can be corrected by (a model that thinks without streaming it, a
/// provider that reports a running total): it is left out rather than believed.
const TRUSTED_RATIO: (f64, f64) = (0.4, 2.5);
const SCALE_RANGE: (f64, f64) = (0.5, 2.0);

/// What a piece of text is worth in tokens before the real count is known: about a
/// quarter token per ASCII character (prose, code and JSON average 3 to 4 characters
/// a token), a token or so per wide character (CJK, emoji), and two to three
/// characters a token for everything else (Cyrillic, accented Latin, symbols).
pub fn estimate_tokens(text: &str) -> f64 {
    text.chars()
        .map(|c| {
            if c.is_ascii() {
                0.27
            } else {
                match c.width() {
                    Some(0) => 0.0,
                    Some(w) if w >= 2 => 0.9,
                    _ => 0.42,
                }
            }
        })
        .sum()
}

/// A measured speed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Reading {
    pub tokens_per_s: f64,
    /// True until a provider's exact token count has corrected the estimate.
    pub approximate: bool,
}

impl Reading {
    /// `48 tok/s`, `3.4 tok/s`, `1.2k tok/s`; a `~` in front while it is an estimate.
    pub fn label(&self) -> String {
        let n = self.tokens_per_s;
        let num = if n >= 1000.0 {
            format!("{:.1}k", n / 1000.0)
        } else if n >= 9.95 {
            format!("{n:.0}")
        } else {
            format!("{n:.1}")
        };
        format!("{}{num} tok/s", if self.approximate { "~" } else { "" })
    }
}

#[derive(Debug, Clone)]
pub struct SpeedMeter {
    /// (arrival, estimated tokens received up to and including that piece), oldest first.
    window: VecDeque<(Instant, f64)>,
    /// Estimated tokens received over the meter's life; the window holds slices of it.
    total: f64,
    /// Estimated tokens of the request in flight: what its exact count is compared with.
    request: f64,
    /// Real tokens per estimated token, as last learned from a provider's count.
    scale: f64,
    calibrated: bool,
    /// The last figure handed out and when, so frames in quick succession agree.
    shown: Cell<Option<(Instant, Option<Reading>)>>,
}

impl Default for SpeedMeter {
    fn default() -> Self {
        Self::new()
    }
}

impl SpeedMeter {
    pub fn new() -> Self {
        Self {
            window: VecDeque::new(),
            total: 0.0,
            request: 0.0,
            scale: 1.0,
            calibrated: false,
            shown: Cell::new(None),
        }
    }

    /// A piece of generated text (answer, reasoning or tool arguments) arrived at `at`.
    pub fn feed(&mut self, text: &str, at: Instant) {
        if text.is_empty() {
            return;
        }
        // A long silence means a new stream (a tool ran in between): do not average
        // the pause into it.
        if self.window.back().is_some_and(|(t, _)| at.saturating_duration_since(*t) > STALL) {
            self.window.clear();
        }
        let tokens = estimate_tokens(text);
        self.request += tokens;
        self.total += tokens;
        self.window.push_back((at, self.total));
        while self.window.len() > 1
            && self.window.front().is_some_and(|(t, _)| at.saturating_duration_since(*t) > WINDOW)
        {
            self.window.pop_front();
        }
    }

    /// The request ended and the provider counted `completion` tokens in it.
    pub fn settle(&mut self, completion: usize) {
        if completion > 0 && self.request > 0.0 {
            let ratio = completion as f64 / self.request;
            if ratio >= TRUSTED_RATIO.0 && ratio <= TRUSTED_RATIO.1 {
                let next = if self.calibrated { 0.6 * ratio + 0.4 * self.scale } else { ratio };
                self.scale = next.clamp(SCALE_RANGE.0, SCALE_RANGE.1);
                self.calibrated = true;
            }
        }
        self.request = 0.0;
    }

    /// The request ended with no count (a tool is about to run, the turn is over).
    pub fn end_request(&mut self) {
        self.request = 0.0;
    }

    /// The model or provider changed: the old tokenizer's correction no longer holds.
    pub fn forget_calibration(&mut self) {
        self.scale = 1.0;
        self.calibrated = false;
        self.shown.set(None);
    }

    pub fn is_calibrated(&self) -> bool {
        self.calibrated
    }

    /// The speed right now, or `None` while there is nothing to say: before enough
    /// has streamed to divide, and once the stream has gone quiet.
    pub fn reading(&self, now: Instant) -> Option<Reading> {
        if let Some((at, reading)) = self.shown.get() {
            if now >= at && now.duration_since(at) < HOLD {
                return reading;
            }
        }
        let reading = self.measure(now);
        self.shown.set(Some((now, reading)));
        reading
    }

    fn measure(&self, now: Instant) -> Option<Reading> {
        let (last_at, last_total) = *self.window.back()?;
        if now.saturating_duration_since(last_at) > STALL {
            return None;
        }
        let (base_at, base_total) = *self.window.front()?;
        // Up to `now`, not to the last piece: a stream that stops then fades out
        // instead of freezing on its last good number.
        let span = now.saturating_duration_since(base_at).max(last_at.saturating_duration_since(base_at));
        if span < MIN_SPAN {
            return None;
        }
        let tokens = (last_total - base_total) * self.scale;
        if tokens < MIN_TOKENS {
            return None;
        }
        Some(Reading { tokens_per_s: tokens / span.as_secs_f64(), approximate: !self.calibrated })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clock() -> impl Fn(u64) -> Instant {
        let t0 = Instant::now();
        move |ms| t0 + Duration::from_millis(ms)
    }

    /// `n` pieces of four ASCII characters (1.08 estimated tokens each), `every_ms`
    /// apart, the first at `from_ms`. Returns the time of the last one.
    fn stream(m: &mut SpeedMeter, at: &impl Fn(u64) -> Instant, from_ms: u64, n: u64, every_ms: u64) -> u64 {
        for i in 0..n {
            m.feed("abcd", at(from_ms + i * every_ms));
        }
        from_ms + (n - 1) * every_ms
    }

    #[test]
    fn a_character_is_worth_less_in_prose_than_in_a_script_that_needs_more_tokens() {
        let ascii = estimate_tokens(&"a".repeat(100));
        let cyrillic = estimate_tokens(&"я".repeat(100));
        let cjk = estimate_tokens(&"日".repeat(100));
        assert!(ascii < cyrillic && cyrillic < cjk, "{ascii} {cyrillic} {cjk}");
        assert!((ascii - 27.0).abs() < 1e-6);
        assert_eq!(estimate_tokens(""), 0.0);
        assert_eq!(estimate_tokens("\u{301}"), 0.0, "a combining mark adds no token of its own");
    }

    #[test]
    fn a_steady_stream_reads_as_its_real_rate() {
        let at = clock();
        let mut m = SpeedMeter::new();
        // 20 pieces a second of 1.08 tokens: 21.6 tok/s
        let end = stream(&mut m, &at, 0, 100, 50);
        let r = m.reading(at(end)).expect("a reading after five seconds of stream");
        assert!((r.tokens_per_s - 21.6).abs() < 0.5, "{r:?}");
        assert!(r.approximate, "nothing has corrected it yet");
    }

    #[test]
    fn there_is_no_speed_until_there_is_something_to_divide() {
        let at = clock();
        let mut m = SpeedMeter::new();
        assert_eq!(m.reading(at(0)), None, "nothing streamed");
        m.feed("abcd", at(0));
        assert_eq!(m.reading(at(100)), None, "one piece only marks the start");
        let mut m = SpeedMeter::new();
        stream(&mut m, &at, 0, 6, 50);
        assert_eq!(m.reading(at(250)), None, "a quarter of a second is too little");
        let mut m = SpeedMeter::new();
        m.feed("a", at(0));
        m.feed("b", at(1000));
        assert_eq!(m.reading(at(1000)), None, "two characters are not three tokens");
    }

    #[test]
    fn the_wait_for_the_first_token_is_not_generation_time() {
        let at = clock();
        let mut m = SpeedMeter::new();
        // Nothing for 20 s while the prompt is read; the stream then runs at 21.6 tok/s.
        let end = stream(&mut m, &at, 20_000, 60, 50);
        let r = m.reading(at(end)).unwrap();
        assert!((r.tokens_per_s - 21.6).abs() < 0.8, "{r:?}");
    }

    #[test]
    fn a_stream_that_goes_quiet_fades_and_then_disappears() {
        let at = clock();
        let mut m = SpeedMeter::new();
        let end = stream(&mut m, &at, 0, 80, 50);
        let live = m.reading(at(end)).unwrap().tokens_per_s;
        let fading = m.reading(at(end + 1500)).unwrap().tokens_per_s;
        assert!(fading < live * 0.8, "{fading} should be well under {live}");
        assert_eq!(m.reading(at(end + 3000)), None, "three silent seconds: nothing is being generated");
    }

    #[test]
    fn the_pause_between_two_requests_does_not_drag_the_next_one_down() {
        let at = clock();
        let mut m = SpeedMeter::new();
        let first_end = stream(&mut m, &at, 0, 100, 25); // 43 tok/s
        // a tool ran for 8 s, then a slow stream: 4 pieces a second, 4.3 tok/s
        let end = stream(&mut m, &at, first_end + 8000, 12, 250);
        let r = m.reading(at(end)).unwrap();
        assert!((r.tokens_per_s - 4.3).abs() < 0.6, "{r:?}");
    }

    #[test]
    fn only_the_last_few_seconds_count() {
        let at = clock();
        let mut m = SpeedMeter::new();
        let slow_end = stream(&mut m, &at, 0, 50, 100); // 10.8 tok/s for five seconds
        let end = stream(&mut m, &at, slow_end + 50, 120, 25); // then 43 tok/s
        let r = m.reading(at(end)).unwrap();
        assert!((r.tokens_per_s - 43.2).abs() < 3.0, "{r:?}");
    }

    #[test]
    fn a_providers_exact_count_corrects_the_next_estimate() {
        let at = clock();
        let mut m = SpeedMeter::new();
        let end = stream(&mut m, &at, 0, 100, 50); // estimated 108 tokens
        let before = m.reading(at(end)).unwrap();
        assert!(before.approximate);
        m.settle(162); // the real tokenizer found one and a half times as many
        assert!(m.is_calibrated());
        // the next request, same shape, a while later
        let later = end + 10_000;
        let end2 = stream(&mut m, &at, later, 100, 50);
        let after = m.reading(at(end2)).unwrap();
        assert!(!after.approximate);
        assert!((after.tokens_per_s / before.tokens_per_s - 1.5).abs() < 0.05, "{before:?} -> {after:?}");
    }

    #[test]
    fn a_count_far_from_the_estimate_is_not_believed() {
        let at = clock();
        let mut m = SpeedMeter::new();
        stream(&mut m, &at, 0, 100, 50);
        m.settle(5000); // a model that reasons without streaming it
        assert!(!m.is_calibrated());
        let mut m = SpeedMeter::new();
        stream(&mut m, &at, 0, 100, 50);
        m.settle(0);
        assert!(!m.is_calibrated(), "no count, nothing to learn from");
    }

    #[test]
    fn corrections_are_averaged_so_one_odd_request_does_not_swing_the_figure() {
        let at = clock();
        let mut m = SpeedMeter::new();
        stream(&mut m, &at, 0, 100, 50);
        m.settle(216); // the first correction is taken as it comes: exactly 2.0 times the estimate
        assert!((m.scale - 2.0).abs() < 1e-9);
        let end = stream(&mut m, &at, 20_000, 100, 50);
        m.settle(108); // a request the estimate got right pulls it back, but only part of the way
        assert!((m.scale - 1.4).abs() < 1e-9, "0.6 * 1.0 + 0.4 * 2.0, got {}", m.scale);
        stream(&mut m, &at, end + 10_000, 100, 50);
        m.settle(108);
        assert!((m.scale - 1.16).abs() < 1e-9, "got {}", m.scale);
    }

    #[test]
    fn a_correction_stays_within_bounds() {
        let at = clock();
        let mut m = SpeedMeter::new();
        stream(&mut m, &at, 0, 100, 50);
        m.settle(259); // 2.4 times: trusted, but never applied past the limit
        assert_eq!(m.scale, SCALE_RANGE.1);
        let mut m = SpeedMeter::new();
        stream(&mut m, &at, 0, 100, 50);
        m.settle(45); // 0.42 times
        assert_eq!(m.scale, SCALE_RANGE.0);
    }

    #[test]
    fn the_correction_is_dropped_when_the_model_changes() {
        let at = clock();
        let mut m = SpeedMeter::new();
        stream(&mut m, &at, 0, 100, 50);
        m.settle(162);
        assert!(m.is_calibrated());
        m.forget_calibration();
        assert!(!m.is_calibrated());
        assert_eq!(m.scale, 1.0);
    }

    #[test]
    fn a_request_without_a_count_is_forgotten_so_it_cannot_skew_the_next_one() {
        let at = clock();
        let mut m = SpeedMeter::new();
        stream(&mut m, &at, 0, 100, 50); // 108 estimated tokens, never counted
        m.end_request();
        stream(&mut m, &at, 30_000, 100, 50);
        m.settle(108); // the second request alone is 108 tokens, ratio 1.0
        assert!((m.scale - 1.0).abs() < 1e-9, "scale {} (the first request leaked into the second)", m.scale);
    }

    #[test]
    fn frames_in_quick_succession_show_the_same_figure() {
        let at = clock();
        let mut m = SpeedMeter::new();
        let end = stream(&mut m, &at, 0, 100, 50);
        let first = m.reading(at(end)).unwrap();
        m.feed("abcdabcdabcdabcdabcdabcd", at(end + 40));
        assert_eq!(m.reading(at(end + 100)), Some(first), "held for a moment");
        let later = m.reading(at(end + 400)).unwrap();
        assert_ne!(later.tokens_per_s, first.tokens_per_s, "then it moves");
    }

    #[test]
    fn labels_read_naturally_at_every_scale() {
        let l = |tokens_per_s, approximate| Reading { tokens_per_s, approximate }.label();
        assert_eq!(l(48.2, false), "48 tok/s");
        assert_eq!(l(48.2, true), "~48 tok/s");
        assert_eq!(l(3.44, false), "3.4 tok/s");
        assert_eq!(l(9.97, false), "10 tok/s", "no `10.0` next to `9.9`");
        assert_eq!(l(1234.0, false), "1.2k tok/s");
    }
}
