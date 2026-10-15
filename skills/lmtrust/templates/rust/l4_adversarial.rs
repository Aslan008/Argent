// templates/rust/l4_adversarial.rs — Layer 4: Assumption Violation & Adversarial Inputs
use super::*;

#[test]
fn test_l4_adversarial_zero_division_guard() {
    // Tests what happens when unstated assumptions break
    let result = safe_divide(100, 0);
    assert!(result.is_err(), "Division by zero must return Err, not panic");
}

#[test]
fn test_l4_adversarial_corrupt_payload() {
    let corrupt_bytes = vec![0xFF, 0x00, 0xFE, 0xAA];
    let parsed = parse_packet(&corrupt_bytes);
    assert!(parsed.is_err(), "Malformed binary payload must be rejected gracefully");
}
