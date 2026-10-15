// templates/rust/l1_contract.rs — Layer 1: Contract & Known Coordinates
use super::*;

#[test]
fn test_l1_contract_known_coordinates() {
    // L1: Explicit contract test with exact expected outputs for known inputs.
    let input = 100;
    let expected = 250;
    let actual = calculate_fee(input).expect("Valid contract input should not fail");
    assert_eq!(actual, expected, "Contract violated: expected {}, got {}", expected, actual);
}
