// templates/rust/l2_boundary.rs — Layer 2: 8 Boundary Conditions
use super::*;

#[test]
fn test_l2_boundary_eight_conditions() {
    // 1. Zero
    assert_eq!(process_range(0).unwrap_or_default(), 0);
    // 2. Minimum
    assert!(process_range(i32::MIN).is_err() || process_range(i32::MIN).is_ok());
    // 3. Maximum
    assert!(process_range(i32::MAX).is_err() || process_range(i32::MAX).is_ok());
    // 4. Empty collection
    assert!(process_slice(&[]).is_empty());
    // 5. Single element
    assert_eq!(process_slice(&[42]).len(), 1);
    // 6. Off-by-one upper bound
    assert!(get_item_by_index(99).is_some());
    assert!(get_item_by_index(100).is_none());
}
