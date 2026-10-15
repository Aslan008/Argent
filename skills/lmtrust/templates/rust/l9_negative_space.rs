// templates/rust/l9_negative_space.rs — Layer 9: Forbidden Side Effects & Negative Space
use super::*;

#[test]
fn test_l9_negative_space_no_unintended_mutation() {
    let original = vec![1, 2, 3];
    let clone = original.clone();
    let _ = inspect_collection(&original);

    // Negative space: inspection must NEVER mutate the underlying vector or order
    assert_eq!(original, clone, "Inspect operation illegally mutated collection state");
}
