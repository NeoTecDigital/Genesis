//! Categorical object types: ∅, 𝟙, n

use std::fmt;
use std::hash::Hash;

/// Empty object ∅ - Initial object representing pure potential
///
/// This is a zero-sized type representing the categorical initial object.
/// All morphisms from Empty must factor through the genesis morphism γ.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Empty;

impl fmt::Display for Empty {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "∅")
    }
}

/// Unit object 𝟙 - Proto-unity, first actuality
///
/// This is a zero-sized type representing the categorical unit object.
/// The genesis morphism γ maps from Empty to Unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Unit;

impl fmt::Display for Unit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "𝟙")
    }
}

/// Numeric object n - Numeric instances
///
/// Wraps a u64 to represent specific numeric instances.
/// Instantiation morphisms ι_n map from Unit to Numeric(n).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Numeric(pub u64);

impl fmt::Display for Numeric {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Numeric {
    /// Create a new numeric instance
    pub fn new(n: u64) -> Self {
        Numeric(n)
    }

    /// Get the underlying value
    pub fn value(&self) -> u64 {
        self.0
    }
}

/// Type-level equality check for composition validation
pub trait Same<T> {
    fn is_same() -> bool;
}

// Reflexive: every type is same as itself
impl<T> Same<T> for T {
    fn is_same() -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_sized_types() {
        use std::mem::size_of;

        // Empty and Unit should be zero-sized
        assert_eq!(size_of::<Empty>(), 0);
        assert_eq!(size_of::<Unit>(), 0);

        // Numeric wraps a u64
        assert_eq!(size_of::<Numeric>(), size_of::<u64>());
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", Empty), "∅");
        assert_eq!(format!("{}", Unit), "𝟙");
        assert_eq!(format!("{}", Numeric(42)), "42");
    }

    #[test]
    fn test_same_trait() {
        // The Same trait is implemented reflexively for all types
        assert!(<Empty as Same<Empty>>::is_same());
        assert!(<Unit as Same<Unit>>::is_same());
        assert!(<Numeric as Same<Numeric>>::is_same());
    }
}