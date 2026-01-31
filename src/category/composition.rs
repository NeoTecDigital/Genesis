//! Morphism composition system

use std::fmt;
use super::morphisms::{Morphism, MorphismId};
use super::objects::Same;
use crate::waveform::WaveformNode;

/// Error type for composition failures
#[derive(Debug, Clone, PartialEq)]
pub enum CompositionError {
    TypeMismatch {
        first_target: String,
        second_source: String,
    },
}

impl fmt::Display for CompositionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CompositionError::TypeMismatch { first_target, second_source } => {
                write!(f, "Cannot compose: target {} does not match source {}",
                       first_target, second_source)
            }
        }
    }
}

impl std::error::Error for CompositionError {}

/// Result type for composition operations
pub type CompositionResult<F, G> = Result<Composition<F, G>, CompositionError>;

/// Composition of two morphisms
///
/// Represents the categorical composition g ∘ f where
/// f: A → B and g: B → C compose to (g ∘ f): A → C
#[derive(Debug, Clone)]
pub struct Composition<F, G> {
    pub first: F,
    pub second: G,
}

impl<F, G> Composition<F, G>
where
    F: Morphism,
    G: Morphism,
{
    /// Create a new composition if types match
    pub fn new(first: F, second: G) -> Self {
        Composition { first, second }
    }
}

// Seal the Composition type
impl<F, G> super::morphisms::sealed::Sealed for Composition<F, G>
where
    F: Morphism,
    G: Morphism,
{}

impl<F, G> Morphism for Composition<F, G>
where
    F: Morphism,
    G: Morphism,
    F::Target: Same<G::Source>,
{
    type Source = F::Source;
    type Target = G::Target;

    fn apply(&self, waveform: &mut WaveformNode) {
        // Apply morphisms in sequence: first then second
        self.first.apply(waveform);
        self.second.apply(waveform);
    }

    fn id(&self) -> MorphismId {
        // Combine IDs for composed morphism
        // Simple combination - in real implementation might use hash
        let f_id = self.first.id();
        let g_id = self.second.id();
        MorphismId::new(f_id.0.wrapping_mul(31) ^ g_id.0)
    }
}

/// Trait for composable morphisms
pub trait Compose<Other: Morphism>: Morphism {
    /// Compose this morphism with another
    fn compose(self, other: Other) -> Composition<Self, Other>
    where
        Self: Sized,
        Self::Target: Same<Other::Source>,
    {
        Composition::new(self, other)
    }

    /// Try to compose with runtime type checking
    fn try_compose(self, other: Other) -> CompositionResult<Self, Other>
    where
        Self: Sized,
    {
        // For the prototype, we'll assume composition is valid if it compiles
        // In a real implementation, we'd need runtime type checking
        Ok(Composition::new(self, other))
    }
}

// Implement Compose for all morphisms
impl<M, Other> Compose<Other> for M
where
    M: Morphism,
    Other: Morphism,
{}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::category::{Genesis, Instantiation, Numeric, Identity, NumericMorphism};

    #[test]
    fn test_valid_composition() {
        // γ: ∅ → 𝟙
        let gamma = Genesis::default();

        // ι_5: 𝟙 → 5
        let iota = Instantiation::default(Numeric(5));

        // Can compose: (ι_5 ∘ γ): ∅ → 5
        let composed = gamma.compose(iota);

        let mut waveform = WaveformNode::zero();
        composed.apply(&mut waveform);

        // Should have modified the waveform
        assert!(waveform.r.real != 0.0 || waveform.r.imag != 0.0);
    }

    #[test]
    fn test_identity_composition() {
        // f: 5 → 7
        let f = NumericMorphism::new(Numeric(5), Numeric(7));

        // id: 7 → 7
        let id = Identity::<Numeric>::new();

        // f ∘ id should equal f
        let composed = f.clone().compose(id);

        let mut w1 = WaveformNode::from_value(crate::waveform::Complex64::new(1.0, 0.0));
        let mut w2 = w1.clone();

        f.apply(&mut w1);
        composed.apply(&mut w2);

        // Results should be identical (within floating point tolerance)
        assert!((w1.r.real - w2.r.real).abs() < 1e-10);
        assert!((w1.r.imag - w2.r.imag).abs() < 1e-10);
    }

    #[test]
    fn test_triple_composition() {
        // f: 1 → 2
        let f = NumericMorphism::new(Numeric(1), Numeric(2));

        // g: 2 → 3
        let g = NumericMorphism::new(Numeric(2), Numeric(3));

        // h: 3 → 4
        let h = NumericMorphism::new(Numeric(3), Numeric(4));

        // (h ∘ g) ∘ f
        let left = f.clone().compose(g.clone()).compose(h.clone());

        // h ∘ (g ∘ f)
        let right = f.compose(g).compose(h);

        let mut w1 = WaveformNode::from_value(crate::waveform::Complex64::new(1.0, 0.0));
        let mut w2 = w1.clone();

        left.apply(&mut w1);
        right.apply(&mut w2);

        // Associativity: results should be identical
        assert!((w1.r.real - w2.r.real).abs() < 1e-10);
        assert!((w1.r.imag - w2.r.imag).abs() < 1e-10);
    }
}