//! Builder pattern that enforces universal factorization

use super::morphisms::{Genesis, Instantiation, NumericMorphism};
use super::objects::Numeric;
use super::composition::Composition;
use crate::waveform::{GenesisParams, InstantiationParams};

/// Main builder that enforces categorical constraints
///
/// This builder ensures that:
/// 1. All morphisms from ∅ must go through γ
/// 2. No direct morphisms from ∅ to n are possible
/// 3. The factorization ∅ → γ → 𝟙 → ι_n → n is enforced
pub struct MorphismBuilder {
    // Private field prevents external construction
    _private: (),
}

impl MorphismBuilder {
    /// Create the builder (internal use only)
    fn new() -> Self {
        MorphismBuilder { _private: () }
    }

    /// Start building from the empty object
    ///
    /// This is the only way to create morphisms from ∅,
    /// and it forces you to go through the genesis morphism.
    pub fn from_empty() -> GenesisBuilder {
        GenesisBuilder::new()
    }

    /// Start building from a numeric object
    pub fn from_numeric(n: Numeric) -> NumericBuilder {
        NumericBuilder::new(n)
    }
}

/// Builder for morphisms starting from ∅
///
/// This builder enforces that all morphisms from ∅
/// must factor through the genesis morphism γ.
pub struct GenesisBuilder {
    params: Option<GenesisParams>,
}

impl GenesisBuilder {
    fn new() -> Self {
        GenesisBuilder { params: None }
    }

    /// Set custom genesis parameters
    pub fn with_params(mut self, params: GenesisParams) -> Self {
        self.params = Some(params);
        self
    }

    /// Create the genesis morphism γ: ∅ → 𝟙
    pub fn to_unit(self) -> Genesis {
        match self.params {
            Some(params) => Genesis::new(params),
            None => Genesis::default(),
        }
    }

    /// Continue to a numeric instance
    ///
    /// This automatically creates the composition ι_n ∘ γ
    pub fn then_instantiate(self, n: Numeric) -> InstantiationBuilder {
        InstantiationBuilder {
            genesis: self.to_unit(),
            target: n,
            params: None,
        }
    }

    /// Shorthand for creating the full factorization ∅ → n
    ///
    /// Returns the composed morphism (ι_n ∘ γ): ∅ → n
    pub fn to_numeric(self, n: Numeric) -> Composition<Genesis, Instantiation> {
        let genesis = self.to_unit();
        let instantiation = Instantiation::default(n);
        Composition::new(genesis, instantiation)
    }
}

/// Builder for instantiation morphisms
pub struct InstantiationBuilder {
    genesis: Genesis,
    target: Numeric,
    params: Option<InstantiationParams>,
}

impl InstantiationBuilder {
    /// Set custom instantiation parameters
    pub fn with_params(mut self, params: InstantiationParams) -> Self {
        self.params = Some(params);
        self
    }

    /// Build the composed morphism (ι_n ∘ γ): ∅ → n
    pub fn build(self) -> Composition<Genesis, Instantiation> {
        let instantiation = match self.params {
            Some(params) => Instantiation::new(self.target, params),
            None => Instantiation::default(self.target),
        };
        Composition::new(self.genesis, instantiation)
    }
}

/// Builder for numeric morphisms
pub struct NumericBuilder {
    source: Numeric,
}

impl NumericBuilder {
    fn new(source: Numeric) -> Self {
        NumericBuilder { source }
    }

    /// Create morphism to target numeric
    pub fn to(self, target: Numeric) -> NumericMorphism {
        NumericMorphism::new(self.source, target)
    }
}

/// Public API entry point for creating morphisms
///
/// This returns a builder that can be used to construct morphisms
/// with compile-time enforcement of categorical laws.
pub fn morphism() -> MorphismBuilder {
    MorphismBuilder::new()
}

// Alternative: make MorphismBuilder methods work as a fluent API
impl MorphismBuilder {
    /// Alternative entry point that can be chained
    pub fn new_builder() -> Self {
        MorphismBuilder { _private: () }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::category::Morphism;

    #[test]
    fn test_factorization_enforced() {
        // This is the ONLY way to create ∅ → n morphisms
        let morph = MorphismBuilder::from_empty()
            .to_numeric(Numeric(5));

        // The result is a composition of Genesis and Instantiation
        // This proves factorization is enforced at compile time

        let mut waveform = crate::waveform::WaveformNode::zero();
        morph.apply(&mut waveform);

        // Should have modified the waveform
        assert!(waveform.r.real != 0.0 || waveform.r.imag != 0.0);
    }

    #[test]
    fn test_custom_parameters() {
        let params_g = GenesisParams {
            base_frequency: 880.0, // A5
            initial_phase: std::f64::consts::PI / 4.0,
            amplitude: 0.5,
            _padding: 0.0,
        };

        let params_i = InstantiationParams {
            harmonic_coeffs: [0.9, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02],
            modulation_freq: 10.0,
            modulation_depth: 0.2,
            filter_cutoff: 12000.0,
            filter_resonance: 2.0,
        };

        let morph = MorphismBuilder::from_empty()
            .with_params(params_g)
            .then_instantiate(Numeric(10))
            .with_params(params_i)
            .build();

        let mut waveform = crate::waveform::WaveformNode::zero();
        morph.apply(&mut waveform);

        // Custom parameters should produce different results
        assert!(waveform.r.real != 0.0 || waveform.r.imag != 0.0);
    }

    #[test]
    fn test_numeric_morphisms() {
        let morph = MorphismBuilder::from_numeric(Numeric(3))
            .to(Numeric(7));

        let mut waveform = crate::waveform::WaveformNode::from_value(
            crate::waveform::Complex64::new(1.0, 0.0)
        );

        morph.apply(&mut waveform);

        // Should have transformed the waveform
        assert!(waveform.r.real != 1.0);
    }

    #[test]
    fn test_genesis_singleton() {
        // Multiple genesis morphisms should have the same ID
        let g1 = MorphismBuilder::from_empty().to_unit();
        let g2 = MorphismBuilder::from_empty().to_unit();

        assert_eq!(g1.id(), g2.id());
    }

    // This test demonstrates what CANNOT be done:
    // There's no way to directly create ∅ → n without factorization
    #[test]
    fn test_no_direct_empty_to_numeric() {
        // This code structure doesn't exist:
        // let invalid = DirectMorphism::<Empty, Numeric>::new(); // COMPILE ERROR

        // The only way is through factorization:
        let _valid = MorphismBuilder::from_empty()
            .to_numeric(Numeric(42));

        // This enforces the universal property at compile time
    }
}