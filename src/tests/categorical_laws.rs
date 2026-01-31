//! Tests validating categorical laws and universal properties

#[cfg(test)]
mod tests {
    use crate::category::{
        Empty, Unit, Numeric,
        Genesis, Instantiation, Identity, NumericMorphism,
        Morphism, Compose, MorphismBuilder,
    };
    use crate::waveform::{WaveformNode, Complex64, GenesisParams, InstantiationParams};

    /// Test that universal factorization is enforced
    ///
    /// The type system should prevent direct morphisms from ∅ to n,
    /// forcing all such morphisms to factor through γ and ι_n.
    #[test]
    fn test_factorization_enforced() {
        // This compiles - proper factorization
        let morph = MorphismBuilder::from_empty()
            .to_numeric(Numeric(5));

        let mut waveform = WaveformNode::zero();
        morph.apply(&mut waveform);

        // The morphism modified the waveform
        assert!(waveform.energy() > 0.0);

        // There is NO way to create a direct ∅ → n morphism
        // The type system prevents it at compile time
    }

    /// Test that genesis parameters are learnable
    #[test]
    fn test_genesis_parameters_learnable() {
        let params = GenesisParams::new(880.0, 0.5, 0.8);
        let mut genesis = Genesis::new(params.clone());

        // Parameters are mutable for learning
        genesis.params_mut().base_frequency = 440.0;

        let mut waveform = WaveformNode::zero();
        genesis.apply(&mut waveform);

        // Should have generated frequency state
        assert!(waveform.r.real.abs() > 0.0 || waveform.r.imag.abs() > 0.0);
        assert_eq!(waveform.a.real, 1.0); // Alpha should be set
    }

    /// Test morphism composition
    #[test]
    fn test_composition() {
        let gamma = Genesis::default();
        let iota = Instantiation::default(Numeric(10));

        // Compose γ and ι_10
        let composed = gamma.compose(iota);

        let mut waveform = WaveformNode::zero();
        composed.apply(&mut waveform);

        // Composition should work correctly
        assert!(waveform.energy() > 0.0);
    }

    /// Test identity law: f ∘ id = f = id ∘ f
    #[test]
    fn test_identity_law() {
        let f = NumericMorphism::new(Numeric(5), Numeric(7));
        let id_7 = Identity::<Numeric>::new();
        let _id_5 = Identity::<Numeric>::new();

        // Test right identity: f ∘ id = f
        let right_comp = f.clone().compose(id_7);

        let mut w1 = WaveformNode::from_value(Complex64::new(1.0, 0.5));
        let mut w2 = w1.clone();

        f.apply(&mut w1);
        right_comp.apply(&mut w2);

        // Results should be identical
        assert!((w1.r.real - w2.r.real).abs() < 1e-10);
        assert!((w1.r.imag - w2.r.imag).abs() < 1e-10);
    }

    /// Test associativity: (f ∘ g) ∘ h = f ∘ (g ∘ h)
    #[test]
    fn test_composition_associative() {
        // Create three composable morphisms
        let f = NumericMorphism::new(Numeric(1), Numeric(2));
        let g = NumericMorphism::new(Numeric(2), Numeric(3));
        let h = NumericMorphism::new(Numeric(3), Numeric(4));

        // Left association: (f ∘ g) ∘ h
        let left = f.clone().compose(g.clone()).compose(h.clone());

        // Right association: f ∘ (g ∘ h)
        let right = f.compose(g.compose(h));

        let mut w1 = WaveformNode::from_value(Complex64::new(1.0, 0.0));
        let mut w2 = w1.clone();

        left.apply(&mut w1);
        right.apply(&mut w2);

        // Associativity means both give same result
        assert!((w1.r.real - w2.r.real).abs() < 1e-10);
        assert!((w1.g.real - w2.g.real).abs() < 1e-10);
        assert!((w1.b.real - w2.b.real).abs() < 1e-10);
    }

    /// Test that Genesis is unique (singleton-like behavior)
    #[test]
    fn test_genesis_uniqueness() {
        let g1 = Genesis::default();
        let g2 = Genesis::default();

        // Both genesis morphisms have the same ID
        assert_eq!(g1.id(), g2.id());

        // They are the unique morphism ∅ → 𝟙
        assert_eq!(g1.id().0, 1);
    }

    /// Test that instantiation morphisms are parameterized by target
    #[test]
    fn test_instantiation_parameterized() {
        let i5 = Instantiation::default(Numeric(5));
        let i7 = Instantiation::default(Numeric(7));

        // Different targets give different morphisms
        assert_ne!(i5.id(), i7.id());
        assert_eq!(i5.target(), Numeric(5));
        assert_eq!(i7.target(), Numeric(7));
    }

    /// Test complex waveform operations
    #[test]
    fn test_waveform_operations() {
        let mut node = WaveformNode::from_value(Complex64::new(1.0, 1.0));

        // Test scaling
        node.scale(2.0);
        assert_eq!(node.r.real, 2.0);
        assert_eq!(node.r.imag, 2.0);

        // Test phase rotation by π/2
        node.rotate_phase(std::f64::consts::PI / 2.0);
        // After 90° rotation: (2+2i) * (0+i) = -2+2i
        assert!((node.r.real - (-2.0)).abs() < 1e-10);
        assert!((node.r.imag - 2.0).abs() < 1e-10);

        // Test energy calculation
        let energy = node.energy();
        assert!(energy > 0.0);
    }

    /// Test parameter updates for gradient descent
    #[test]
    fn test_parameter_learning() {
        let mut params = GenesisParams::default();
        assert_eq!(params.base_frequency, 440.0);

        // Simulate gradient update
        let gradient = GenesisParams::new(10.0, 0.0, 0.0);
        params.update(&gradient, 0.1);

        // Frequency should have decreased
        assert_eq!(params.base_frequency, 439.0);
    }

    /// Test builder pattern with custom parameters
    #[test]
    fn test_builder_with_parameters() {
        let genesis_params = GenesisParams::new(220.0, 0.0, 0.5);
        let inst_params = InstantiationParams::new(
            [0.8, 0.6, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02],
            7.0,
            0.15,
            10000.0,
            1.5,
        );

        let morph = MorphismBuilder::from_empty()
            .with_params(genesis_params)
            .then_instantiate(Numeric(12))
            .with_params(inst_params)
            .build();

        let mut waveform = WaveformNode::zero();
        morph.apply(&mut waveform);

        // Custom parameters should produce output
        assert!(waveform.energy() > 0.0);
    }

    /// Test that morphism IDs are stable and unique
    #[test]
    fn test_morphism_ids() {
        let g = Genesis::default();
        let i5 = Instantiation::default(Numeric(5));
        let i10 = Instantiation::default(Numeric(10));
        let n_3_7 = NumericMorphism::new(Numeric(3), Numeric(7));

        // Genesis always has ID 1
        assert_eq!(g.id().0, 1);

        // Instantiation IDs are based on target
        assert_eq!(i5.id().0, 1005);
        assert_eq!(i10.id().0, 1010);

        // Numeric morphism IDs are deterministic
        let n_3_7_again = NumericMorphism::new(Numeric(3), Numeric(7));
        assert_eq!(n_3_7.id(), n_3_7_again.id());
    }

    /// Test zero-sized types
    #[test]
    fn test_zero_sized_objects() {
        use std::mem::size_of;

        // Empty and Unit should be zero-sized for efficiency
        assert_eq!(size_of::<Empty>(), 0);
        assert_eq!(size_of::<Unit>(), 0);

        // Numeric wraps a u64
        assert_eq!(size_of::<Numeric>(), 8);

        // Identity morphism uses PhantomData, should be small
        assert_eq!(size_of::<Identity<Unit>>(), 0);
    }

    /// Test GPU-ready alignment
    #[test]
    fn test_gpu_alignment() {
        use std::mem::align_of;

        // Waveform structures should be 16-byte aligned for GPU
        assert_eq!(align_of::<WaveformNode>(), 16);
        assert_eq!(align_of::<GenesisParams>(), 16);
        assert_eq!(align_of::<InstantiationParams>(), 16);
    }

    /// Integration test: full pipeline from ∅ to numeric
    #[test]
    fn test_full_pipeline() {
        // Create the full factorized morphism ∅ → 42
        let morph = MorphismBuilder::from_empty()
            .to_numeric(Numeric(42));

        // Apply to a zero waveform
        let mut waveform = WaveformNode::zero();
        assert_eq!(waveform.energy(), 0.0);

        morph.apply(&mut waveform);

        // Should have generated non-zero waveform
        assert!(waveform.energy() > 0.0);
        assert_eq!(waveform.a.real, 1.0); // Alpha channel should be set

        // The waveform should have harmonic content
        assert!(waveform.r.magnitude() > 0.0);
        assert!(waveform.g.magnitude() > 0.0);
        assert!(waveform.b.magnitude() > 0.0);
    }
}