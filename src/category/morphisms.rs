//! Morphism types with sealed trait pattern

use std::fmt;
use std::marker::PhantomData;

use super::objects::{Empty, Unit, Numeric};
use crate::waveform::{WaveformNode, GenesisParams, InstantiationParams, Complex64};

#[cfg(feature = "gpu")]
use crate::gpu::{GpuPipeline, GpuBuffer, GpuError};

/// Sealed trait pattern to prevent external morphism creation
pub(super) mod sealed {
    pub trait Sealed {}
}

/// Unique identifier for morphism caching and composition
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MorphismId(pub(crate) u64);

impl MorphismId {
    pub(crate) fn new(id: u64) -> Self {
        MorphismId(id)
    }
}

/// Base morphism trait
///
/// All morphisms must be sealed - this prevents external code from
/// creating invalid morphisms that violate categorical laws.
pub trait Morphism: sealed::Sealed + fmt::Debug {
    /// Source object of the morphism
    type Source;
    /// Target object of the morphism
    type Target;

    /// Apply morphism to waveform representation
    fn apply(&self, waveform: &mut WaveformNode);

    /// Unique identifier for caching/composition
    fn id(&self) -> MorphismId;

    /// Human-readable description
    fn description(&self) -> String {
        format!("{:?}", self)
    }
}

/// Genesis morphism γ: ∅ → 𝟙
///
/// The unique morphism from Empty to Unit.
/// Contains learnable parameters θ_γ for training.
#[derive(Debug, Clone)]
pub struct Genesis {
    params: GenesisParams,
}

impl Genesis {
    /// Create with default parameters
    pub fn new(params: GenesisParams) -> Self {
        Genesis { params }
    }

    /// Create with default parameters
    pub fn default() -> Self {
        Genesis {
            params: GenesisParams {
                base_frequency: 440.0, // A440
                initial_phase: 0.0,
                amplitude: 1.0,
                _padding: 0.0,
            }
        }
    }

    /// Get mutable reference to parameters for learning
    pub fn params_mut(&mut self) -> &mut GenesisParams {
        &mut self.params
    }

    /// Execute genesis morphism on GPU (γ: ∅ → 𝟙)
    #[cfg(feature = "gpu")]
    pub fn apply_gpu(
        &self,
        pipeline: &mut GpuPipeline,
        output: &mut GpuBuffer<WaveformNode>,
    ) -> Result<(), GpuError> {
        pipeline.execute_genesis(&self.params, output)
    }

    /// Apply genesis morphism on CPU (for testing/fallback)
    pub fn apply_cpu(&self, waveform: &mut WaveformNode) {
        self.apply(waveform);
    }
}

impl sealed::Sealed for Genesis {}

impl Morphism for Genesis {
    type Source = Empty;
    type Target = Unit;

    fn apply(&self, waveform: &mut WaveformNode) {
        // Generate fundamental frequency from void
        // This is a CPU placeholder for the GPU shader
        let freq = self.params.base_frequency;
        let phase = self.params.initial_phase;
        let amp = self.params.amplitude;

        // Simple sine wave generation in frequency domain
        // Real part represents cosine component, imaginary represents sine
        let omega = 2.0 * std::f64::consts::PI * freq / 44100.0; // Assuming 44.1kHz sample rate

        waveform.r = Complex64 {
            real: amp * (phase + omega).cos(),
            imag: amp * (phase + omega).sin(),
        };

        waveform.g = Complex64 {
            real: amp * 0.8 * (phase + omega * 1.5).cos(),
            imag: amp * 0.8 * (phase + omega * 1.5).sin(),
        };

        waveform.b = Complex64 {
            real: amp * 0.6 * (phase + omega * 2.0).cos(),
            imag: amp * 0.6 * (phase + omega * 2.0).sin(),
        };

        waveform.a = Complex64 {
            real: 1.0, // Alpha channel fully present
            imag: 0.0,
        };
    }

    fn id(&self) -> MorphismId {
        // Genesis is unique, always ID 1
        MorphismId::new(1)
    }
}

/// Instantiation morphism ι_n: 𝟙 → n
///
/// Maps from Unit to a specific Numeric instance.
/// Contains learnable parameters θ_ι for training.
#[derive(Debug, Clone)]
pub struct Instantiation {
    target: Numeric,
    params: InstantiationParams,
}

impl Instantiation {
    /// Create instantiation morphism to target n
    pub fn new(target: Numeric, params: InstantiationParams) -> Self {
        Instantiation { target, params }
    }

    /// Create with default parameters
    pub fn default(target: Numeric) -> Self {
        Instantiation {
            target,
            params: InstantiationParams {
                harmonic_coeffs: [1.0, 0.5, 0.33, 0.25, 0.2, 0.16, 0.14, 0.125],
                modulation_freq: 5.0,
                modulation_depth: 0.1,
                filter_cutoff: 8000.0,
                filter_resonance: 1.0,
            }
        }
    }

    /// Get the target numeric value
    pub fn target(&self) -> Numeric {
        self.target
    }

    /// Get mutable reference to parameters for learning
    pub fn params_mut(&mut self) -> &mut InstantiationParams {
        &mut self.params
    }

    /// Execute instantiation morphism on GPU (ι_n: 𝟙 → n)
    #[cfg(feature = "gpu")]
    pub fn apply_gpu(
        &self,
        pipeline: &mut GpuPipeline,
        input: &GpuBuffer<WaveformNode>,
        output: &mut GpuBuffer<WaveformNode>,
    ) -> Result<(), GpuError> {
        pipeline.execute_instantiation(&self.params, input, output)
    }

    /// Apply instantiation morphism on CPU (for testing/fallback)
    pub fn apply_cpu(&self, waveform: &mut WaveformNode) {
        self.apply(waveform);
    }
}

impl sealed::Sealed for Instantiation {}

impl Morphism for Instantiation {
    type Source = Unit;
    type Target = Numeric;

    fn apply(&self, waveform: &mut WaveformNode) {
        // Build harmonic complexity from proto-unity
        // This is a CPU placeholder for the GPU shader

        // Apply harmonic coefficients
        for (i, &coeff) in self.params.harmonic_coeffs.iter().enumerate() {
            let harmonic = (i + 2) as f64; // Start from 2nd harmonic

            // Modulate existing waveform with harmonics
            waveform.r.real *= 1.0 + coeff * harmonic.cos();
            waveform.r.imag *= 1.0 + coeff * harmonic.sin();

            waveform.g.real *= 1.0 + coeff * 0.8 * (harmonic * 1.5).cos();
            waveform.g.imag *= 1.0 + coeff * 0.8 * (harmonic * 1.5).sin();

            waveform.b.real *= 1.0 + coeff * 0.6 * (harmonic * 2.0).cos();
            waveform.b.imag *= 1.0 + coeff * 0.6 * (harmonic * 2.0).sin();
        }

        // Apply modulation
        let mod_factor = 1.0 + self.params.modulation_depth * self.params.modulation_freq.sin();
        waveform.r.real *= mod_factor;
        waveform.r.imag *= mod_factor;
        waveform.g.real *= mod_factor;
        waveform.g.imag *= mod_factor;
        waveform.b.real *= mod_factor;
        waveform.b.imag *= mod_factor;

        // Simple filter simulation (cutoff effect)
        let cutoff_factor = (self.params.filter_cutoff / 20000.0).min(1.0);
        waveform.r.real *= cutoff_factor;
        waveform.r.imag *= cutoff_factor;
        waveform.g.real *= cutoff_factor * 0.9;
        waveform.g.imag *= cutoff_factor * 0.9;
        waveform.b.real *= cutoff_factor * 0.8;
        waveform.b.imag *= cutoff_factor * 0.8;
    }

    fn id(&self) -> MorphismId {
        // ID based on target value
        MorphismId::new(1000 + self.target.value())
    }
}

/// Identity morphism id_A: A → A
#[derive(Debug, Clone)]
pub struct Identity<T> {
    _phantom: PhantomData<T>,
}

impl<T> Identity<T> {
    pub fn new() -> Self {
        Identity {
            _phantom: PhantomData,
        }
    }
}

impl<T> sealed::Sealed for Identity<T> {}

impl<T: 'static + fmt::Debug> Morphism for Identity<T> {
    type Source = T;
    type Target = T;

    fn apply(&self, _waveform: &mut WaveformNode) {
        // Identity morphism does nothing
    }

    fn id(&self) -> MorphismId {
        // Identity morphisms have special ID range
        MorphismId::new(100000)
    }
}

/// Numeric morphism between numeric instances
#[derive(Debug, Clone)]
pub struct NumericMorphism {
    source: Numeric,
    target: Numeric,
    transform_id: u64,
}

impl NumericMorphism {
    pub fn new(source: Numeric, target: Numeric) -> Self {
        // Simple transformation ID based on source and target
        let transform_id = (source.value() << 32) | target.value();
        NumericMorphism {
            source,
            target,
            transform_id,
        }
    }
}

impl sealed::Sealed for NumericMorphism {}

impl Morphism for NumericMorphism {
    type Source = Numeric;
    type Target = Numeric;

    fn apply(&self, waveform: &mut WaveformNode) {
        // Apply transformation from source to target numeric
        // This is a placeholder - actual implementation would depend on
        // the specific transformation being modeled

        let ratio = self.target.value() as f64 / self.source.value().max(1) as f64;

        // Simple frequency shift as example transformation
        waveform.r.real *= ratio;
        waveform.r.imag *= ratio;
        waveform.g.real *= ratio * 0.9;
        waveform.g.imag *= ratio * 0.9;
        waveform.b.real *= ratio * 0.8;
        waveform.b.imag *= ratio * 0.8;
    }

    fn id(&self) -> MorphismId {
        MorphismId::new(self.transform_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_genesis_morphism() {
        let genesis = Genesis::default();
        let mut waveform = WaveformNode::zero();

        genesis.apply(&mut waveform);

        // Should have generated some frequency content
        assert!(waveform.r.real != 0.0 || waveform.r.imag != 0.0);
        assert_eq!(waveform.a.real, 1.0); // Alpha should be 1
    }

    #[test]
    fn test_instantiation_morphism() {
        let inst = Instantiation::default(Numeric(5));
        let mut waveform = WaveformNode::from_value(Complex64::new(1.0, 0.0));

        inst.apply(&mut waveform);

        // Should have modified the waveform
        assert!(waveform.r.real != 1.0);
    }

    #[test]
    fn test_identity_morphism() {
        let id: Identity<Unit> = Identity::new();
        let mut waveform = WaveformNode::from_value(Complex64::new(1.0, 2.0));
        let original = waveform.clone();

        id.apply(&mut waveform);

        // Identity should not change waveform
        assert_eq!(waveform, original);
    }

    #[test]
    fn test_morphism_ids() {
        let g1 = Genesis::default();
        let g2 = Genesis::default();
        assert_eq!(g1.id(), g2.id()); // Genesis is unique

        let i1 = Instantiation::default(Numeric(5));
        let i2 = Instantiation::default(Numeric(7));
        assert_ne!(i1.id(), i2.id()); // Different targets have different IDs
    }
}