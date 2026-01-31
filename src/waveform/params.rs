//! Learnable parameters for morphisms

use std::fmt;

/// Learnable parameters for the Genesis morphism γ
///
/// These parameters (θ_γ) control how the genesis morphism
/// generates waveforms from the empty object.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug)]
pub struct GenesisParams {
    /// Base frequency in Hz for fundamental tone generation
    pub base_frequency: f64,

    /// Initial phase offset in radians
    pub initial_phase: f64,

    /// Amplitude scaling factor (0.0 to 1.0 typically)
    pub amplitude: f64,

    /// Padding for 16-byte alignment
    pub _padding: f64,
}

impl GenesisParams {
    /// Create new genesis parameters
    pub fn new(base_frequency: f64, initial_phase: f64, amplitude: f64) -> Self {
        GenesisParams {
            base_frequency,
            initial_phase,
            amplitude,
            _padding: 0.0,
        }
    }

    /// Default parameters (A440 at full amplitude)
    pub fn default() -> Self {
        GenesisParams {
            base_frequency: 440.0,
            initial_phase: 0.0,
            amplitude: 1.0,
            _padding: 0.0,
        }
    }

    /// Apply gradient update for learning
    pub fn update(&mut self, gradient: &GenesisParams, learning_rate: f64) {
        self.base_frequency -= learning_rate * gradient.base_frequency;
        self.initial_phase -= learning_rate * gradient.initial_phase;
        self.amplitude -= learning_rate * gradient.amplitude;

        // Clamp amplitude to valid range
        self.amplitude = self.amplitude.max(0.0).min(1.0);
    }

    /// Initialize with random values for training
    pub fn random() -> Self {
        // Note: In real implementation, would use a proper RNG
        GenesisParams {
            base_frequency: 200.0 + (1000.0 * 0.5), // 200-1200 Hz range
            initial_phase: 0.0,
            amplitude: 0.5 + (0.5 * 0.5), // 0.5-1.0 range
            _padding: 0.0,
        }
    }
}

impl fmt::Display for GenesisParams {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GenesisParams {{ freq: {:.1}Hz, phase: {:.3}rad, amp: {:.3} }}",
               self.base_frequency, self.initial_phase, self.amplitude)
    }
}

/// Learnable parameters for Instantiation morphisms ι_n
///
/// These parameters (θ_ι) control how the instantiation morphism
/// builds harmonic complexity when mapping from Unit to Numeric.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug)]
pub struct InstantiationParams {
    /// Harmonic coefficients for the first 8 harmonics
    /// Each coefficient scales the corresponding harmonic amplitude
    pub harmonic_coeffs: [f64; 8],

    /// Modulation frequency in Hz
    pub modulation_freq: f64,

    /// Modulation depth (0.0 to 1.0)
    pub modulation_depth: f64,

    /// Filter cutoff frequency in Hz
    pub filter_cutoff: f64,

    /// Filter resonance/Q factor (typically 0.5 to 10.0)
    pub filter_resonance: f64,
}

impl InstantiationParams {
    /// Create new instantiation parameters
    pub fn new(
        harmonic_coeffs: [f64; 8],
        modulation_freq: f64,
        modulation_depth: f64,
        filter_cutoff: f64,
        filter_resonance: f64,
    ) -> Self {
        InstantiationParams {
            harmonic_coeffs,
            modulation_freq,
            modulation_depth,
            filter_cutoff,
            filter_resonance,
        }
    }

    /// Default parameters (natural harmonic series)
    pub fn default() -> Self {
        InstantiationParams {
            harmonic_coeffs: [1.0, 0.5, 0.33, 0.25, 0.2, 0.16, 0.14, 0.125],
            modulation_freq: 5.0,
            modulation_depth: 0.1,
            filter_cutoff: 8000.0,
            filter_resonance: 1.0,
        }
    }

    /// Apply gradient update for learning
    pub fn update(&mut self, gradient: &InstantiationParams, learning_rate: f64) {
        // Update harmonic coefficients
        for i in 0..8 {
            self.harmonic_coeffs[i] -= learning_rate * gradient.harmonic_coeffs[i];
            // Clamp to reasonable range
            self.harmonic_coeffs[i] = self.harmonic_coeffs[i].max(-2.0).min(2.0);
        }

        // Update modulation parameters
        self.modulation_freq -= learning_rate * gradient.modulation_freq;
        self.modulation_depth -= learning_rate * gradient.modulation_depth;
        self.modulation_depth = self.modulation_depth.max(0.0).min(1.0);

        // Update filter parameters
        self.filter_cutoff -= learning_rate * gradient.filter_cutoff;
        self.filter_cutoff = self.filter_cutoff.max(20.0).min(20000.0);

        self.filter_resonance -= learning_rate * gradient.filter_resonance;
        self.filter_resonance = self.filter_resonance.max(0.5).min(10.0);
    }

    /// Initialize with random values for training
    pub fn random() -> Self {
        // Note: In real implementation, would use a proper RNG
        let mut coeffs = [0.0; 8];
        for (i, coeff) in coeffs.iter_mut().enumerate() {
            *coeff = 1.0 / ((i + 1) as f64); // Natural harmonic series as base
        }

        InstantiationParams {
            harmonic_coeffs: coeffs,
            modulation_freq: 1.0 + (10.0 * 0.5),
            modulation_depth: 0.05 + (0.2 * 0.5),
            filter_cutoff: 1000.0 + (10000.0 * 0.5),
            filter_resonance: 0.7 + (2.0 * 0.5),
        }
    }

    /// Get total number of parameters
    pub fn param_count() -> usize {
        8 + 4 // 8 harmonic coeffs + 4 other params
    }
}

impl fmt::Display for InstantiationParams {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "InstantiationParams {{\n")?;
        write!(f, "  harmonics: [")?;
        for (i, &coeff) in self.harmonic_coeffs.iter().enumerate() {
            if i > 0 { write!(f, ", ")?; }
            write!(f, "{:.2}", coeff)?;
        }
        write!(f, "],\n")?;
        write!(f, "  mod: {:.1}Hz @ {:.1}%,\n",
               self.modulation_freq, self.modulation_depth * 100.0)?;
        write!(f, "  filter: {:.0}Hz Q={:.1}\n",
               self.filter_cutoff, self.filter_resonance)?;
        write!(f, "}}")
    }
}

/// Combined parameter set for full morphism chain
#[derive(Clone, Copy, Debug)]
pub struct MorphismParameters {
    pub genesis: GenesisParams,
    pub instantiation: InstantiationParams,
}

impl MorphismParameters {
    /// Create new combined parameters
    pub fn new(genesis: GenesisParams, instantiation: InstantiationParams) -> Self {
        MorphismParameters {
            genesis,
            instantiation,
        }
    }

    /// Default parameters
    pub fn default() -> Self {
        MorphismParameters {
            genesis: GenesisParams::default(),
            instantiation: InstantiationParams::default(),
        }
    }

    /// Random initialization for training
    pub fn random() -> Self {
        MorphismParameters {
            genesis: GenesisParams::random(),
            instantiation: InstantiationParams::random(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{align_of, size_of};

    #[test]
    fn test_alignment_and_size() {
        // Check GPU-friendly alignment
        assert_eq!(align_of::<GenesisParams>(), 16);
        assert_eq!(size_of::<GenesisParams>(), 32); // 4 * 8 bytes

        assert_eq!(align_of::<InstantiationParams>(), 16);
        assert_eq!(size_of::<InstantiationParams>(), 96); // 8*8 + 4*8 = 96 bytes
    }

    #[test]
    fn test_parameter_updates() {
        let mut params = GenesisParams::default();
        let gradient = GenesisParams::new(10.0, 0.1, 0.05);
        let lr = 0.01;

        params.update(&gradient, lr);

        assert_eq!(params.base_frequency, 440.0 - 0.1);
        assert_eq!(params.initial_phase, 0.0 - 0.001);
        assert_eq!(params.amplitude, 1.0 - 0.0005);
    }

    #[test]
    fn test_clamping() {
        let mut params = InstantiationParams::default();

        // Try to push parameters out of bounds
        let mut gradient = InstantiationParams::default();
        gradient.modulation_depth = 100.0; // Large gradient
        gradient.filter_cutoff = -30000.0; // Negative gradient

        params.update(&gradient, 1.0);

        // Check clamping worked
        assert!(params.modulation_depth >= 0.0);
        assert!(params.modulation_depth <= 1.0);
        assert!(params.filter_cutoff >= 20.0);
        assert!(params.filter_cutoff <= 20000.0);
    }
}