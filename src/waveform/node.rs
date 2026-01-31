//! Waveform node structure for frequency domain representation

use std::fmt;

/// Complex number representation for frequency domain
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Complex64 {
    pub real: f64,
    pub imag: f64,
}

impl Complex64 {
    /// Create a new complex number
    pub fn new(real: f64, imag: f64) -> Self {
        Complex64 { real, imag }
    }

    /// Zero complex number
    pub fn zero() -> Self {
        Complex64 { real: 0.0, imag: 0.0 }
    }

    /// Unity complex number (1 + 0i)
    pub fn one() -> Self {
        Complex64 { real: 1.0, imag: 0.0 }
    }

    /// Magnitude (absolute value)
    pub fn magnitude(&self) -> f64 {
        (self.real * self.real + self.imag * self.imag).sqrt()
    }

    /// Phase angle in radians
    pub fn phase(&self) -> f64 {
        self.imag.atan2(self.real)
    }

    /// Complex conjugate
    pub fn conjugate(&self) -> Self {
        Complex64 {
            real: self.real,
            imag: -self.imag,
        }
    }

    /// Complex multiplication
    pub fn multiply(&self, other: &Complex64) -> Self {
        Complex64 {
            real: self.real * other.real - self.imag * other.imag,
            imag: self.real * other.imag + self.imag * other.real,
        }
    }

    /// Complex addition
    pub fn add(&self, other: &Complex64) -> Self {
        Complex64 {
            real: self.real + other.real,
            imag: self.imag + other.imag,
        }
    }
}

impl fmt::Display for Complex64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.imag >= 0.0 {
            write!(f, "{:.3} + {:.3}i", self.real, self.imag)
        } else {
            write!(f, "{:.3} - {:.3}i", self.real, -self.imag)
        }
    }
}

/// Waveform node representing frequency domain state
///
/// Each node contains four complex numbers representing
/// RGBA channels in the frequency domain. This structure
/// is aligned for efficient GPU processing.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaveformNode {
    /// Red channel frequency components
    pub r: Complex64,
    /// Green channel frequency components
    pub g: Complex64,
    /// Blue channel frequency components
    pub b: Complex64,
    /// Alpha channel frequency components
    pub a: Complex64,
}

impl WaveformNode {
    /// Create a new waveform node with given values
    pub fn new(r: Complex64, g: Complex64, b: Complex64, a: Complex64) -> Self {
        WaveformNode { r, g, b, a }
    }

    /// Create a zero waveform (silence/black)
    pub fn zero() -> Self {
        WaveformNode {
            r: Complex64::zero(),
            g: Complex64::zero(),
            b: Complex64::zero(),
            a: Complex64::zero(),
        }
    }

    /// Create a unity waveform (all channels at 1+0i)
    pub fn one() -> Self {
        WaveformNode {
            r: Complex64::one(),
            g: Complex64::one(),
            b: Complex64::one(),
            a: Complex64::one(),
        }
    }

    /// Create waveform with all channels set to same value
    pub fn from_value(value: Complex64) -> Self {
        WaveformNode {
            r: value,
            g: value,
            b: value,
            a: value,
        }
    }

    /// Calculate total energy (sum of magnitudes)
    pub fn energy(&self) -> f64 {
        self.r.magnitude() + self.g.magnitude() + self.b.magnitude() + self.a.magnitude()
    }

    /// Apply complex multiplication to all channels
    pub fn multiply(&mut self, factor: Complex64) {
        self.r = self.r.multiply(&factor);
        self.g = self.g.multiply(&factor);
        self.b = self.b.multiply(&factor);
        self.a = self.a.multiply(&factor);
    }

    /// Add another waveform node
    pub fn add(&mut self, other: &WaveformNode) {
        self.r = self.r.add(&other.r);
        self.g = self.g.add(&other.g);
        self.b = self.b.add(&other.b);
        self.a = self.a.add(&other.a);
    }

    /// Scale all channels by a real factor
    pub fn scale(&mut self, factor: f64) {
        self.r.real *= factor;
        self.r.imag *= factor;
        self.g.real *= factor;
        self.g.imag *= factor;
        self.b.real *= factor;
        self.b.imag *= factor;
        self.a.real *= factor;
        self.a.imag *= factor;
    }

    /// Apply phase rotation to all channels
    pub fn rotate_phase(&mut self, angle: f64) {
        let rotation = Complex64::new(angle.cos(), angle.sin());
        self.r = self.r.multiply(&rotation);
        self.g = self.g.multiply(&rotation);
        self.b = self.b.multiply(&rotation);
        self.a = self.a.multiply(&rotation);
    }
}

impl fmt::Display for WaveformNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "WaveformNode {{\n")?;
        write!(f, "  R: {},\n", self.r)?;
        write!(f, "  G: {},\n", self.g)?;
        write!(f, "  B: {},\n", self.b)?;
        write!(f, "  A: {}\n", self.a)?;
        write!(f, "}}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complex_operations() {
        let a = Complex64::new(3.0, 4.0);
        let b = Complex64::new(1.0, 2.0);

        // Test magnitude: |3+4i| = 5
        assert_eq!(a.magnitude(), 5.0);

        // Test multiplication: (3+4i)(1+2i) = 3+6i+4i+8i² = 3+10i-8 = -5+10i
        let product = a.multiply(&b);
        assert_eq!(product.real, -5.0);
        assert_eq!(product.imag, 10.0);

        // Test conjugate
        let conj = a.conjugate();
        assert_eq!(conj.real, 3.0);
        assert_eq!(conj.imag, -4.0);
    }

    #[test]
    fn test_waveform_node() {
        let mut node = WaveformNode::zero();
        assert_eq!(node.energy(), 0.0);

        node.r = Complex64::new(1.0, 0.0);
        node.g = Complex64::new(0.0, 1.0);
        assert_eq!(node.energy(), 2.0);

        node.scale(2.0);
        assert_eq!(node.r.real, 2.0);
        assert_eq!(node.g.imag, 2.0);

        node.rotate_phase(std::f64::consts::PI / 2.0);
        // After 90° rotation, real becomes -imag, imag becomes real
        assert!((node.r.real - 0.0).abs() < 1e-10);
        assert!((node.r.imag - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_alignment() {
        use std::mem::{align_of, size_of};

        // Ensure proper alignment for GPU
        assert_eq!(align_of::<WaveformNode>(), 16);
        assert_eq!(size_of::<WaveformNode>(), 64); // 4 * 16 bytes

        assert_eq!(size_of::<Complex64>(), 16); // 2 * 8 bytes
    }
}