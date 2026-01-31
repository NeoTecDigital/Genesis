/*!
 * Zero Object (○) - Initial ∧ Terminal Object
 *
 * Implements the corrected categorical structure:
 * - ○ (ZeroObject): Both initial and terminal
 * - ∅ (EmptyAspect): Emergence/potential path via η: ○ → ∅
 * - ∞ (InfiniteAspect): Evaluation/actual path via ζ: ∞ → ○
 *
 * Complete cycle:
 * Forward:  ○ → η → ∅ → γ → 𝟙 → ι → n
 * Backward: n → τ → 𝟙 → ε → ∞ → ζ → ○
 *
 * Self-optimization: The ○ → ... → ○ cycle IS Bayesian optimization
 */

use std::collections::HashMap;

/// Zero Object (○) - Initial ∧ Terminal
///
/// Represents the current best model/understanding.
/// Both initial (unique morphism to all objects) and terminal (unique morphism from all objects).
#[derive(Debug, Clone)]
pub struct ZeroObject {
    /// Current best parameter configuration
    pub theta: ModelParameters,

    /// Convergence history
    pub iteration: usize,

    /// Current loss value
    pub loss: f32,

    /// Metadata for tracking
    pub metadata: HashMap<String, f32>,
}

/// Model Parameters (learnable θ)
#[derive(Debug, Clone)]
pub struct ModelParameters {
    /// γ (genesis) parameters
    pub gamma: GammaParams,

    /// ι (instantiation) parameters
    pub iota: IotaParams,

    /// τ (encoder) parameters
    pub tau: TauParams,

    /// ε (collapse) parameters
    pub epsilon: EpsilonParams,
}

#[derive(Debug, Clone)]
pub struct GammaParams {
    pub base_frequency: f32,
    pub initial_phase: f32,
    pub amplitude: f32,
    pub envelope_sigma: f32,
    pub num_harmonics: u32,
    pub harmonic_decay: f32,
}

#[derive(Debug, Clone)]
pub struct IotaParams {
    pub K: usize,
    pub harmonic_coeffs: Vec<f32>,  // Length K
    pub phase_shifts: Vec<f32>,     // Length K
    pub global_amplitude: f32,
    pub frequency_range: f32,
}

#[derive(Debug, Clone)]
pub struct TauParams {
    pub normalization_epsilon: f32,
    pub projection_strength: f32,
    pub noise_threshold: f32,
    pub use_template_normalization: bool,
}

#[derive(Debug, Clone)]
pub struct EpsilonParams {
    pub energy_weight: f32,
    pub coherence_weight: f32,
    pub sparsity_weight: f32,
    pub quality_weight: f32,
    pub reduction_factor: u32,
    pub coherence_threshold: f32,
}

/// Empty Aspect (∅) - Sampling Distribution
///
/// Represents where to explore next in parameter space.
/// Generated from ○ via η: ○ → ∅
#[derive(Debug, Clone)]
pub struct EmptyAspect {
    /// Gaussian Process mean function (learned from ○)
    pub gp_mean: ModelParameters,

    /// Gaussian Process kernel parameters
    pub gp_lengthscale: Vec<f32>,
    pub gp_variance: f32,

    /// Acquisition function type
    pub acquisition: AcquisitionFunction,

    /// Exploration vs exploitation trade-off
    pub exploration_weight: f32,
}

#[derive(Debug, Clone)]
pub enum AcquisitionFunction {
    /// Expected Improvement
    EI,
    /// Upper Confidence Bound
    UCB { beta: f32 },
    /// Probability of Improvement
    PI,
}

/// Infinite Aspect (∞) - Evaluation History
///
/// Represents all past evaluations and learned knowledge.
/// Updated via ε: 𝟙 → ∞, then used to update ○ via ζ: ∞ → ○
#[derive(Debug)]
pub struct InfiniteAspect {
    /// Evaluation history (θ, loss) pairs
    pub history: Vec<(ModelParameters, f32)>,

    /// Evaluation metrics history
    pub metrics_history: Vec<EvaluationMetrics>,

    /// Pareto frontier (for multi-objective optimization)
    pub pareto_frontier: Vec<(ModelParameters, Vec<f32>)>,

    /// GP hyperparameters (learned from history)
    pub gp_hyperparams: GaussianProcessHyperparams,
}

#[derive(Debug, Clone)]
pub struct EvaluationMetrics {
    pub energy: f32,
    pub coherence: f32,
    pub sparsity: f32,
    pub quality: f32,
    pub factorization_loss: f32,
}

#[derive(Debug, Clone)]
pub struct GaussianProcessHyperparams {
    pub lengthscale: Vec<f32>,
    pub signal_variance: f32,
    pub noise_variance: f32,
}

/// η (eta): ○ → ∅
///
/// Enter potential space - generate sampling distribution from current best
pub fn eta(zero: &ZeroObject) -> EmptyAspect {
    // Create sampling distribution centered on current best
    EmptyAspect {
        gp_mean: zero.theta.clone(),
        gp_lengthscale: vec![0.1; 20], // 20D parameter space (simplified)
        gp_variance: 1.0,
        acquisition: AcquisitionFunction::UCB { beta: 2.0 },
        exploration_weight: 1.0 / (1.0 + zero.iteration as f32).sqrt(), // Decay over time
    }
}

/// ζ (zeta): ∞ → ○
///
/// Return to ground state - update ○ based on evaluation history ∞
pub fn zeta(infinity: &InfiniteAspect) -> ZeroObject {
    // Find best configuration from history
    let (best_theta, best_loss) = infinity.history
        .iter()
        .min_by(|(_, loss1), (_, loss2)| loss1.partial_cmp(loss2).unwrap())
        .map(|(theta, loss)| (theta.clone(), *loss))
        .expect("History must not be empty");

    let mut metadata = HashMap::new();
    metadata.insert("history_size".to_string(), infinity.history.len() as f32);
    metadata.insert("pareto_size".to_string(), infinity.pareto_frontier.len() as f32);

    ZeroObject {
        theta: best_theta,
        iteration: infinity.history.len(),
        loss: best_loss,
        metadata,
    }
}

/// Morphism: ∅ → ○ (sample from potential space)
///
/// Sample next configuration to try from sampling distribution
pub fn sample_from_empty(empty: &EmptyAspect, rng: &mut impl rand::Rng) -> ModelParameters {
    // Simplified: Perturb mean with Gaussian noise
    let mut sampled = empty.gp_mean.clone();

    // Perturb gamma params
    sampled.gamma.base_frequency += rng.gen_range(-0.5..0.5) * empty.gp_lengthscale[0];
    sampled.gamma.amplitude += rng.gen_range(-10.0..10.0) * empty.gp_lengthscale[1];
    sampled.gamma.envelope_sigma += rng.gen_range(-0.1..0.1) * empty.gp_lengthscale[2];
    sampled.gamma.num_harmonics = (sampled.gamma.num_harmonics as i32
        + rng.gen_range(-2..3)).max(1).min(24) as u32;

    // Perturb iota params (first few coefficients as example)
    for i in 0..sampled.iota.harmonic_coeffs.len().min(10) {
        sampled.iota.harmonic_coeffs[i] += rng.gen_range(-0.2..0.2) * empty.gp_lengthscale[3];
        sampled.iota.harmonic_coeffs[i] = sampled.iota.harmonic_coeffs[i].max(0.1).min(2.0);
    }

    // Perturb tau params
    sampled.tau.projection_strength += rng.gen_range(-0.1..0.1) * empty.gp_lengthscale[4];
    sampled.tau.projection_strength = sampled.tau.projection_strength.clamp(0.0, 1.0);

    // Perturb epsilon params
    sampled.epsilon.coherence_weight += rng.gen_range(-20.0..20.0) * empty.gp_lengthscale[5];
    sampled.epsilon.coherence_weight = sampled.epsilon.coherence_weight.max(1.0);

    sampled
}

/// Morphism: evaluation → ∞ (add to history)
///
/// Store evaluation results in infinite space
pub fn update_infinite(infinity: &mut InfiniteAspect, theta: ModelParameters, metrics: EvaluationMetrics) {
    let loss = compute_loss(&metrics);

    // Add to history
    infinity.history.push((theta.clone(), loss));
    infinity.metrics_history.push(metrics.clone());

    // Update Pareto frontier (multi-objective)
    let objectives = vec![
        metrics.factorization_loss,
        -metrics.quality, // Negative because we want to maximize quality
        metrics.coherence,
    ];

    update_pareto_frontier(&mut infinity.pareto_frontier, theta, objectives);

    // Update GP hyperparameters (simplified - would use MLE in practice)
    if infinity.history.len() > 10 {
        // Estimate lengthscales from history variance
        for i in 0..infinity.gp_hyperparams.lengthscale.len() {
            infinity.gp_hyperparams.lengthscale[i] *= 0.99; // Slowly decrease (more confident)
        }
    }
}

fn compute_loss(metrics: &EvaluationMetrics) -> f32 {
    // Weighted combination
    0.5 * metrics.factorization_loss
        + 0.3 * (1.0 - metrics.quality / 100.0) // Normalize quality to [0,1]
        + 0.2 * (1.0 - metrics.coherence)
}

fn update_pareto_frontier(
    frontier: &mut Vec<(ModelParameters, Vec<f32>)>,
    theta: ModelParameters,
    objectives: Vec<f32>,
) {
    // Check if this point dominates any existing points
    frontier.retain(|(_, existing_objs)| {
        !dominates(&objectives, existing_objs)
    });

    // Check if this point is dominated by any existing point
    let is_dominated = frontier.iter().any(|(_, existing_objs)| {
        dominates(existing_objs, &objectives)
    });

    if !is_dominated {
        frontier.push((theta, objectives));
    }
}

fn dominates(a: &[f32], b: &[f32]) -> bool {
    // a dominates b if a is better or equal in all objectives and strictly better in at least one
    let all_better_or_equal = a.iter().zip(b.iter()).all(|(ai, bi)| ai <= bi);
    let at_least_one_better = a.iter().zip(b.iter()).any(|(ai, bi)| ai < bi);
    all_better_or_equal && at_least_one_better
}

impl Default for ModelParameters {
    fn default() -> Self {
        Self {
            gamma: GammaParams {
                base_frequency: 2.0,
                initial_phase: 0.0,
                amplitude: 100.0,
                envelope_sigma: 0.45,
                num_harmonics: 12,
                harmonic_decay: 0.75,
            },
            iota: IotaParams {
                K: 256,
                harmonic_coeffs: vec![1.0; 256],
                phase_shifts: vec![0.0; 256],
                global_amplitude: 1.0,
                frequency_range: 2.0,
            },
            tau: TauParams {
                normalization_epsilon: 1e-6,
                projection_strength: 0.8,
                noise_threshold: 0.01,
                use_template_normalization: true,
            },
            epsilon: EpsilonParams {
                energy_weight: 1.0,
                coherence_weight: 100.0,
                sparsity_weight: 10.0,
                quality_weight: 1.0,
                reduction_factor: 8,
                coherence_threshold: 0.8,
            },
        }
    }
}

impl Default for ZeroObject {
    fn default() -> Self {
        Self {
            theta: ModelParameters::default(),
            iteration: 0,
            loss: f32::INFINITY,
            metadata: HashMap::new(),
        }
    }
}

impl Default for InfiniteAspect {
    fn default() -> Self {
        Self {
            history: Vec::new(),
            metrics_history: Vec::new(),
            pareto_frontier: Vec::new(),
            gp_hyperparams: GaussianProcessHyperparams {
                lengthscale: vec![0.1; 20],
                signal_variance: 1.0,
                noise_variance: 0.01,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn test_zero_object_creation() {
        let zero = ZeroObject::default();
        assert_eq!(zero.iteration, 0);
        assert_eq!(zero.loss, f32::INFINITY);
        assert_eq!(zero.theta.gamma.num_harmonics, 12);
    }

    #[test]
    fn test_eta_morphism() {
        let zero = ZeroObject::default();
        let empty = eta(&zero);

        assert_eq!(empty.gp_mean.gamma.num_harmonics, 12);
        assert_eq!(empty.exploration_weight, 1.0); // iteration 0
    }

    #[test]
    fn test_sample_from_empty() {
        let zero = ZeroObject::default();
        let empty = eta(&zero);
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        let sampled = sample_from_empty(&empty, &mut rng);

        // Should be different from mean (with high probability)
        assert_ne!(sampled.gamma.base_frequency, empty.gp_mean.gamma.base_frequency);
    }

    #[test]
    fn test_infinite_update() {
        let mut infinity = InfiniteAspect::default();
        let theta = ModelParameters::default();
        let metrics = EvaluationMetrics {
            energy: 100.0,
            coherence: 0.9,
            sparsity: 0.1,
            quality: 85.0,
            factorization_loss: 2.5,
        };

        update_infinite(&mut infinity, theta, metrics);

        assert_eq!(infinity.history.len(), 1);
        assert_eq!(infinity.metrics_history.len(), 1);
    }

    #[test]
    fn test_zeta_morphism() {
        let mut infinity = InfiniteAspect::default();

        // Add some evaluations
        let theta1 = ModelParameters::default();
        let metrics1 = EvaluationMetrics {
            energy: 100.0,
            coherence: 0.8,
            sparsity: 0.1,
            quality: 80.0,
            factorization_loss: 5.0,
        };
        update_infinite(&mut infinity, theta1, metrics1);

        let mut theta2 = ModelParameters::default();
        theta2.gamma.amplitude = 150.0;
        let metrics2 = EvaluationMetrics {
            energy: 120.0,
            coherence: 0.95,
            sparsity: 0.05,
            quality: 95.0,
            factorization_loss: 1.5,
        };
        update_infinite(&mut infinity, theta2.clone(), metrics2);

        let new_zero = zeta(&infinity);

        assert_eq!(new_zero.iteration, 2);
        assert_eq!(new_zero.theta.gamma.amplitude, 150.0); // Should select better config
        assert!(new_zero.loss < 5.0); // Should have lower loss
    }

    #[test]
    fn test_pareto_dominance() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![2.0, 3.0, 4.0];
        let c = vec![1.0, 2.0, 4.0];
        let d = vec![0.5, 2.5, 3.5];

        assert!(dominates(&a, &b)); // a dominates b (better in all objectives)
        assert!(!dominates(&b, &a)); // b doesn't dominate a
        assert!(dominates(&a, &c)); // a dominates c (equal in 1st two, better in 3rd)
        assert!(!dominates(&c, &a)); // c doesn't dominate a
        assert!(!dominates(&a, &d)); // a doesn't dominate d (better in some, worse in others)
        assert!(!dominates(&d, &a)); // d doesn't dominate a (non-dominated pair)
    }
}
