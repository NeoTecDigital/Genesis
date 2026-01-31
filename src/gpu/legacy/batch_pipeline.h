/**
 * GPU-Resident Batch Pipeline for Genesis
 *
 * NEW ARCHITECTURE (Sprint 4):
 * - Batch over training data {n₁, n₂, ..., n_batch}, not γ parameters
 * - Execute γ ONCE to produce ONE persistent proto-unity
 * - Execute τ shader N times in parallel on training data
 * - Download proto-unity embeddings for clustering
 *
 * Optimizations:
 * - All intermediate results stay on GPU (no transfers)
 * - Batch encoding (16 training examples in parallel)
 * - Persistent proto-unity (shared across batch)
 * - One-time layout initialization (prevents freezes)
 *
 * Performance:
 * - ~0.5ms per encoding (vs 10ms baseline)
 * - 20× speedup for batch processing
 */

#ifndef GENESIS_BATCH_PIPELINE_H
#define GENESIS_BATCH_PIPELINE_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// NEW: Training data structure (input for batch encoding)
typedef struct {
    float* waveform;      // 512×512×4 RGBA32F frequency domain data
    uint32_t width;      // Image width (default 512)
    uint32_t height;     // Image height (default 512)
} TrainingData;

// NEW: Proto-unity embedding (output from τ shader)
typedef struct {
    float* embedding;    // 512×512×4 RGBA32F proto-unity space
    uint32_t width;
    uint32_t height;
} ProtoEmbedding;

// NEW: Learned γ parameters (separate from batch, used once per epoch)
typedef struct {
    float base_frequency;
    float initial_phase;
    float amplitude;
    float envelope_sigma;
    uint32_t num_harmonics;
    float harmonic_decay;
    uint32_t _pad[2];
} GammaParams;

// NEW: Learned τ parameters (separate from batch, shared across batch)
typedef struct {
    float normalization_epsilon;
    float projection_strength;
    float noise_threshold;
    uint32_t use_template_normalization;
    uint32_t _pad[4];
} TauParams;

// NEW: ι (instantiation) parameters
typedef struct {
    float harmonic_coeffs[10];  // First 10 harmonic coefficients
    float global_amplitude;
    float frequency_range;
    uint32_t _pad[1];
} IotaParams;

// NEW: ε (evaluation) parameters
typedef struct {
    float energy_weight;
    float coherence_weight;
    float sparsity_weight;
    float quality_weight;
    uint32_t reduction_factor;  // Default 8 (512→64)
    float coherence_threshold;
    uint32_t _pad[2];
} EpsilonParams;

// DEPRECATED: Old ModelParameters (kept for backward compatibility)
// Use GammaParams and TauParams instead
typedef struct {
    // γ (genesis) parameters
    float gamma_base_frequency;
    float gamma_initial_phase;
    float gamma_amplitude;
    float gamma_envelope_sigma;
    uint32_t gamma_num_harmonics;
    float gamma_harmonic_decay;
    uint32_t _gamma_pad[2];

    // ι (instantiation) parameters (first 10 coefficients)
    float iota_harmonic_coeffs[10];
    float iota_global_amplitude;
    float iota_frequency_range;

    // τ (encoder) parameters
    float tau_normalization_epsilon;
    float tau_projection_strength;
    float tau_noise_threshold;
    uint32_t tau_use_template_normalization;

    // ε (collapse) parameters
    float epsilon_energy_weight;
    float epsilon_coherence_weight;
    float epsilon_sparsity_weight;
    float epsilon_quality_weight;
    uint32_t epsilon_reduction_factor;
    float epsilon_coherence_threshold;
    uint32_t _epsilon_pad[2];
} ModelParameters;

// DEPRECATED: Evaluation metrics (kept for backward compatibility)
typedef struct {
    float energy;
    float coherence;
    float sparsity;
    float quality;
    float factorization_loss;  // ||𝟙 - 𝟙'|| RMSE
} EvaluationMetrics;

// Opaque handle to batch pipeline
typedef struct GenesisBatchPipeline GenesisBatchPipeline;

/**
 * Initialize batch pipeline (one-time setup)
 *
 * NEW ARCHITECTURE:
 * - Creates ONE persistent proto_unity image (not batch_size)
 * - Creates batch_size training input images
 * - Creates batch_size proto_embedding images
 * - One-time layout initialization (no repeated transitions)
 *
 * @param batch_size Number of training examples to encode in parallel (recommend 16)
 * @param width Image width (default 512)
 * @param height Image height (default 512)
 * @return Pipeline handle, or NULL on error
 */
GenesisBatchPipeline* genesis_batch_pipeline_init(
    uint32_t batch_size,
    uint32_t width,
    uint32_t height
);

/**
 * Execute γ morphism ONCE to produce persistent proto-unity
 *
 * This should be called:
 * - Once per training epoch (after γ params updated)
 * - Before any batch_encode calls
 *
 * @param pipeline Pipeline handle
 * @param gamma_params Learned γ parameters (ONE set)
 * @return 0 on success, -1 on error
 */
int genesis_execute_gamma_once(
    GenesisBatchPipeline* pipeline,
    const GammaParams* gamma_params
);

/**
 * Download working buffer proto-identity to CPU
 *
 * Downloads the current proto-identity from working buffer to CPU memory.
 *
 * @param pipeline Pipeline handle
 * @param output Output buffer (width × height × 4 RGBA32F, caller allocates)
 * @return 0 on success, -1 on error
 */
int genesis_download_working_buffer(GenesisBatchPipeline* pipeline, float* output);

/**
 * Download filter image to CPU
 *
 * Downloads a morphism filter image to CPU memory.
 *
 * @param pipeline Pipeline handle
 * @param filter_type 0=gamma, 1=iota, 2=tau, 3=epsilon
 * @param output Output buffer (width × height × 4 RGBA32F, caller allocates)
 * @return 0 on success, -1 on error
 */
int genesis_download_filter(GenesisBatchPipeline* pipeline, uint32_t filter_type, float* output);

/**
 * Upload instance data (n) to GPU
 *
 * Uploads custom instance data to the GPU pipeline for use in morphism operations.
 * This allows you to set custom input data before running operations like τ (encoding).
 *
 * @param pipeline Pipeline handle
 * @param instance_index Instance slot index (0 to batch_size-1)
 * @param instance_data Input data (width × height × 4 RGBA32F)
 * @return 0 on success, -1 on error
 */
int genesis_upload_instance(GenesisBatchPipeline* pipeline, uint32_t instance_index, const float* instance_data);

/**
 * Download instance data (n) from GPU
 *
 * Downloads instance data from the GPU pipeline to CPU memory.
 *
 * @param pipeline Pipeline handle
 * @param instance_index Instance slot index (0 to batch_size-1)
 * @param output Output buffer (width × height × 4 RGBA32F, caller allocates)
 * @return 0 on success, -1 on error
 */
int genesis_download_instance(GenesisBatchPipeline* pipeline, uint32_t instance_index, float* output);

/**
 * Encode training data batch via τ shader
 *
 * Process:
 * 1. Upload training data waveforms to GPU
 * 2. Execute τ shader in parallel: each nᵢ → 𝟙'ᵢ
 * 3. Download proto-unity embeddings to CPU
 *
 * @param pipeline Pipeline handle
 * @param training_data Array of training data (length = count)
 * @param proto_embeddings Output array for embeddings (length = count)
 * @param tau_params Learned τ parameters (ONE set, shared)
 * @param count Number of training examples (≤ batch_size)
 * @return 0 on success, -1 on error
 */
int genesis_batch_encode(
    GenesisBatchPipeline* pipeline,
    const TrainingData* training_data,
    ProtoEmbedding* proto_embeddings,
    const TauParams* tau_params,
    uint32_t count
);

/**
 * Execute ι morphism: 𝟙 → n (instantiation)
 *
 * Executes instantiation shader on a single proto-unity to produce an instance.
 * Uses slot 0 of the batch pipeline.
 *
 * @param pipeline Pipeline handle
 * @param iota_params ι parameters
 * @param output Output array (width × height × 4 RGBA32F)
 * @return 0 on success, -1 on error
 */
int genesis_execute_iota_once(
    GenesisBatchPipeline* pipeline,
    const IotaParams* iota_params,
    float* output
);

/**
 * Execute τ morphism: n → 𝟙' (encoding)
 *
 * Executes the encoding shader on a single instance to produce a proto-identity.
 * Uses slot 0 of the batch pipeline. The result is stored in the internal
 * proto_recovered[0] image.
 *
 * @param pipeline Pipeline handle
 * @param tau_params τ parameters
 * @param instance_index The index of the instance to use as input.
 * @return 0 on success, -1 on error
 */
int genesis_execute_tau_once(
    GenesisBatchPipeline* pipeline,
    const TauParams* tau_params,
    uint32_t instance_index
);

/**
 * Execute ε morphism: 𝟙 → ∞ (evaluation)
 *
 * Executes evaluation shader on a single proto-unity to produce evaluation metrics.
 * Uses slot 0 of the batch pipeline.
 *
 * @param pipeline Pipeline handle
 * @param epsilon_params ε parameters
 * @param output Output array (64 × 64 × 4 RGBA32F, reduced resolution)
 * @return 0 on success, -1 on error
 */
int genesis_execute_epsilon_once(
    GenesisBatchPipeline* pipeline,
    const EpsilonParams* epsilon_params,
    float* output
);

/**
 * Execute ε⁻¹ morphism: ∞ → 𝟙 (reverse evaluation)
 *
 * Expands evaluation back to proto-unity space.
 * Uses slot 0 of the batch pipeline.
 *
 * @param pipeline Pipeline handle
 * @param epsilon_params ε parameters (for reverse operation)
 * @param input Input array (64 × 64 × 4 RGBA32F, evaluation space)
 * @param output Output array (width × height × 4 RGBA32F, proto-unity space)
 * @return 0 on success, -1 on error
 */
int genesis_execute_epsilon_reverse(
    GenesisBatchPipeline* pipeline,
    const EpsilonParams* epsilon_params,
    const float* input,
    float* output
);

/**
 * Execute τ⁻¹ morphism: 𝟙 → n (reverse encoding)
 *
 * Expands proto-unity to instance space (reverse of encoding).
 * Uses slot 0 of the batch pipeline.
 *
 * @param pipeline Pipeline handle
 * @param tau_params τ parameters (for reverse operation)
 * @param input Input array (width × height × 4 RGBA32F, proto-unity space)
 * @param output Output array (width × height × 4 RGBA32F, instance space)
 * @return 0 on success, -1 on error
 */
int genesis_execute_tau_reverse(
    GenesisBatchPipeline* pipeline,
    const TauParams* tau_params,
    const float* input,
    float* output
);

/**
 * Execute ι⁻¹ morphism: n → 𝟙 (reverse instantiation)
 *
 * Collapses instance back to proto-unity space (reverse of instantiation).
 * Uses slot 0 of the batch pipeline.
 *
 * @param pipeline Pipeline handle
 * @param iota_params ι parameters (for reverse operation)
 * @param input Input array (width × height × 4 RGBA32F, instance space)
 * @param output Output array (width × height × 4 RGBA32F, proto-unity space)
 * @return 0 on success, -1 on error
 */
int genesis_execute_iota_reverse(
    GenesisBatchPipeline* pipeline,
    const IotaParams* iota_params,
    const float* input,
    float* output
);

/**
 * Execute γ⁻¹ morphism: 𝟙 → ∅ (reverse genesis)
 *
 * Dissolves proto-unity back to empty space (reverse of genesis).
 * Uses slot 0 of the batch pipeline.
 *
 * @param pipeline Pipeline handle
 * @param gamma_params γ parameters (for reverse operation)
 * @param input Input array (width × height × 4 RGBA32F, proto-unity space)
 * @param output Output array (width × height × 4 RGBA32F, empty space)
 * @return 0 on success, -1 on error
 */
int genesis_execute_gamma_reverse(
    GenesisBatchPipeline* pipeline,
    const GammaParams* gamma_params,
    const float* input,
    float* output
);

/**
 * DEPRECATED: Evaluate a batch of configurations on GPU
 *
 * ⚠️  WARNING: This function batches over γ parameters (WRONG approach).
 * Use genesis_execute_gamma_once() + genesis_batch_encode() instead.
 *
 * Kept for backward compatibility only.
 *
 * @param pipeline Pipeline handle
 * @param params Array of parameters (length = batch_size)
 * @param metrics Output array for metrics (length = batch_size)
 * @param count Number of configs to evaluate (≤ batch_size)
 * @return 0 on success, -1 on error
 */
int genesis_batch_evaluate(
    GenesisBatchPipeline* pipeline,
    const ModelParameters* params,
    EvaluationMetrics* metrics,
    uint32_t count
) __attribute__((deprecated("Use genesis_execute_gamma_once + genesis_batch_encode instead")));

/**
 * Get default model parameters
 *
 * @param params Output parameter struct
 */
void genesis_get_default_params(ModelParameters* params);

/**
 * Compute loss from evaluation metrics
 *
 * @param metrics Evaluation metrics
 * @return Scalar loss value
 */
float genesis_compute_loss(const EvaluationMetrics* metrics);

/**
 * Memory Pool Operations
 * 
 * The memory pool is a tensor of proto-identities [𝟙₁, 𝟙₂, ..., 𝟙ₙ]
 * representing learned structures discovered through clustering.
 */

/**
 * Add proto-identity to memory pool
 *
 * Copies the working buffer proto-identity to the memory pool.
 * If memory pool is full, returns error (prune first).
 *
 * @param pipeline Pipeline handle
 * @return Memory index (0-based) on success, -1 on error
 */
int genesis_memory_add(GenesisBatchPipeline* pipeline);

/**
 * Find coherent cluster of proto-identities in memory pool
 *
 * Correlates the working buffer against all proto-identities in memory
 * and finds all memories above the coherence threshold, forming a cluster.
 *
 * @param pipeline Pipeline handle
 * @param coherence_threshold Minimum coherence (0-1) to include in cluster
 * @param cluster_indices Output: Array of memory indices in cluster (caller allocates, max size = memory_capacity)
 * @param cluster_coherences Output: Array of coherence scores for each cluster member
 * @param max_cluster_size Maximum number of indices to return
 * @return Number of memories in cluster (0 if none found), or -1 on error
 */
int genesis_memory_find_coherent_cluster(
    GenesisBatchPipeline* pipeline,
    float coherence_threshold,
    uint32_t* cluster_indices,
    float* cluster_coherences,
    uint32_t max_cluster_size
);

/**
 * Average coherent cluster to form new proto-identity
 *
 * Takes a cluster of proto-identities and averages them to form a new
 * coherent proto-identity in the working buffer.
 *
 * @param pipeline Pipeline handle
 * @param cluster_indices Array of memory indices to average
 * @param cluster_coherences Array of coherence scores (weights for weighted average)
 * @param cluster_size Number of memories in cluster
 * @return 0 on success, -1 on error
 */
int genesis_memory_average_cluster(
    GenesisBatchPipeline* pipeline,
    const uint32_t* cluster_indices,
    const float* cluster_coherences,
    uint32_t cluster_size
);

/**
 * Select proto-identity from memory pool to working buffer
 *
 * Copies a proto-identity from memory pool to working buffer for use in operations.
 *
 * @param pipeline Pipeline handle
 * @param memory_index Index in memory pool (0-based)
 * @return 0 on success, -1 on error (invalid index)
 */
int genesis_memory_select(GenesisBatchPipeline* pipeline, uint32_t memory_index);

/**
 * Update existing proto-identity in memory pool
 *
 * Updates a proto-identity in memory with the current working buffer.
 * Used for refining learned structures through learning.
 *
 * @param pipeline Pipeline handle
 * @param memory_index Index in memory pool (0-based)
 * @return 0 on success, -1 on error (invalid index)
 */
int genesis_memory_update(GenesisBatchPipeline* pipeline, uint32_t memory_index);

/**
 * Prune rarely-used memories from memory pool
 *
 * Removes proto-identities that haven't been accessed recently.
 * Frees up space for new memories.
 *
 * @param pipeline Pipeline handle
 * @param min_usage_threshold Minimum usage threshold (0-1)
 * @return Number of memories pruned, or -1 on error
 */
int genesis_memory_prune(GenesisBatchPipeline* pipeline, float min_usage_threshold);

/**
 * Get memory pool statistics
 *
 * @param pipeline Pipeline handle
 * @param count Output: Current number of memories
 * @param capacity Output: Maximum capacity
 * @return 0 on success, -1 on error
 */
int genesis_memory_get_stats(GenesisBatchPipeline* pipeline, uint32_t* count, uint32_t* capacity);

/**
 * Download memory pool image to CPU
 *
 * Downloads a specific proto-identity from memory pool to CPU memory.
 *
 * @param pipeline Pipeline handle
 * @param memory_index Index in memory pool (0-based)
 * @param output Output buffer (width × height × 4 RGBA32F, caller allocates)
 * @return 0 on success, -1 on error
 */
int genesis_memory_download(GenesisBatchPipeline* pipeline, uint32_t memory_index, float* output);

/**
 * Find coherence clusters and save all memory images
 *
 * Finds all coherent clusters in memory pool and saves all memory images to disk.
 * This is a convenience function that combines finding clusters and saving images.
 *
 * @param pipeline Pipeline handle
 * @param coherence_threshold Minimum coherence (0-1) to include in cluster
 * @param output_dir Output directory for saved images (must exist)
 * @param cluster_indices Output: Array of cluster indices found (caller allocates)
 * @param cluster_coherences Output: Array of coherence scores (caller allocates)
 * @param max_cluster_size Maximum cluster size
 * @return Number of clusters found, or -1 on error
 */
int genesis_memory_find_coherence_and_save(
    GenesisBatchPipeline* pipeline,
    float coherence_threshold,
    const char* output_dir,
    uint32_t* cluster_indices,
    float* cluster_coherences,
    uint32_t max_cluster_size
);

/**
 * Free batch pipeline resources
 *
 * @param pipeline Pipeline handle
 */
void genesis_batch_pipeline_free(GenesisBatchPipeline* pipeline);

#ifdef __cplusplus
}
#endif

#endif // GENESIS_BATCH_PIPELINE_H
