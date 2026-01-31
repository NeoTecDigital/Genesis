/**
 * Test program for Genesis batch pipeline
 */

#include "batch_pipeline.h"
#include <iostream>
#include <vector>
#include <cmath>

int main() {
    std::cout << "\n=== Genesis Batch Pipeline Test ===\n\n";

    // Initialize batch pipeline
    const uint32_t batch_size = 4;
    const uint32_t width = 512;
    const uint32_t height = 512;

    std::cout << "Initializing batch pipeline (batch=" << batch_size
              << ", " << width << "×" << height << ")...\n";

    GenesisBatchPipeline* pipeline = genesis_batch_pipeline_init(batch_size, width, height);
    if (!pipeline) {
        std::cerr << "❌ Failed to initialize batch pipeline\n";
        return 1;
    }

    std::cout << "✅ Pipeline initialized successfully\n\n";

    // Prepare test parameters
    std::vector<ModelParameters> params(batch_size);
    std::vector<EvaluationMetrics> metrics(batch_size);

    // Get default parameters and create variations
    for (uint32_t i = 0; i < batch_size; i++) {
        genesis_get_default_params(&params[i]);

        // Create variations for testing
        params[i].gamma_base_frequency = 2.0f + i * 0.5f;
        params[i].gamma_amplitude = 100.0f + i * 20.0f;
        params[i].gamma_num_harmonics = 8 + i * 2;

        params[i].iota_global_amplitude = 0.8f + i * 0.1f;
        params[i].tau_projection_strength = 0.7f + i * 0.05f;
        params[i].epsilon_coherence_weight = 100.0f - i * 10.0f;
    }

    std::cout << "Evaluating " << batch_size << " configurations...\n";

    // Run batch evaluation
    int result = genesis_batch_evaluate(pipeline, params.data(), metrics.data(), batch_size);
    if (result != 0) {
        std::cerr << "❌ Batch evaluation failed\n";
        genesis_batch_pipeline_free(pipeline);
        return 1;
    }

    std::cout << "✅ Batch evaluation completed\n\n";

    // Display results
    std::cout << "=== Results ===\n\n";
    for (uint32_t i = 0; i < batch_size; i++) {
        std::cout << "Configuration " << (i + 1) << ":\n";
        std::cout << "  Parameters:\n";
        std::cout << "    γ base frequency: " << params[i].gamma_base_frequency << " Hz\n";
        std::cout << "    γ amplitude: " << params[i].gamma_amplitude << "\n";
        std::cout << "    γ harmonics: " << params[i].gamma_num_harmonics << "\n";
        std::cout << "  Metrics:\n";
        std::cout << "    Energy: " << metrics[i].energy << "\n";
        std::cout << "    Coherence: " << metrics[i].coherence << "\n";
        std::cout << "    Sparsity: " << metrics[i].sparsity << "\n";
        std::cout << "    Quality: " << metrics[i].quality << "\n";
        std::cout << "    Factorization loss: " << metrics[i].factorization_loss << "\n";

        float loss = genesis_compute_loss(&metrics[i]);
        std::cout << "    Total loss: " << loss << "\n\n";
    }

    // Find best configuration
    float best_loss = genesis_compute_loss(&metrics[0]);
    uint32_t best_idx = 0;
    for (uint32_t i = 1; i < batch_size; i++) {
        float loss = genesis_compute_loss(&metrics[i]);
        if (loss < best_loss) {
            best_loss = loss;
            best_idx = i;
        }
    }

    std::cout << "=== Best Configuration ===\n";
    std::cout << "Index: " << (best_idx + 1) << "\n";
    std::cout << "Loss: " << best_loss << "\n";
    std::cout << "γ base frequency: " << params[best_idx].gamma_base_frequency << " Hz\n";
    std::cout << "γ amplitude: " << params[best_idx].gamma_amplitude << "\n";
    std::cout << "Quality: " << metrics[best_idx].quality << "\n\n";

    // Cleanup
    std::cout << "Cleaning up...\n";
    genesis_batch_pipeline_free(pipeline);
    std::cout << "✅ Done\n\n";

    return 0;
}