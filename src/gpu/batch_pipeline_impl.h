/**
 * Internal implementation header for batch pipeline
 * 
 * Contains the GenesisBatchPipeline struct definition.
 * This is separate from the public API header to allow splitting
 * the implementation across multiple .cpp files.
 */

#ifndef GENESIS_BATCH_PIPELINE_IMPL_H
#define GENESIS_BATCH_PIPELINE_IMPL_H

#include <vector>
#include <cstdint>
#include <atomic>

struct GenesisBatchPipeline {
    // Nova context
    void* context;
    void* queue;
    uint32_t queue_family;
    void* cmd_pool;

    // Batch configuration
    uint32_t batch_size;
    uint32_t width;
    uint32_t height;

    // Shaders (loaded once) - Forward
    void* gamma_shader;
    void* iota_shader;
    void* tau_shader;
    void* epsilon_shader;
    
    // Shaders (loaded once) - Reverse
    void* epsilon_reverse_shader;  // ε⁻¹
    void* tau_reverse_shader;      // τ⁻¹
    void* iota_reverse_shader;     // ι⁻¹
    void* gamma_reverse_shader;    // γ⁻¹
    
    // Utility shaders
    void* image_copy_shader;       // Image-to-image copy for memory operations

    // Pipelines (created once) - Forward
    void* gamma_pipeline;
    void* iota_pipeline;
    void* tau_pipeline;
    void* epsilon_pipeline;
    
    // Pipelines (created once) - Reverse
    void* epsilon_reverse_pipeline;
    void* tau_reverse_pipeline;
    void* iota_reverse_pipeline;
    void* gamma_reverse_pipeline;
    
    // Utility pipelines
    void* image_copy_pipeline;

    // Descriptor set layouts - Forward
    void* gamma_desc_layout;
    void* iota_desc_layout;
    void* tau_desc_layout;
    void* epsilon_desc_layout;
    
    // Descriptor set layouts - Reverse
    void* epsilon_reverse_desc_layout;
    void* tau_reverse_desc_layout;
    void* iota_reverse_desc_layout;
    void* gamma_reverse_desc_layout;
    
    // Utility descriptor set layouts
    void* image_copy_desc_layout;

    // GPU images (batch_size of each) - STAY ON GPU
    std::vector<void*> proto_unity;       // γ output, ι input
    std::vector<void*> instance;          // ι output, τ input
    std::vector<void*> proto_recovered;   // τ output, ε input
    std::vector<void*> evaluation;        // ε output (64×64)

    // Image views
    std::vector<void*> proto_unity_views;
    std::vector<void*> instance_views;
    std::vector<void*> proto_recovered_views;
    std::vector<void*> evaluation_views;

    // Parameter buffers (batch_size) - CPU→GPU transfer
    std::vector<void*> gamma_params_buffers;
    std::vector<void*> iota_params_buffers;
    std::vector<void*> tau_params_buffers;
    std::vector<void*> epsilon_params_buffers;

    // Persistent mappings (write params here)
    std::vector<void*> gamma_params_mapped;
    std::vector<void*> iota_params_mapped;
    std::vector<void*> tau_params_mapped;
    std::vector<void*> epsilon_params_mapped;

    // Metrics download buffer - GPU→CPU transfer
    void* metrics_buffer;
    void* metrics_mapped;

    // Descriptor sets (batch_size of each)
    std::vector<void*> gamma_desc_sets;
    std::vector<void*> iota_desc_sets;
    std::vector<void*> tau_desc_sets;
    std::vector<void*> epsilon_desc_sets;

    // Command buffers (batch_size) - pre-recorded
    std::vector<void*> cmd_buffers;

    // NEW ARCHITECTURE: Memory Pool Model
    // MEMORY: Collection of proto-identities [𝟙₁, 𝟙₂, ..., 𝟙ₙ] - THE ACTUAL MEMORY
    std::vector<void*> proto_identity_memory;        // [𝟙₁, 𝟙₂, ..., 𝟙ₙ] - Memory pool
    std::vector<void*> proto_identity_memory_views;   // Views for each memory
    uint32_t memory_capacity;                         // Max number of memories
    uint32_t memory_count;                            // Current number of memories
    
    // WORKING BUFFER: Temporary proto-identity for current operation
    void* proto_identity_working;                     // Temporary 𝟙 for current morphism
    void* proto_identity_working_view;
    
    // Proto-identity synchronization (for working buffer)
    // Using atomic flag for simple gate mechanism
    // 0 = available, 1 = locked
    std::atomic<uint32_t> proto_identity_lock;
    
    // Memory states (shared, persistent)
    void* empty_state_image;         // ∅: Empty memory state (for γ operations)
    void* empty_state_view;
    void* infinity_state_image;      // ∞: Evaluation memory state (for ε operations)
    void* infinity_state_view;
    
    // Training data input images (batch_size)
    std::vector<void*> training_inputs;  // Input waveforms nᵢ
    std::vector<void*> training_input_views;
    
    // Per-morphism filter images: "How it thinks"
    void* gamma_filter_image;        // γ filter (how genesis thinks)
    void* iota_filter_image;          // ι filter (how instantiation thinks)
    void* tau_filter_image;           // τ filter (how encoding thinks)
    void* epsilon_filter_image;       // ε filter (how evaluation thinks)
    
    void* gamma_filter_view;
    void* iota_filter_view;
    void* tau_filter_view;
    void* epsilon_filter_view;
    
    // Single descriptor sets for individual operations (using slot 0)
    void* gamma_desc_set_single;      // For individual γ execution
    void* iota_desc_set_single;      // For individual ι execution
    void* epsilon_desc_set_single;    // For individual ε execution
    void* gamma_cmd_buffer;
    void* iota_cmd_buffer;
    void* epsilon_cmd_buffer;
};

#endif // GENESIS_BATCH_PIPELINE_IMPL_H

