#pragma once

#include "../../../Nova/nova_compute_api.h"
#include "../batch_pipeline.h"
#include <vector>
#include <cstring>
#include <atomic>
#include <thread>
#include <chrono>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <iostream>

struct GenesisBatchPipeline {
    void* context;
    void* queue;
    uint32_t queue_family;
    void* cmd_pool;
    uint32_t batch_size;
    uint32_t width;
    uint32_t height;

    // Shaders
    void* gamma_shader;
    void* iota_shader;
    void* tau_shader;
    void* epsilon_shader;
    void* epsilon_reverse_shader;
    void* tau_reverse_shader;
    void* iota_reverse_shader;
    void* gamma_reverse_shader;
    void* image_copy_shader;

    // Descriptor Set Layouts
    void* gamma_desc_layout;
    void* iota_desc_layout;
    void* tau_desc_layout;
    void* epsilon_desc_layout;
    void* epsilon_reverse_desc_layout;
    void* tau_reverse_desc_layout;
    void* iota_reverse_desc_layout;
    void* gamma_reverse_desc_layout;
    void* image_copy_desc_layout;

    // Pipelines
    void* gamma_pipeline;
    void* iota_pipeline;
    void* tau_pipeline;
    void* epsilon_pipeline;
    void* epsilon_reverse_pipeline;
    void* tau_reverse_pipeline;
    void* iota_reverse_pipeline;
    void* gamma_reverse_pipeline;
    void* image_copy_pipeline;

    // Batch Resources
    std::vector<void*> cmd_buffers;
    std::vector<void*> proto_unity;
    std::vector<void*> instance;
    std::vector<void*> proto_recovered;
    std::vector<void*> evaluation;
    std::vector<void*> proto_unity_views;
    std::vector<void*> instance_views;
    std::vector<void*> proto_recovered_views;
    std::vector<void*> evaluation_views;
    std::vector<void*> gamma_params_buffers;
    std::vector<void*> iota_params_buffers;
    std::vector<void*> tau_params_buffers;
    std::vector<void*> epsilon_params_buffers;
    std::vector<void*> gamma_params_mapped;
    std::vector<void*> iota_params_mapped;
    std::vector<void*> tau_params_mapped;
    std::vector<void*> epsilon_params_mapped;
    std::vector<void*> gamma_desc_sets;
    std::vector<void*> iota_desc_sets;
    std::vector<void*> tau_desc_sets;
    std::vector<void*> epsilon_desc_sets;

    // Single execution resources
    void* gamma_desc_set_single;
    void* iota_desc_set_single;
    void* epsilon_desc_set_single;
    void* gamma_cmd_buffer;
    void* iota_cmd_buffer;
    void* epsilon_cmd_buffer;
    
    // Metrics buffer
    void* metrics_buffer;
    void* metrics_mapped;

    // Memory Pool
    std::atomic<uint32_t> proto_identity_lock;
    void* proto_identity_working;
    void* proto_identity_working_view;
    void* empty_state_image;
    void* empty_state_view;
    void* infinity_state_image;
    void* infinity_state_view;
    void* gamma_filter_image;
    void* gamma_filter_view;
    void* iota_filter_image;
    void* iota_filter_view;
    void* tau_filter_image;
    void* tau_filter_view;
    void* epsilon_filter_image;
    void* epsilon_filter_view;
    std::vector<void*> training_inputs;
    std::vector<void*> training_input_views;

    uint32_t memory_capacity;
    uint32_t memory_count;
    std::vector<void*> proto_identity_memory;
    std::vector<void*> proto_identity_memory_views;
};
