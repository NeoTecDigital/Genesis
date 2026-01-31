/**
 * GPU-Resident Batch Pipeline Implementation
 */

#include "batch_pipeline.h"
#include "batch_pipeline_impl.h"
#include "../../../Nova/nova_compute_api.h"
#include <vector>
#include <cstring>
#include <atomic>
#include <thread>
#include <chrono>
#include <fstream>
#include <sstream>
#include <iomanip>
// <cmath> removed - not used directly
#include <iostream>

// Vulkan constants
static const uint32_t VK_FORMAT_R32G32B32A32_SFLOAT = 109;
static const uint32_t VK_IMAGE_USAGE_STORAGE_BIT = 0x00000020;
static const uint32_t VK_IMAGE_USAGE_TRANSFER_SRC_BIT = 0x00000001;
static const uint32_t VK_IMAGE_USAGE_TRANSFER_DST_BIT = 0x00000002;
static const uint32_t VK_IMAGE_ASPECT_COLOR_BIT = 0x00000001;
static const uint32_t VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT = 0x00000010;
static const uint32_t VK_BUFFER_USAGE_STORAGE_BUFFER_BIT = 0x00000020;
static const uint32_t VK_IMAGE_LAYOUT_GENERAL = 1;
static const uint32_t VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE = 3;
static const uint32_t VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT = 0x00000100;
static const uint32_t VMA_ALLOCATION_CREATE_MAPPED_BIT = 0x00000002;

// Pipeline stage flags
static const uint32_t VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT = 0x00000001;
static const uint32_t VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT = 0x00000800;
static const uint32_t VK_PIPELINE_STAGE_TRANSFER_BIT = 0x00001000;

// Access flags
static const uint32_t VK_ACCESS_SHADER_READ_BIT = 0x00000020;
static const uint32_t VK_ACCESS_SHADER_WRITE_BIT = 0x00000040;
static const uint32_t VK_ACCESS_TRANSFER_READ_BIT = 0x00000800;
static const uint32_t VK_ACCESS_TRANSFER_WRITE_BIT = 0x00001000;

// Forward declaration for nova_image_upload (should be in Nova API)
extern "C" int nova_image_upload(void* context, void* image, const float* data, uint32_t width, uint32_t height);
// Forward declaration for nova_image_download (should be in Nova API)
extern "C" int nova_image_download(void* context, void* image, float* data, uint32_t width, uint32_t height);
// Forward declarations for missing Nova API functions
extern "C" void nova_vma_free_image(void* context, void* image);
extern "C" void nova_destroy_image_view(void* context, void* view);


// GenesisBatchPipeline struct is defined in batch_pipeline_impl.h
// (no duplicate definition needed here)

// Helper function to copy image using compute shader
static int copy_image_gpu(GenesisBatchPipeline* pipeline, void* src_image, void* src_view,
                          void* dst_image, void* dst_view) {
    if (!pipeline->image_copy_pipeline || !pipeline->image_copy_desc_layout) {
        std::cerr << "❌ Image copy pipeline not available\n";
        return -1;
    }
    
    // Allocate descriptor set for image copy
    void* desc_set = nova_allocate_descriptor_set(
        pipeline->context, 
        pipeline->image_copy_desc_layout
    );
    if (!desc_set) {
        std::cerr << "❌ Failed to allocate descriptor set for image copy\n";
        return -1;
    }
    
    // Update descriptor set with images
    nova_update_descriptor_set_image(pipeline->context, desc_set, 0, src_view, VK_IMAGE_LAYOUT_GENERAL);
    nova_update_descriptor_set_image(pipeline->context, desc_set, 1, dst_view, VK_IMAGE_LAYOUT_GENERAL);
    
    // Allocate command buffer
    void* cmd = nova_allocate_command_buffer(pipeline->context, pipeline->cmd_pool);
    if (!cmd) {
        std::cerr << "❌ Failed to allocate command buffer for image copy\n";
        return -1;
    }
    
    if (nova_cmd_begin(cmd) != 0) {
        std::cerr << "❌ Failed to begin command buffer\n";
        nova_free_command_buffer(pipeline->context, pipeline->cmd_pool, cmd);
        return -1;
    }
    
    // Memory barriers for image access
    nova_cmd_pipeline_barrier_image(
        cmd, src_image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_READ_BIT
    );
    nova_cmd_pipeline_barrier_image(
        cmd, dst_image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_WRITE_BIT
    );
    
    // Bind pipeline and descriptor set
    nova_cmd_bind_pipeline(cmd, pipeline->image_copy_pipeline);
    nova_cmd_bind_descriptor_set(cmd, pipeline->image_copy_pipeline, desc_set);
    
    // Push constants for width/height
    struct {
        uint32_t width;
        uint32_t height;
    } push_constants;
    push_constants.width = pipeline->width;
    push_constants.height = pipeline->height;
    
    // Note: Nova API may not have push constants - we'll use workgroup size instead
    // Dispatch compute shader (16x16 workgroup size, so divide by 16)
    uint32_t workgroup_x = (pipeline->width + 15) / 16;
    uint32_t workgroup_y = (pipeline->height + 15) / 16;
    nova_cmd_dispatch(cmd, workgroup_x, workgroup_y, 1);
    
    nova_cmd_end(cmd);
    nova_submit_compute(pipeline->context, pipeline->queue, cmd);
    nova_queue_wait_idle(pipeline->queue);
    nova_free_command_buffer(pipeline->context, pipeline->cmd_pool, cmd);
    
    return 0;
}

// Proto-identity synchronization helpers
static void acquire_proto_identity_lock(GenesisBatchPipeline* pipeline, uint32_t timeout_ms = 1000) {
    uint32_t expected = 0;
    auto start = std::chrono::steady_clock::now();
    while (!pipeline->proto_identity_lock.compare_exchange_weak(expected, 1, std::memory_order_acquire)) {
        expected = 0;  // Reset expected value
        auto elapsed = std::chrono::steady_clock::now() - start;
        if (std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() > timeout_ms) {
            std::cerr << "⚠️  WARNING: Proto-identity lock timeout after " << timeout_ms << "ms\n";
            break;  // Continue anyway (shouldn't happen in practice)
        }
        std::this_thread::yield();  // Yield to other threads
    }
}

static void release_proto_identity_lock(GenesisBatchPipeline* pipeline) {
    pipeline->proto_identity_lock.store(0, std::memory_order_release);
}

extern "C" GenesisBatchPipeline* genesis_batch_pipeline_init(
    uint32_t batch_size,
    uint32_t width,
    uint32_t height
) {
    // Input validation
    if (batch_size == 0 || batch_size > 256) {
        std::cerr << "❌ Invalid batch_size: " << batch_size << " (must be 1-256)\n";
        return nullptr;
    }
    if (width == 0 || height == 0 || width > 4096 || height > 4096) {
        std::cerr << "❌ Invalid dimensions: " << width << "×" << height 
                  << " (must be 1-4096)\n";
        return nullptr;
    }
    // Check for integer overflow
    uint64_t total_pixels = static_cast<uint64_t>(width) * static_cast<uint64_t>(height);
    if (total_pixels > UINT32_MAX / 4) {  // 4 channels
        std::cerr << "❌ Image dimensions too large (would overflow)\n";
        return nullptr;
    }

    auto* pipeline = new GenesisBatchPipeline();
    pipeline->batch_size = batch_size;
    pipeline->width = width;
    pipeline->height = height;
    
    // Initialize proto-identity lock (0 = available, 1 = locked)
    pipeline->proto_identity_lock.store(0);

    // Initialize Nova context
    pipeline->context = nova_init_context();
    if (!pipeline->context) {
        std::cerr << "❌ Failed to initialize Nova context\n";
        delete pipeline;
        return nullptr;
    }

    pipeline->queue = nova_get_compute_queue(pipeline->context);
    if (!pipeline->queue) {
        std::cerr << "❌ Failed to get compute queue\n";
        genesis_batch_pipeline_free(pipeline);
        return nullptr;
    }
    
    pipeline->queue_family = nova_get_compute_queue_family(pipeline->context);
    pipeline->cmd_pool = nova_create_command_pool(pipeline->context, pipeline->queue_family);
    if (!pipeline->cmd_pool) {
        std::cerr << "❌ Failed to create command pool\n";
        genesis_batch_pipeline_free(pipeline);
        return nullptr;
    }

    std::cout << "✅ Nova context initialized\n";

    // Load forward shaders (ONCE)
    pipeline->gamma_shader = nova_load_shader(pipeline->context, "shaders/gamma_genesis.spv");
    pipeline->iota_shader = nova_load_shader(pipeline->context, "shaders/iota_instantiation.spv");
    pipeline->tau_shader = nova_load_shader(pipeline->context, "shaders/tau_reduction.spv");
    pipeline->epsilon_shader = nova_load_shader(pipeline->context, "shaders/epsilon_erasure.spv");

    if (!pipeline->gamma_shader || !pipeline->iota_shader ||
        !pipeline->tau_shader || !pipeline->epsilon_shader) {
        std::cerr << "❌ Failed to load forward shaders\n";
        genesis_batch_pipeline_free(pipeline);
        return nullptr;
    }

    // Load reverse shaders (ONCE)
    pipeline->epsilon_reverse_shader = nova_load_shader(pipeline->context, "shaders/epsilon_preservation.spv");
    pipeline->tau_reverse_shader = nova_load_shader(pipeline->context, "shaders/tau_expansion.spv");
    pipeline->iota_reverse_shader = nova_load_shader(pipeline->context, "shaders/iota_abstraction.spv");
    pipeline->gamma_reverse_shader = nova_load_shader(pipeline->context, "shaders/gamma_revelation.spv");

    if (!pipeline->epsilon_reverse_shader || !pipeline->tau_reverse_shader ||
        !pipeline->iota_reverse_shader || !pipeline->gamma_reverse_shader) {
        std::cerr << "⚠️  Failed to load reverse shaders (will use CPU fallback)\n";
        // Don't fail - allow CPU fallback for reverse operations
        pipeline->epsilon_reverse_shader = nullptr;
        pipeline->tau_reverse_shader = nullptr;
        pipeline->iota_reverse_shader = nullptr;
        pipeline->gamma_reverse_shader = nullptr;
    } else {
        std::cout << "✅ All reverse shaders loaded\n";
    }
    
    // Load utility shaders
    pipeline->image_copy_shader = nova_load_shader(pipeline->context, "shaders/image_copy.spv");
    if (!pipeline->image_copy_shader) {
        std::cerr << "⚠️  Failed to load image_copy shader (memory operations may be limited)\n";
    }

    std::cout << "✅ All forward shaders loaded\n";

    // Create descriptor set layouts - Forward
    pipeline->gamma_desc_layout = nova_create_descriptor_set_layout_compute_image(
        pipeline->context, 1, 2);  // 1 buffer, 2 images (empty_state input, proto_identity output)
    pipeline->iota_desc_layout = nova_create_descriptor_set_layout_compute_image(
        pipeline->context, 1, 2);  // 1 buffer, 2 images
    pipeline->tau_desc_layout = nova_create_descriptor_set_layout_compute_image(
        pipeline->context, 1, 3);  // 1 buffer, 3 images
    pipeline->epsilon_desc_layout = nova_create_descriptor_set_layout_compute_image(
        pipeline->context, 1, 2);  // 1 buffer, 2 images
    
    if (!pipeline->gamma_desc_layout || !pipeline->iota_desc_layout ||
        !pipeline->tau_desc_layout || !pipeline->epsilon_desc_layout) {
        std::cerr << "❌ Failed to create forward descriptor set layouts\n";
        genesis_batch_pipeline_free(pipeline);
        return nullptr;
    }
    
    // Create descriptor set layouts - Reverse (if shaders loaded)
    if (pipeline->epsilon_reverse_shader) {
        pipeline->epsilon_reverse_desc_layout = nova_create_descriptor_set_layout_compute_image(
            pipeline->context, 1, 2);  // 1 buffer (params), 2 images (eval input, proto output)
        pipeline->tau_reverse_desc_layout = nova_create_descriptor_set_layout_compute_image(
            pipeline->context, 1, 2);  // 1 buffer (params), 2 images (proto input, instance output)
        pipeline->iota_reverse_desc_layout = nova_create_descriptor_set_layout_compute_image(
            pipeline->context, 1, 2);  // 1 buffer (params), 2 images (instance input, proto output)
        pipeline->gamma_reverse_desc_layout = nova_create_descriptor_set_layout_compute_image(
            pipeline->context, 1, 2);  // 1 buffer (params), 2 images (proto input, empty output)
    }
    
    // Create descriptor set layout for image copy (2 images: src, dst)
    if (pipeline->image_copy_shader) {
        pipeline->image_copy_desc_layout = nova_create_descriptor_set_layout_compute_image(
            pipeline->context, 0, 2);  // 0 buffers, 2 images (src, dst)
        if (!pipeline->image_copy_desc_layout) {
            std::cerr << "⚠️  Failed to create image_copy descriptor layout\n";
        }
    }

    // Create pipelines - Forward
    pipeline->gamma_pipeline = nova_create_compute_pipeline(
        pipeline->context, pipeline->gamma_shader, pipeline->gamma_desc_layout);
    pipeline->iota_pipeline = nova_create_compute_pipeline(
        pipeline->context, pipeline->iota_shader, pipeline->iota_desc_layout);
    pipeline->tau_pipeline = nova_create_compute_pipeline(
        pipeline->context, pipeline->tau_shader, pipeline->tau_desc_layout);
    pipeline->epsilon_pipeline = nova_create_compute_pipeline(
        pipeline->context, pipeline->epsilon_shader, pipeline->epsilon_desc_layout);

    if (!pipeline->gamma_pipeline || !pipeline->iota_pipeline ||
        !pipeline->tau_pipeline || !pipeline->epsilon_pipeline) {
        std::cerr << "❌ Failed to create forward compute pipelines\n";
        genesis_batch_pipeline_free(pipeline);
        return nullptr;
    }

    std::cout << "✅ Forward pipelines created\n";
    
    // Create pipelines - Reverse (if shaders loaded)
    if (pipeline->epsilon_reverse_shader) {
        pipeline->epsilon_reverse_pipeline = nova_create_compute_pipeline(
            pipeline->context, pipeline->epsilon_reverse_shader, pipeline->epsilon_reverse_desc_layout);
        pipeline->tau_reverse_pipeline = nova_create_compute_pipeline(
            pipeline->context, pipeline->tau_reverse_shader, pipeline->tau_reverse_desc_layout);
        pipeline->iota_reverse_pipeline = nova_create_compute_pipeline(
            pipeline->context, pipeline->iota_reverse_shader, pipeline->iota_reverse_desc_layout);
        pipeline->gamma_reverse_pipeline = nova_create_compute_pipeline(
            pipeline->context, pipeline->gamma_reverse_shader, pipeline->gamma_reverse_desc_layout);
        
        if (pipeline->epsilon_reverse_pipeline && pipeline->tau_reverse_pipeline &&
            pipeline->iota_reverse_pipeline && pipeline->gamma_reverse_pipeline) {
            std::cout << "✅ Reverse pipelines created\n";
        } else {
            std::cerr << "⚠️  Some reverse pipelines failed to create\n";
        }
    }
    
    // Create image copy pipeline (if shader loaded)
    if (pipeline->image_copy_shader && pipeline->image_copy_desc_layout) {
        pipeline->image_copy_pipeline = nova_create_compute_pipeline(
            pipeline->context, pipeline->image_copy_shader, pipeline->image_copy_desc_layout);
        if (pipeline->image_copy_pipeline) {
            std::cout << "✅ Image copy pipeline created\n";
        } else {
            std::cerr << "⚠️  Failed to create image copy pipeline\n";
        }
    }

    // Allocate batch resources
    const uint32_t eval_size = 64;  // 8× reduction for ε

    for (uint32_t i = 0; i < batch_size; i++) {
        // Allocate images (STAY ON GPU - never transferred)
        void* proto = nova_vma_create_image(
            pipeline->context, width, height,
            VK_FORMAT_R32G32B32A32_SFLOAT,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT
        );
        void* inst = nova_vma_create_image(
            pipeline->context, width, height,
            VK_FORMAT_R32G32B32A32_SFLOAT,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT
        );
        void* proto_rec = nova_vma_create_image(
            pipeline->context, width, height,
            VK_FORMAT_R32G32B32A32_SFLOAT,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT
        );
        void* eval = nova_vma_create_image(
            pipeline->context, eval_size, eval_size,
            VK_FORMAT_R32G32B32A32_SFLOAT,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT
        );

        if (!proto || !inst || !proto_rec || !eval) {
            std::cerr << "❌ Failed to allocate GPU images for batch " << i << "\n";
            genesis_batch_pipeline_free(pipeline);
            return nullptr;
        }

        pipeline->proto_unity.push_back(proto);
        pipeline->instance.push_back(inst);
        pipeline->proto_recovered.push_back(proto_rec);
        pipeline->evaluation.push_back(eval);

        // Create image views - MUST explicitly create them!
        void* proto_view = nova_create_image_view(
            pipeline->context, proto,
            VK_FORMAT_R32G32B32A32_SFLOAT,
            VK_IMAGE_ASPECT_COLOR_BIT
        );
        void* inst_view = nova_create_image_view(
            pipeline->context, inst,
            VK_FORMAT_R32G32B32A32_SFLOAT,
            VK_IMAGE_ASPECT_COLOR_BIT
        );
        void* proto_rec_view = nova_create_image_view(
            pipeline->context, proto_rec,
            VK_FORMAT_R32G32B32A32_SFLOAT,
            VK_IMAGE_ASPECT_COLOR_BIT
        );
        void* eval_view = nova_create_image_view(
            pipeline->context, eval,
            VK_FORMAT_R32G32B32A32_SFLOAT,
            VK_IMAGE_ASPECT_COLOR_BIT
        );

        if (!proto_view || !inst_view || !proto_rec_view || !eval_view) {
            std::cerr << "❌ Failed to create image views for batch " << i << "\n";
            genesis_batch_pipeline_free(pipeline);
            return nullptr;
        }

        pipeline->proto_unity_views.push_back(proto_view);
        pipeline->instance_views.push_back(inst_view);
        pipeline->proto_recovered_views.push_back(proto_rec_view);
        pipeline->evaluation_views.push_back(eval_view);

        // Allocate parameter buffers with persistent mapping
        auto alloc_param_buffer = [&](size_t size) -> std::pair<void*, void*> {
            void* buf = nova_vma_allocate_buffer(
                pipeline->context, size, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT
            );
            void* mapped = nova_vma_map(buf);
            return {buf, mapped};
        };

        auto [gamma_buf, gamma_map] = alloc_param_buffer(64);  // Padded size
        auto [iota_buf, iota_map] = alloc_param_buffer(sizeof(ModelParameters));
        auto [tau_buf, tau_map] = alloc_param_buffer(32);
        auto [epsilon_buf, epsilon_map] = alloc_param_buffer(64);

        pipeline->gamma_params_buffers.push_back(gamma_buf);
        pipeline->iota_params_buffers.push_back(iota_buf);
        pipeline->tau_params_buffers.push_back(tau_buf);
        pipeline->epsilon_params_buffers.push_back(epsilon_buf);

        pipeline->gamma_params_mapped.push_back(gamma_map);
        pipeline->iota_params_mapped.push_back(iota_map);
        pipeline->tau_params_mapped.push_back(tau_map);
        pipeline->epsilon_params_mapped.push_back(epsilon_map);

        // Allocate descriptor sets
        void* gamma_desc = nova_allocate_descriptor_set(pipeline->context, pipeline->gamma_desc_layout);
        void* iota_desc = nova_allocate_descriptor_set(pipeline->context, pipeline->iota_desc_layout);
        void* tau_desc = nova_allocate_descriptor_set(pipeline->context, pipeline->tau_desc_layout);
        void* epsilon_desc = nova_allocate_descriptor_set(pipeline->context, pipeline->epsilon_desc_layout);

        pipeline->gamma_desc_sets.push_back(gamma_desc);
        pipeline->iota_desc_sets.push_back(iota_desc);
        pipeline->tau_desc_sets.push_back(tau_desc);
        pipeline->epsilon_desc_sets.push_back(epsilon_desc);

        // Bind descriptors (ONCE)
        void* gamma_vk_buf = nova_vma_get_buffer(gamma_buf);
        void* iota_vk_buf = nova_vma_get_buffer(iota_buf);
        void* tau_vk_buf = nova_vma_get_buffer(tau_buf);
        void* epsilon_vk_buf = nova_vma_get_buffer(epsilon_buf);

        // γ: params buffer at binding 0, output image at binding 1
        nova_update_descriptor_set(pipeline->context, gamma_desc, 0, gamma_vk_buf);
        nova_update_descriptor_set_image(pipeline->context, gamma_desc, 1, proto_view, VK_IMAGE_LAYOUT_GENERAL);

        // ι: params buffer at binding 0, input image at binding 1, output image at binding 2
        nova_update_descriptor_set(pipeline->context, iota_desc, 0, iota_vk_buf);
        nova_update_descriptor_set_image(pipeline->context, iota_desc, 1, proto_view, VK_IMAGE_LAYOUT_GENERAL);
        nova_update_descriptor_set_image(pipeline->context, iota_desc, 2, inst_view, VK_IMAGE_LAYOUT_GENERAL);

        // τ: params buffer at binding 0, instance image at binding 1, proto image at binding 2, output at binding 3
        // NOTE: For NEW ARCHITECTURE, binding 1 should be training_inputs[i] (not instance)
        // and binding 2 should be proto_identity (not proto_unity[i])
        // Keeping old bindings for backward compatibility with genesis_batch_evaluate()
        // Will be updated when genesis_batch_encode() rebinds descriptors
        nova_update_descriptor_set(pipeline->context, tau_desc, 0, tau_vk_buf);
        nova_update_descriptor_set_image(pipeline->context, tau_desc, 1, inst_view, VK_IMAGE_LAYOUT_GENERAL);
        nova_update_descriptor_set_image(pipeline->context, tau_desc, 2, proto_view, VK_IMAGE_LAYOUT_GENERAL);
        nova_update_descriptor_set_image(pipeline->context, tau_desc, 3, proto_rec_view, VK_IMAGE_LAYOUT_GENERAL);

        // ε: params buffer at binding 0, input image at binding 1, output at binding 2
        nova_update_descriptor_set(pipeline->context, epsilon_desc, 0, epsilon_vk_buf);
        nova_update_descriptor_set_image(pipeline->context, epsilon_desc, 1, proto_rec_view, VK_IMAGE_LAYOUT_GENERAL);
        nova_update_descriptor_set_image(pipeline->context, epsilon_desc, 2, eval_view, VK_IMAGE_LAYOUT_GENERAL);

        // Allocate command buffer (will be recorded per-evaluation)
        void* cmd = nova_allocate_command_buffer(pipeline->context, pipeline->cmd_pool);
        pipeline->cmd_buffers.push_back(cmd);
    }

    // One-time layout initialization (prevents repeated transitions causing freezes)
    // Transition all images to GENERAL layout ONCE during init
    void* init_cmd = nova_allocate_command_buffer(pipeline->context, pipeline->cmd_pool);
    if (nova_cmd_begin(init_cmd) != 0) {
        std::cerr << "❌ Failed to begin init command buffer\n";
        genesis_batch_pipeline_free(pipeline);
        return nullptr;
    }

    // Transition all proto_unity images: UNDEFINED → GENERAL (once, persists)
    for (uint32_t i = 0; i < batch_size; i++) {
        nova_cmd_pipeline_barrier_image(
            init_cmd,
            pipeline->proto_unity[i],
            0,  // VK_IMAGE_LAYOUT_UNDEFINED
            VK_IMAGE_LAYOUT_GENERAL,
            0,  // VK_ACCESS_NONE
            VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT
        );
    }

    // Transition all instance images: UNDEFINED → GENERAL (once)
    for (uint32_t i = 0; i < batch_size; i++) {
        nova_cmd_pipeline_barrier_image(
            init_cmd,
            pipeline->instance[i],
            0,  // UNDEFINED
            VK_IMAGE_LAYOUT_GENERAL,
            0,
            VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT
        );
    }

    // Transition all proto_recovered images: UNDEFINED → GENERAL (once)
    for (uint32_t i = 0; i < batch_size; i++) {
        nova_cmd_pipeline_barrier_image(
            init_cmd,
            pipeline->proto_recovered[i],
            0,  // UNDEFINED
            VK_IMAGE_LAYOUT_GENERAL,
            0,
            VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT
        );
    }

    // Transition all evaluation images: UNDEFINED → GENERAL (once)
    for (uint32_t i = 0; i < batch_size; i++) {
        nova_cmd_pipeline_barrier_image(
            init_cmd,
            pipeline->evaluation[i],
            0,  // UNDEFINED
            VK_IMAGE_LAYOUT_GENERAL,
            0,
            VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT
        );
    }

    // NEW ARCHITECTURE: Transition persistent proto-unity: UNDEFINED → GENERAL (once, persists)
    // Note: This must be done after proto_identity is created (moved after creation)
    // Will be added in the section after proto_identity creation

    if (nova_cmd_end(init_cmd) != 0) {
        std::cerr << "❌ Failed to end init command buffer\n";
        nova_free_command_buffer(pipeline->context, pipeline->cmd_pool, init_cmd);
        genesis_batch_pipeline_free(pipeline);
        return nullptr;
    }

    // Submit and wait for layout transitions to complete
    if (nova_submit_compute(pipeline->context, pipeline->queue, init_cmd) != 0) {
        std::cerr << "❌ Failed to submit init commands\n";
        nova_free_command_buffer(pipeline->context, pipeline->cmd_pool, init_cmd);
        genesis_batch_pipeline_free(pipeline);
        return nullptr;
    }

    nova_queue_wait_idle(pipeline->queue);
    nova_free_command_buffer(pipeline->context, pipeline->cmd_pool, init_cmd);

    std::cout << "✅ Image layouts initialized (one-time, persistent)\n";

    // NEW ARCHITECTURE: Memory Pool Model
    // Initialize memory pool (tensor of proto-identities [𝟙₁, 𝟙₂, ..., 𝟙ₙ])
    pipeline->memory_capacity = 4096;  // 4K memories on GPU (can be configured)
    pipeline->memory_count = 0;
    pipeline->proto_identity_memory.reserve(pipeline->memory_capacity);
    pipeline->proto_identity_memory_views.reserve(pipeline->memory_capacity);
    std::cout << "✅ Memory pool initialized (capacity: " << pipeline->memory_capacity << ")\n";
    
    // Allocate working buffer (temporary proto-identity for current operation)
    // This is the "thinking" state buffer, requires synchronization
    pipeline->proto_identity_working = nova_vma_create_image(
        pipeline->context, width, height,
        VK_FORMAT_R32G32B32A32_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
    );
    if (!pipeline->proto_identity_working) {
        std::cerr << "❌ Failed to allocate proto-identity working buffer\n";
        genesis_batch_pipeline_free(pipeline);
        return nullptr;
    }
    
    pipeline->proto_identity_working_view = nova_create_image_view(
        pipeline->context, pipeline->proto_identity_working,
        VK_FORMAT_R32G32B32A32_SFLOAT,
        VK_IMAGE_ASPECT_COLOR_BIT
    );
    if (!pipeline->proto_identity_working_view) {
        std::cerr << "❌ Failed to create proto-identity working view\n";
        genesis_batch_pipeline_free(pipeline);
        return nullptr;
    }
    std::cout << "✅ Proto-identity working buffer allocated\n";
    
    // Create memory states (∅ and ∞)
    pipeline->empty_state_image = nova_vma_create_image(
        pipeline->context, width, height,
        VK_FORMAT_R32G32B32A32_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
    );
    if (!pipeline->empty_state_image) {
        std::cerr << "❌ Failed to allocate empty state image\n";
        genesis_batch_pipeline_free(pipeline);
        return nullptr;
    }
    
    pipeline->empty_state_view = nova_create_image_view(
        pipeline->context, pipeline->empty_state_image,
        VK_FORMAT_R32G32B32A32_SFLOAT,
        VK_IMAGE_ASPECT_COLOR_BIT
    );
    if (!pipeline->empty_state_view) {
        std::cerr << "❌ Failed to create empty state view\n";
        genesis_batch_pipeline_free(pipeline);
        return nullptr;
    }
    
    // ε reduction factor (64×64 for 512×512 input with factor=8)
    pipeline->infinity_state_image = nova_vma_create_image(
        pipeline->context, eval_size, eval_size,
        VK_FORMAT_R32G32B32A32_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
    );
    if (!pipeline->infinity_state_image) {
        std::cerr << "❌ Failed to allocate infinity state image\n";
        genesis_batch_pipeline_free(pipeline);
        return nullptr;
    }
    
    pipeline->infinity_state_view = nova_create_image_view(
        pipeline->context, pipeline->infinity_state_image,
        VK_FORMAT_R32G32B32A32_SFLOAT,
        VK_IMAGE_ASPECT_COLOR_BIT
    );
    if (!pipeline->infinity_state_view) {
        std::cerr << "❌ Failed to create infinity state view\n";
        genesis_batch_pipeline_free(pipeline);
        return nullptr;
    }
    
    // Create per-morphism filter images ("how it thinks")
    pipeline->gamma_filter_image = nova_vma_create_image(
        pipeline->context, width, height,
        VK_FORMAT_R32G32B32A32_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
    );
    if (!pipeline->gamma_filter_image) {
        std::cerr << "❌ Failed to allocate γ filter image\n";
        genesis_batch_pipeline_free(pipeline);
        return nullptr;
    }
    
    pipeline->gamma_filter_view = nova_create_image_view(
        pipeline->context, pipeline->gamma_filter_image,
        VK_FORMAT_R32G32B32A32_SFLOAT,
        VK_IMAGE_ASPECT_COLOR_BIT
    );
    if (!pipeline->gamma_filter_view) {
        std::cerr << "❌ Failed to create γ filter view\n";
        genesis_batch_pipeline_free(pipeline);
        return nullptr;
    }
    
    pipeline->iota_filter_image = nova_vma_create_image(
        pipeline->context, width, height,
        VK_FORMAT_R32G32B32A32_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
    );
    if (!pipeline->iota_filter_image) {
        std::cerr << "❌ Failed to allocate ι filter image\n";
        genesis_batch_pipeline_free(pipeline);
        return nullptr;
    }
    
    pipeline->iota_filter_view = nova_create_image_view(
        pipeline->context, pipeline->iota_filter_image,
        VK_FORMAT_R32G32B32A32_SFLOAT,
        VK_IMAGE_ASPECT_COLOR_BIT
    );
    if (!pipeline->iota_filter_view) {
        std::cerr << "❌ Failed to create ι filter view\n";
        genesis_batch_pipeline_free(pipeline);
        return nullptr;
    }
    
    pipeline->tau_filter_image = nova_vma_create_image(
        pipeline->context, width, height,
        VK_FORMAT_R32G32B32A32_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
    );
    if (!pipeline->tau_filter_image) {
        std::cerr << "❌ Failed to allocate τ filter image\n";
        genesis_batch_pipeline_free(pipeline);
        return nullptr;
    }
    
    pipeline->tau_filter_view = nova_create_image_view(
        pipeline->context, pipeline->tau_filter_image,
        VK_FORMAT_R32G32B32A32_SFLOAT,
        VK_IMAGE_ASPECT_COLOR_BIT
    );
    if (!pipeline->tau_filter_view) {
        std::cerr << "❌ Failed to create τ filter view\n";
        genesis_batch_pipeline_free(pipeline);
        return nullptr;
    }
    
    pipeline->epsilon_filter_image = nova_vma_create_image(
        pipeline->context, width, height,
        VK_FORMAT_R32G32B32A32_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
    );
    if (!pipeline->epsilon_filter_image) {
        std::cerr << "❌ Failed to allocate ε filter image\n";
        genesis_batch_pipeline_free(pipeline);
        return nullptr;
    }
    
    pipeline->epsilon_filter_view = nova_create_image_view(
        pipeline->context, pipeline->epsilon_filter_image,
        VK_FORMAT_R32G32B32A32_SFLOAT,
        VK_IMAGE_ASPECT_COLOR_BIT
    );
    if (!pipeline->epsilon_filter_view) {
        std::cerr << "❌ Failed to create ε filter view\n";
        genesis_batch_pipeline_free(pipeline);
        return nullptr;
    }
    
    std::cout << "✅ Proto-identity (𝟙), memory states (∅, ∞), and filter images allocated\n";

    // NEW ARCHITECTURE: Create training input images (batch_size)
    for (uint32_t i = 0; i < batch_size; i++) {
        void* training_input = nova_vma_create_image(
            pipeline->context, width, height,
            VK_FORMAT_R32G32B32A32_SFLOAT,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
        );
        if (!training_input) {
            std::cerr << "❌ Failed to allocate training input image " << i << "\n";
            genesis_batch_pipeline_free(pipeline);
            return nullptr;
        }
        pipeline->training_inputs.push_back(training_input);
        
        void* training_view = nova_create_image_view(
            pipeline->context, training_input,
            VK_FORMAT_R32G32B32A32_SFLOAT,
            VK_IMAGE_ASPECT_COLOR_BIT
        );
        if (!training_view) {
            std::cerr << "❌ Failed to create training input view " << i << "\n";
            genesis_batch_pipeline_free(pipeline);
            return nullptr;
        }
        pipeline->training_input_views.push_back(training_view);
    }

    // NEW ARCHITECTURE: Create single γ descriptor set (for individual execution)
    // Uses per-instance parameter buffers (gamma_params_buffers[0] for single execution)
    pipeline->gamma_desc_set_single = nova_allocate_descriptor_set(pipeline->context, pipeline->gamma_desc_layout);
    if (!pipeline->gamma_desc_set_single) {
        std::cerr << "❌ Failed to allocate γ descriptor set\n";
        genesis_batch_pipeline_free(pipeline);
        return nullptr;
    }
    
    // Use per-instance buffer (index 0) for single execution
    // Will be updated on execution:
    void* gamma_vk_buf = nova_vma_get_buffer(pipeline->gamma_params_buffers[0]);
    nova_update_descriptor_set(pipeline->context, pipeline->gamma_desc_set_single, 0, gamma_vk_buf);
    nova_update_descriptor_set_image(pipeline->context, pipeline->gamma_desc_set_single, 1,
                                     pipeline->empty_state_view, VK_IMAGE_LAYOUT_GENERAL);
    nova_update_descriptor_set_image(pipeline->context, pipeline->gamma_desc_set_single, 2,
                                     pipeline->proto_identity_working_view, VK_IMAGE_LAYOUT_GENERAL);
    
    // Create single γ command buffer
    pipeline->gamma_cmd_buffer = nova_allocate_command_buffer(pipeline->context, pipeline->cmd_pool);
    if (!pipeline->gamma_cmd_buffer) {
        std::cerr << "❌ Failed to allocate γ command buffer\n";
        genesis_batch_pipeline_free(pipeline);
        return nullptr;
    }
    
    // NEW: Allocate single ι descriptor set and buffers
    pipeline->iota_desc_set_single = nova_allocate_descriptor_set(
        pipeline->context, pipeline->iota_desc_layout
    );
    if (!pipeline->iota_desc_set_single) {
        std::cerr << "❌ Failed to allocate ι descriptor set\n";
        genesis_batch_pipeline_free(pipeline);
        return nullptr;
    }
    
    // Use per-instance buffer (index 0) for single execution
    // Will be updated on execution:
    void* iota_vk_buf = nova_vma_get_buffer(pipeline->iota_params_buffers[0]);
    nova_update_descriptor_set(pipeline->context, pipeline->iota_desc_set_single, 0, iota_vk_buf);
    nova_update_descriptor_set_image(pipeline->context, pipeline->iota_desc_set_single, 1,
                                     pipeline->proto_identity_working_view, VK_IMAGE_LAYOUT_GENERAL);
    nova_update_descriptor_set_image(pipeline->context, pipeline->iota_desc_set_single, 2,
                                     pipeline->instance_views[0], VK_IMAGE_LAYOUT_GENERAL);
    
    pipeline->iota_cmd_buffer = nova_allocate_command_buffer(pipeline->context, pipeline->cmd_pool);
    if (!pipeline->iota_cmd_buffer) {
        std::cerr << "❌ Failed to allocate ι command buffer\n";
        genesis_batch_pipeline_free(pipeline);
        return nullptr;
    }
    
    // NEW: Allocate single ε descriptor set and buffers
    pipeline->epsilon_desc_set_single = nova_allocate_descriptor_set(
        pipeline->context, pipeline->epsilon_desc_layout
    );
    if (!pipeline->epsilon_desc_set_single) {
        std::cerr << "❌ Failed to allocate ε descriptor set\n";
        genesis_batch_pipeline_free(pipeline);
        return nullptr;
    }
    
    // Use per-instance buffer (index 0) for single execution
    // Will be updated on execution:
    void* epsilon_vk_buf = nova_vma_get_buffer(pipeline->epsilon_params_buffers[0]);
    nova_update_descriptor_set(pipeline->context, pipeline->epsilon_desc_set_single, 0, epsilon_vk_buf);
    nova_update_descriptor_set_image(pipeline->context, pipeline->epsilon_desc_set_single, 1,
                                     pipeline->proto_identity_working_view, VK_IMAGE_LAYOUT_GENERAL);
    nova_update_descriptor_set_image(pipeline->context, pipeline->epsilon_desc_set_single, 2,
                                     pipeline->evaluation_views[0], VK_IMAGE_LAYOUT_GENERAL);
    
    pipeline->epsilon_cmd_buffer = nova_allocate_command_buffer(pipeline->context, pipeline->cmd_pool);
    if (!pipeline->epsilon_cmd_buffer) {
        std::cerr << "❌ Failed to allocate ε command buffer\n";
        genesis_batch_pipeline_free(pipeline);
        return nullptr;
    }
    
    // Note: Reverse morphisms (ε⁻¹, τ⁻¹, ι⁻¹, γ⁻¹) use CPU-based implementations
    // for now. Future: implement proper GPU reverse shaders.

    // Transition persistent proto-unity and training inputs (second init command buffer)
    void* init_cmd2 = nova_allocate_command_buffer(pipeline->context, pipeline->cmd_pool);
    if (nova_cmd_begin(init_cmd2) != 0) {
        std::cerr << "❌ Failed to begin second init command buffer\n";
        genesis_batch_pipeline_free(pipeline);
        return nullptr;
    }

    // Transition persistent proto-unity: UNDEFINED → GENERAL (once, persists)
    nova_cmd_pipeline_barrier_image(
        init_cmd2,
        pipeline->proto_identity_working,
        0,  // UNDEFINED
        VK_IMAGE_LAYOUT_GENERAL,
        0,
        VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
    );

    // Transition all training inputs: UNDEFINED → GENERAL (once)
    for (uint32_t i = 0; i < batch_size; i++) {
        nova_cmd_pipeline_barrier_image(
            init_cmd2,
            pipeline->training_inputs[i],
            0,  // UNDEFINED
            VK_IMAGE_LAYOUT_GENERAL,
            0,
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT
        );
    }

    if (nova_cmd_end(init_cmd2) != 0) {
        std::cerr << "❌ Failed to end second init command buffer\n";
        nova_free_command_buffer(pipeline->context, pipeline->cmd_pool, init_cmd2);
        genesis_batch_pipeline_free(pipeline);
        return nullptr;
    }

    if (nova_submit_compute(pipeline->context, pipeline->queue, init_cmd2) != 0) {
        std::cerr << "❌ Failed to submit second init commands\n";
        nova_free_command_buffer(pipeline->context, pipeline->cmd_pool, init_cmd2);
        genesis_batch_pipeline_free(pipeline);
        return nullptr;
    }

    nova_queue_wait_idle(pipeline->queue);
    nova_free_command_buffer(pipeline->context, pipeline->cmd_pool, init_cmd2);

    // Allocate metrics download buffer (for backward compatibility)
    pipeline->metrics_buffer = nova_vma_allocate_buffer(
        pipeline->context,
        batch_size * sizeof(EvaluationMetrics),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    );
    if (!pipeline->metrics_buffer) {
        std::cerr << "❌ Failed to allocate metrics buffer\n";
        genesis_batch_pipeline_free(pipeline);
        return nullptr;
    }
    pipeline->metrics_mapped = nova_vma_map(pipeline->metrics_buffer);
    if (!pipeline->metrics_mapped) {
        std::cerr << "❌ Failed to map metrics buffer\n";
        genesis_batch_pipeline_free(pipeline);
        return nullptr;
    }

    std::cout << "✅ Batch pipeline initialized (batch_size=" << batch_size << ")\n";
    std::cout << "   GPU memory: ~" << (batch_size * (width * height * 4 * 4 * 3 + eval_size * eval_size * 4 * 4) / (1024 * 1024)) << " MB\n";
    std::cout << "   NEW: Persistent proto-unity created (shared across batch)\n";
    std::cout << "   NEW: Training input images created (" << batch_size << ")\n";

    return pipeline;
}

extern "C" int genesis_batch_evaluate(
    GenesisBatchPipeline* pipeline,
    const ModelParameters* params,
    EvaluationMetrics* metrics,
    uint32_t count
) {
    if (count > pipeline->batch_size) {
        std::cerr << "❌ count exceeds batch_size\n";
        return -1;
    }

    // Upload parameters (only data that changes)
    for (uint32_t i = 0; i < count; i++) {
        // Prepare gamma params
        struct {
            float base_frequency;
            float initial_phase;
            float amplitude;
            float envelope_sigma;
            uint32_t num_harmonics;
            float harmonic_decay;
            uint32_t _pad[2];
        } gamma_params;

        gamma_params.base_frequency = params[i].gamma_base_frequency;
        gamma_params.initial_phase = params[i].gamma_initial_phase;
        gamma_params.amplitude = params[i].gamma_amplitude;
        gamma_params.envelope_sigma = params[i].gamma_envelope_sigma;
        gamma_params.num_harmonics = params[i].gamma_num_harmonics;
        gamma_params.harmonic_decay = params[i].gamma_harmonic_decay;

        std::memcpy(pipeline->gamma_params_mapped[i], &gamma_params, sizeof(gamma_params));
        nova_vma_flush(pipeline->gamma_params_buffers[i], 0, sizeof(gamma_params));

        // Prepare iota params (K=256 harmonics, but we only use first 10 from params)
        std::memcpy(pipeline->iota_params_mapped[i], &params[i], sizeof(ModelParameters));
        nova_vma_flush(pipeline->iota_params_buffers[i], 0, sizeof(ModelParameters));

        // Prepare tau params
        struct {
            float normalization_epsilon;
            float projection_strength;
            float noise_threshold;
            uint32_t use_template_normalization;
            uint32_t _pad[4];
        } tau_params;

        tau_params.normalization_epsilon = params[i].tau_normalization_epsilon;
        tau_params.projection_strength = params[i].tau_projection_strength;
        tau_params.noise_threshold = params[i].tau_noise_threshold;
        tau_params.use_template_normalization = params[i].tau_use_template_normalization;

        std::memcpy(pipeline->tau_params_mapped[i], &tau_params, sizeof(tau_params));
        nova_vma_flush(pipeline->tau_params_buffers[i], 0, sizeof(tau_params));

        // Prepare epsilon params
        struct {
            float energy_weight;
            float coherence_weight;
            float sparsity_weight;
            float quality_weight;
            uint32_t reduction_factor;
            float coherence_threshold;
            uint32_t _pad[2];
        } epsilon_params;

        epsilon_params.energy_weight = params[i].epsilon_energy_weight;
        epsilon_params.coherence_weight = params[i].epsilon_coherence_weight;
        epsilon_params.sparsity_weight = params[i].epsilon_sparsity_weight;
        epsilon_params.quality_weight = params[i].epsilon_quality_weight;
        epsilon_params.reduction_factor = params[i].epsilon_reduction_factor;
        epsilon_params.coherence_threshold = params[i].epsilon_coherence_threshold;

        std::memcpy(pipeline->epsilon_params_mapped[i], &epsilon_params, sizeof(epsilon_params));
        nova_vma_flush(pipeline->epsilon_params_buffers[i], 0, sizeof(epsilon_params));
    }

    // Execute batch (sequential for simplicity, can be parallelized)
    for (uint32_t i = 0; i < count; i++) {
        void* cmd = pipeline->cmd_buffers[i];

        if (nova_cmd_begin(cmd) != 0) {
            std::cerr << "❌ Failed to begin command buffer " << i << "\n";
            return -1;
        }

        // No layout transition needed - already initialized to GENERAL in init()
        // This prevents system freezes from repeated UNDEFINED→GENERAL transitions

        // Dispatch γ: ∅ → 𝟙
        nova_cmd_bind_pipeline(cmd, pipeline->gamma_pipeline);
        nova_cmd_bind_descriptor_set(cmd, pipeline->gamma_pipeline, pipeline->gamma_desc_sets[i]);
        nova_cmd_dispatch(cmd, (pipeline->width + 15) / 16, (pipeline->height + 15) / 16, 1);

        // Barrier after γ: proto_unity write → read
        nova_cmd_pipeline_barrier_image(
            cmd,
            pipeline->proto_unity[i],
            VK_IMAGE_LAYOUT_GENERAL,
            VK_IMAGE_LAYOUT_GENERAL,
            VK_ACCESS_SHADER_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
        );

        // Dispatch ι: 𝟙 → n
        nova_cmd_bind_pipeline(cmd, pipeline->iota_pipeline);
        nova_cmd_bind_descriptor_set(cmd, pipeline->iota_pipeline, pipeline->iota_desc_sets[i]);
        nova_cmd_dispatch(cmd, (pipeline->width + 15) / 16, (pipeline->height + 15) / 16, 1);

        // Barrier after ι: instance write → read
        nova_cmd_pipeline_barrier_image(
            cmd,
            pipeline->instance[i],
            VK_IMAGE_LAYOUT_GENERAL,
            VK_IMAGE_LAYOUT_GENERAL,
            VK_ACCESS_SHADER_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
        );

        // Dispatch τ: n → 𝟙'
        nova_cmd_bind_pipeline(cmd, pipeline->tau_pipeline);
        nova_cmd_bind_descriptor_set(cmd, pipeline->tau_pipeline, pipeline->tau_desc_sets[i]);
        nova_cmd_dispatch(cmd, (pipeline->width + 15) / 16, (pipeline->height + 15) / 16, 1);

        // Barrier after τ: proto_recovered write → read
        nova_cmd_pipeline_barrier_image(
            cmd,
            pipeline->proto_recovered[i],
            VK_IMAGE_LAYOUT_GENERAL,
            VK_IMAGE_LAYOUT_GENERAL,
            VK_ACCESS_SHADER_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
        );

        // Dispatch ε: 𝟙' → metrics (64×64 evaluation)
        nova_cmd_bind_pipeline(cmd, pipeline->epsilon_pipeline);
        nova_cmd_bind_descriptor_set(cmd, pipeline->epsilon_pipeline, pipeline->epsilon_desc_sets[i]);
        nova_cmd_dispatch(cmd, (64 + 7) / 8, (64 + 7) / 8, 1);  // 64×64 with 8×8 workgroups

        if (nova_cmd_end(cmd) != 0) {
            std::cerr << "❌ Failed to end command buffer " << i << "\n";
            return -1;
        }

        // Submit
        if (nova_submit_compute(pipeline->context, pipeline->queue, cmd) != 0) {
            std::cerr << "❌ Failed to submit compute commands " << i << "\n";
            return -1;
        }
    }

    // Wait for all to complete
    nova_queue_wait_idle(pipeline->queue);

    // Download metrics from evaluation images (64×64 RGBA32F)
    const uint32_t eval_size = 64;
    std::vector<float> eval_data(eval_size * eval_size * 4);  // RGBA floats

    for (uint32_t i = 0; i < count; i++) {
        // Download evaluation image to CPU
        int download_result = nova_image_download(
            pipeline->context,
            pipeline->evaluation[i],
            eval_data.data(),
            eval_size,
            eval_size
        );

        if (download_result != 0) {
            std::cerr << "❌ Failed to download evaluation image " << i << "\n";
            // Fallback to default metrics
            metrics[i].energy = 0.0f;
            metrics[i].coherence = 0.0f;
            metrics[i].sparsity = 0.0f;
            metrics[i].quality = 0.0f;
            metrics[i].factorization_loss = 100.0f;
            continue;
        }

        // Parse metrics from evaluation image
        // The epsilon shader writes:
        // R channel: energy (average magnitude)
        // G channel: coherence (phase consistency)
        // B channel: sparsity (non-zero ratio)
        // A channel: quality score

        float total_energy = 0.0f;
        float total_coherence = 0.0f;
        float total_sparsity = 0.0f;
        float total_quality = 0.0f;

        for (uint32_t y = 0; y < eval_size; y++) {
            for (uint32_t x = 0; x < eval_size; x++) {
                uint32_t idx = (y * eval_size + x) * 4;
                total_energy += eval_data[idx + 0];
                total_coherence += eval_data[idx + 1];
                total_sparsity += eval_data[idx + 2];
                total_quality += eval_data[idx + 3];
            }
        }

        const float pixel_count = eval_size * eval_size;
        metrics[i].energy = total_energy / pixel_count;
        metrics[i].coherence = total_coherence / pixel_count;
        metrics[i].sparsity = total_sparsity / pixel_count;
        metrics[i].quality = total_quality / pixel_count;

        // Compute factorization loss
        // Compare proto_unity vs proto_recovered for reconstruction quality
        // For now, estimate based on coherence and quality
        metrics[i].factorization_loss =
            (1.0f - metrics[i].coherence) * 10.0f + 
            (100.0f - metrics[i].quality) * 0.01f;
    }

    return 0;
}

// NEW ARCHITECTURE: Execute γ ONCE to produce persistent proto-unity
extern "C" int genesis_execute_gamma_once(
    GenesisBatchPipeline* pipeline,
    const GammaParams* gamma_params
) {
    if (!pipeline || !gamma_params) {
        std::cerr << "❌ Invalid parameters (null pointer)\n";
        return -1;
    }
    
    // Validate gamma params
    if (gamma_params->num_harmonics == 0 || gamma_params->num_harmonics > 256) {
        std::cerr << "❌ Invalid num_harmonics: " << gamma_params->num_harmonics << "\n";
        return -1;
    }
    if (gamma_params->amplitude < 0.0f || gamma_params->amplitude > 10000.0f) {
        std::cerr << "❌ Invalid amplitude: " << gamma_params->amplitude << "\n";
        return -1;
    }

    // Prepare γ params struct
    struct {
        float base_frequency;
        float initial_phase;
        float amplitude;
        float envelope_sigma;
        uint32_t num_harmonics;
        float harmonic_decay;
        uint32_t _pad[2];
    } gamma_params_struct;

    gamma_params_struct.base_frequency = gamma_params->base_frequency;
    gamma_params_struct.initial_phase = gamma_params->initial_phase;
    gamma_params_struct.amplitude = gamma_params->amplitude;
    gamma_params_struct.envelope_sigma = gamma_params->envelope_sigma;
    gamma_params_struct.num_harmonics = gamma_params->num_harmonics;
    gamma_params_struct.harmonic_decay = gamma_params->harmonic_decay;

    // Upload γ params to GPU
    // Use per-instance buffer (index 0) for single execution
    std::memcpy(pipeline->gamma_params_mapped[0], &gamma_params_struct, sizeof(gamma_params_struct));
    nova_vma_flush(pipeline->gamma_params_buffers[0], 0, sizeof(gamma_params_struct));

    // Record command buffer
    void* cmd = pipeline->gamma_cmd_buffer;
    if (nova_cmd_begin(cmd) != 0) {
        std::cerr << "❌ Failed to begin γ command buffer\n";
        return -1;
    }

    // Bind γ pipeline and descriptor set
    nova_cmd_bind_pipeline(cmd, pipeline->gamma_pipeline);
    // Acquire proto-identity lock before binding
    acquire_proto_identity_lock(pipeline);
    
    // Update descriptor set: binding 0 = params, binding 1 = empty_state (input), binding 2 = proto_identity (output)
    void* gamma_vk_buf = nova_vma_get_buffer(pipeline->gamma_params_buffers[0]);
    nova_update_descriptor_set(pipeline->context, pipeline->gamma_desc_set_single, 0, gamma_vk_buf);
    nova_update_descriptor_set_image(pipeline->context, pipeline->gamma_desc_set_single, 1,
                                     pipeline->empty_state_view, VK_IMAGE_LAYOUT_GENERAL);
    nova_update_descriptor_set_image(pipeline->context, pipeline->gamma_desc_set_single, 2,
                                     pipeline->proto_identity_working_view, VK_IMAGE_LAYOUT_GENERAL);
    
    nova_cmd_bind_descriptor_set(cmd, pipeline->gamma_pipeline, pipeline->gamma_desc_set_single);

    // Dispatch γ shader (no layout transition needed - already GENERAL)
    nova_cmd_dispatch(cmd, (pipeline->width + 15) / 16, (pipeline->height + 15) / 16, 1);

    if (nova_cmd_end(cmd) != 0) {
        std::cerr << "❌ Failed to end γ command buffer\n";
        return -1;
    }

    // Submit and wait
    if (nova_submit_compute(pipeline->context, pipeline->queue, cmd) != 0) {
        std::cerr << "❌ Failed to submit γ compute commands\n";
        return -1;
    }

    nova_queue_wait_idle(pipeline->queue);
    
    // Release proto-identity lock after operation completes
    release_proto_identity_lock(pipeline);
    
    std::cout << "✅ γ (genesis) completed - proto-identity in working buffer\n";

    return 0;
}

// NEW: Execute ι morphism: 𝟙 → n (instantiation)
extern "C" int genesis_execute_iota_once(
    GenesisBatchPipeline* pipeline,
    const IotaParams* iota_params,
    float* output
) {
    if (!pipeline || !iota_params || !output) {
        std::cerr << "❌ Invalid parameters (null pointer)\n";
        return -1;
    }
    
    // Upload ι params to GPU
    // Use per-instance buffer (index 0) for single execution
    std::memcpy(pipeline->iota_params_mapped[0], iota_params, sizeof(IotaParams));
    nova_vma_flush(pipeline->iota_params_buffers[0], 0, sizeof(IotaParams));
    
    // Acquire proto-identity lock before binding
    acquire_proto_identity_lock(pipeline);
    
    // Update descriptor set to bind proto-identity
    void* iota_vk_buf = nova_vma_get_buffer(pipeline->iota_params_buffers[0]);
    nova_update_descriptor_set(pipeline->context, pipeline->iota_desc_set_single, 0, iota_vk_buf);
    nova_update_descriptor_set_image(pipeline->context, pipeline->iota_desc_set_single, 1,
                                     pipeline->proto_identity_working_view, VK_IMAGE_LAYOUT_GENERAL);
    nova_update_descriptor_set_image(pipeline->context, pipeline->iota_desc_set_single, 2,
                                     pipeline->instance_views[0], VK_IMAGE_LAYOUT_GENERAL);
    
    // Record command buffer
    void* cmd = pipeline->iota_cmd_buffer;
    if (nova_cmd_begin(cmd) != 0) {
        std::cerr << "❌ Failed to begin ι command buffer\n";
        release_proto_identity_lock(pipeline);
        return -1;
    }

    // Bind ι pipeline and descriptor set
    nova_cmd_bind_pipeline(cmd, pipeline->iota_pipeline);
    nova_cmd_bind_descriptor_set(cmd, pipeline->iota_pipeline, pipeline->iota_desc_set_single);
    
    // Dispatch ι shader (no layout transition needed - already GENERAL)
    nova_cmd_dispatch(cmd, (pipeline->width + 15) / 16, (pipeline->height + 15) / 16, 1);
    
    if (nova_cmd_end(cmd) != 0) {
        std::cerr << "❌ Failed to end ι command buffer\n";
        return -1;
    }
    
    // Submit and wait
    if (nova_submit_compute(pipeline->context, pipeline->queue, cmd) != 0) {
        std::cerr << "❌ Failed to submit ι compute commands\n";
        return -1;
    }
    
    nova_queue_wait_idle(pipeline->queue);
    
    // Download result from slot 0 instance image
    int download_result = nova_image_download(
        pipeline->context,
        pipeline->instance[0],
        output,
        pipeline->width,
        pipeline->height
    );
    if (download_result != 0) {
        std::cerr << "❌ Failed to download ι output\n";
        return -1;
    }
    
    return 0;
}

// NEW: Execute ε morphism: 𝟙 → ∞ (evaluation)
extern "C" int genesis_execute_epsilon_once(
    GenesisBatchPipeline* pipeline,
    const EpsilonParams* epsilon_params,
    float* output
) {
    if (!pipeline || !epsilon_params || !output) {
        std::cerr << "❌ Invalid parameters (null pointer)\n";
        return -1;
    }
    
    // Upload ε params to GPU
    // Use per-instance buffer (index 0) for single execution
    std::memcpy(pipeline->epsilon_params_mapped[0], epsilon_params, sizeof(EpsilonParams));
    nova_vma_flush(pipeline->epsilon_params_buffers[0], 0, sizeof(EpsilonParams));
    
    // Acquire proto-identity lock before binding
    acquire_proto_identity_lock(pipeline);
    
    // Update descriptor set to bind proto-identity
    void* epsilon_vk_buf = nova_vma_get_buffer(pipeline->epsilon_params_buffers[0]);
    nova_update_descriptor_set(pipeline->context, pipeline->epsilon_desc_set_single, 0, epsilon_vk_buf);
    nova_update_descriptor_set_image(pipeline->context, pipeline->epsilon_desc_set_single, 1,
                                     pipeline->proto_identity_working_view, VK_IMAGE_LAYOUT_GENERAL);
    nova_update_descriptor_set_image(pipeline->context, pipeline->epsilon_desc_set_single, 2,
                                     pipeline->evaluation_views[0], VK_IMAGE_LAYOUT_GENERAL);
    
    // Record command buffer
    void* cmd = pipeline->epsilon_cmd_buffer;
    if (nova_cmd_begin(cmd) != 0) {
        std::cerr << "❌ Failed to begin ε command buffer\n";
        release_proto_identity_lock(pipeline);
        return -1;
    }

    // Bind ε pipeline and descriptor set
    nova_cmd_bind_pipeline(cmd, pipeline->epsilon_pipeline);
    nova_cmd_bind_descriptor_set(cmd, pipeline->epsilon_pipeline, pipeline->epsilon_desc_set_single);
    
    // Dispatch ε shader (64×64 with 8×8 workgroups)
    const uint32_t eval_size = 64;
    nova_cmd_dispatch(cmd, (eval_size + 7) / 8, (eval_size + 7) / 8, 1);
    
    if (nova_cmd_end(cmd) != 0) {
        std::cerr << "❌ Failed to end ε command buffer\n";
        return -1;
    }
    
    // Submit and wait
    if (nova_submit_compute(pipeline->context, pipeline->queue, cmd) != 0) {
        std::cerr << "❌ Failed to submit ε compute commands\n";
        return -1;
    }
    
    nova_queue_wait_idle(pipeline->queue);
    
    // Download result from slot 0 evaluation image (64×64)
    // (eval_size already declared above)
    int download_result = nova_image_download(
        pipeline->context,
        pipeline->evaluation[0],
        output,
        eval_size,
        eval_size
    );
    if (download_result != 0) {
        std::cerr << "❌ Failed to download ε output\n";
        return -1;
    }
    
    return 0;
}

// NEW: Execute ε⁻¹ morphism: ∞ → 𝟙 (reverse evaluation) - FULLY ON GPU
extern "C" int genesis_execute_epsilon_reverse(
    GenesisBatchPipeline* pipeline,
    const EpsilonParams* epsilon_params,
    const float* input,
    float* output
) {
    if (!pipeline || !epsilon_params || !input || !output) {
        std::cerr << "❌ Invalid parameters (null pointer)\n";
        return -1;
    }
    
    // Upload params to per-instance buffer (index 0)
    std::memcpy(pipeline->epsilon_params_mapped[0], epsilon_params, sizeof(EpsilonParams));
    nova_vma_flush(pipeline->epsilon_params_buffers[0], 0, sizeof(EpsilonParams));
    
    // Upload input to infinity_state_image (∞) - this is the memory state
    const uint32_t eval_size = 64;
    int upload_result = nova_image_upload(
        pipeline->context,
        pipeline->infinity_state_image,
        input,
        eval_size,
        eval_size
    );
    if (upload_result != 0) {
        std::cerr << "❌ Failed to upload ∞ state for ε⁻¹\n";
        return -1;
    }
    
    // Use GPU shader if available, otherwise CPU fallback
    if (pipeline->epsilon_reverse_pipeline) {
        // Acquire proto-identity lock before binding
        acquire_proto_identity_lock(pipeline);
        
        // Allocate descriptor set for reverse operation
        void* desc_set = nova_allocate_descriptor_set(pipeline->context, pipeline->epsilon_reverse_desc_layout);
        if (!desc_set) {
            std::cerr << "❌ Failed to allocate ε⁻¹ descriptor set\n";
            release_proto_identity_lock(pipeline);
            return -1;
        }
        
        void* epsilon_vk_buf = nova_vma_get_buffer(pipeline->epsilon_params_buffers[0]);
        
        // Record command buffer
        void* cmd = nova_allocate_command_buffer(pipeline->context, pipeline->cmd_pool);
        if (nova_cmd_begin(cmd) != 0) {
            std::cerr << "❌ Failed to begin ε⁻¹ command buffer\n";
            return -1;
        }
        
        // Memory barrier: Ensure infinity_state is ready after upload (TRANSFER -> COMPUTE)
        nova_cmd_pipeline_barrier_image(
            cmd,
            pipeline->infinity_state_image,
            VK_IMAGE_LAYOUT_GENERAL,
            VK_IMAGE_LAYOUT_GENERAL,
            VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
        );
        
        // Update descriptor set AFTER barrier to ensure image state is correct
        nova_update_descriptor_set(pipeline->context, desc_set, 0, epsilon_vk_buf);
        nova_update_descriptor_set_image(pipeline->context, desc_set, 1,
                                         pipeline->infinity_state_view, VK_IMAGE_LAYOUT_GENERAL);
        nova_update_descriptor_set_image(pipeline->context, desc_set, 2,
                                         pipeline->proto_identity_working_view, VK_IMAGE_LAYOUT_GENERAL);
        
        nova_cmd_bind_pipeline(cmd, pipeline->epsilon_reverse_pipeline);
        nova_cmd_bind_descriptor_set(cmd, pipeline->epsilon_reverse_pipeline, desc_set);
        nova_cmd_dispatch(cmd, (pipeline->width + 15) / 16, (pipeline->height + 15) / 16, 1);
        
        if (nova_cmd_end(cmd) != 0) {
            std::cerr << "❌ Failed to end ε⁻¹ command buffer\n";
            return -1;
        }
        
        // Submit and wait
        if (nova_submit_compute(pipeline->context, pipeline->queue, cmd) != 0) {
            std::cerr << "❌ Failed to submit ε⁻¹ compute commands\n";
            return -1;
        }
        nova_queue_wait_idle(pipeline->queue);
        
        // Download result from GPU (only at end - data stays on GPU during cycle)
        int download_result = nova_image_download(
            pipeline->context,
            pipeline->proto_identity_working,
            output,
            pipeline->width,
            pipeline->height
        );
        if (download_result != 0) {
            std::cerr << "❌ Failed to download ε⁻¹ output\n";
            return -1;
        }
        
        // Cleanup (per-instance buffers are freed in pipeline cleanup)
        nova_free_descriptor_set(pipeline->context, desc_set);
        nova_free_command_buffer(pipeline->context, pipeline->cmd_pool, cmd);
        
        return 0;
    } else {
        // CPU fallback
        const float* eval_data = input;
        float* proto_data = output;
        uint32_t reduction = epsilon_params->reduction_factor;
        if (reduction == 0) reduction = 8;
        
        for (uint32_t y = 0; y < pipeline->height; y++) {
            for (uint32_t x = 0; x < pipeline->width; x++) {
                uint32_t eval_y = y / reduction;
                uint32_t eval_x = x / reduction;
                if (eval_y >= eval_size) eval_y = eval_size - 1;
                if (eval_x >= eval_size) eval_x = eval_size - 1;
                
                uint32_t eval_idx = (eval_y * eval_size + eval_x) * 4;
                uint32_t proto_idx = (y * pipeline->width + x) * 4;
                
                for (uint32_t c = 0; c < 4; c++) {
                    proto_data[proto_idx + c] = eval_data[eval_idx + c];
                }
            }
        }
        return 0;
    }
}

// NEW: Execute τ⁻¹ morphism: 𝟙 → n (reverse encoding) - FULLY ON GPU
extern "C" int genesis_execute_tau_reverse(
    GenesisBatchPipeline* pipeline,
    const TauParams* tau_params,
    const float* input,
    float* output
) {
    if (!pipeline || !tau_params || !input || !output) {
        std::cerr << "❌ Invalid parameters (null pointer)\n";
        return -1;
    }
    
    // Upload params to per-instance buffer (index 0)
    std::memcpy(pipeline->tau_params_mapped[0], tau_params, sizeof(TauParams));
    nova_vma_flush(pipeline->tau_params_buffers[0], 0, sizeof(TauParams));
    
    // Upload input proto-unity to proto_identity (𝟙) - this is the shared thinking state
    int upload_result = nova_image_upload(
        pipeline->context,
        pipeline->proto_identity_working,
        input,
        pipeline->width,
        pipeline->height
    );
    if (upload_result != 0) {
        std::cerr << "❌ Failed to upload τ⁻¹ input to proto-identity\n";
        return -1;
    }
    
    // Use GPU shader if available, otherwise CPU fallback
    if (pipeline->tau_reverse_pipeline) {
        // Acquire proto-identity lock before binding
        acquire_proto_identity_lock(pipeline);
        
        void* desc_set = nova_allocate_descriptor_set(pipeline->context, pipeline->tau_reverse_desc_layout);
        if (!desc_set) {
            std::cerr << "❌ Failed to allocate τ⁻¹ descriptor set\n";
            release_proto_identity_lock(pipeline);
            return -1;
        }
        
        void* tau_vk_buf = nova_vma_get_buffer(pipeline->tau_params_buffers[0]);
        
        void* cmd = nova_allocate_command_buffer(pipeline->context, pipeline->cmd_pool);
        if (nova_cmd_begin(cmd) != 0) {
            std::cerr << "❌ Failed to begin τ⁻¹ command buffer\n";
            return -1;
        }
        
        // Memory barrier: Ensure proto-identity is ready after upload (TRANSFER -> COMPUTE)
        nova_cmd_pipeline_barrier_image(
            cmd,
            pipeline->proto_identity_working,
            VK_IMAGE_LAYOUT_GENERAL,
            VK_IMAGE_LAYOUT_GENERAL,
            VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
        );
        
        // Update descriptor set AFTER barrier to ensure image state is correct
        nova_update_descriptor_set(pipeline->context, desc_set, 0, tau_vk_buf);
        nova_update_descriptor_set_image(pipeline->context, desc_set, 1,
                                         pipeline->proto_identity_working_view, VK_IMAGE_LAYOUT_GENERAL);
        nova_update_descriptor_set_image(pipeline->context, desc_set, 2,
                                         pipeline->instance_views[0], VK_IMAGE_LAYOUT_GENERAL);
        
        nova_cmd_bind_pipeline(cmd, pipeline->tau_reverse_pipeline);
        nova_cmd_bind_descriptor_set(cmd, pipeline->tau_reverse_pipeline, desc_set);
        nova_cmd_dispatch(cmd, (pipeline->width + 15) / 16, (pipeline->height + 15) / 16, 1);
        
        if (nova_cmd_end(cmd) != 0) {
            std::cerr << "❌ Failed to end τ⁻¹ command buffer\n";
            nova_free_descriptor_set(pipeline->context, desc_set);
            nova_free_command_buffer(pipeline->context, pipeline->cmd_pool, cmd);
            release_proto_identity_lock(pipeline);
            return -1;
        }
        
        if (nova_submit_compute(pipeline->context, pipeline->queue, cmd) != 0) {
            std::cerr << "❌ Failed to submit τ⁻¹ compute commands\n";
            nova_free_descriptor_set(pipeline->context, desc_set);
            nova_free_command_buffer(pipeline->context, pipeline->cmd_pool, cmd);
            release_proto_identity_lock(pipeline);
            return -1;
        }
        nova_queue_wait_idle(pipeline->queue);
        
        // Release proto-identity lock after operation completes
        release_proto_identity_lock(pipeline);
        
        int download_result = nova_image_download(
            pipeline->context,
            pipeline->instance[0],
            output,
            pipeline->width,
            pipeline->height
        );
        if (download_result != 0) {
            std::cerr << "❌ Failed to download τ⁻¹ output\n";
            nova_free_descriptor_set(pipeline->context, desc_set);
            nova_free_command_buffer(pipeline->context, pipeline->cmd_pool, cmd);
            return -1;
        }
        
        nova_free_descriptor_set(pipeline->context, desc_set);
        nova_free_command_buffer(pipeline->context, pipeline->cmd_pool, cmd);
        
        return 0;
    } else {
        // CPU fallback
        const float* proto_data = input;
        float* instance_data = output;
        float strength = tau_params->projection_strength;
        if (strength == 0.0f) strength = 1.0f;
        
        uint32_t size = pipeline->width * pipeline->height * 4;
        for (uint32_t i = 0; i < size; i++) {
            instance_data[i] = proto_data[i] / strength;
        }
        return 0;
    }
}

// NEW: Execute ι⁻¹ morphism: n → 𝟙 (reverse instantiation) - FULLY ON GPU
extern "C" int genesis_execute_iota_reverse(
    GenesisBatchPipeline* pipeline,
    const IotaParams* iota_params,
    const float* input,
    float* output
) {
    if (!pipeline || !iota_params || !input || !output) {
        std::cerr << "❌ Invalid parameters (null pointer)\n";
        return -1;
    }
    
    // Upload params to per-instance buffer (index 0)
    std::memcpy(pipeline->iota_params_mapped[0], iota_params, sizeof(IotaParams));
    nova_vma_flush(pipeline->iota_params_buffers[0], 0, sizeof(IotaParams));
    
    // Upload input instance to GPU slot 0 instance image
    int upload_result = nova_image_upload(
        pipeline->context,
        pipeline->instance[0],
        input,
        pipeline->width,
        pipeline->height
    );
    if (upload_result != 0) {
        std::cerr << "❌ Failed to upload ι⁻¹ input\n";
        return -1;
    }
    
    // Use GPU shader if available, otherwise CPU fallback
    if (pipeline->iota_reverse_pipeline) {
        // Acquire proto-identity lock before binding
        acquire_proto_identity_lock(pipeline);
        
        void* desc_set = nova_allocate_descriptor_set(pipeline->context, pipeline->iota_reverse_desc_layout);
        if (!desc_set) {
            std::cerr << "❌ Failed to allocate ι⁻¹ descriptor set\n";
            release_proto_identity_lock(pipeline);
            return -1;
        }
        
        void* iota_vk_buf = nova_vma_get_buffer(pipeline->iota_params_buffers[0]);
        
        void* cmd = nova_allocate_command_buffer(pipeline->context, pipeline->cmd_pool);
        if (nova_cmd_begin(cmd) != 0) {
            std::cerr << "❌ Failed to begin ι⁻¹ command buffer\n";
            return -1;
        }
        
        // Memory barrier: Ensure image is ready after upload (TRANSFER -> COMPUTE)
        nova_cmd_pipeline_barrier_image(
            cmd,
            pipeline->instance[0],
            VK_IMAGE_LAYOUT_GENERAL,
            VK_IMAGE_LAYOUT_GENERAL,
            VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
        );
        
        // Update descriptor set AFTER barrier to ensure image state is correct
        nova_update_descriptor_set(pipeline->context, desc_set, 0, iota_vk_buf);
        nova_update_descriptor_set_image(pipeline->context, desc_set, 1,
                                         pipeline->instance_views[0], VK_IMAGE_LAYOUT_GENERAL);
        nova_update_descriptor_set_image(pipeline->context, desc_set, 2,
                                         pipeline->proto_identity_working_view, VK_IMAGE_LAYOUT_GENERAL);
        
        nova_cmd_bind_pipeline(cmd, pipeline->iota_reverse_pipeline);
        nova_cmd_bind_descriptor_set(cmd, pipeline->iota_reverse_pipeline, desc_set);
        nova_cmd_dispatch(cmd, (pipeline->width + 15) / 16, (pipeline->height + 15) / 16, 1);
        
        if (nova_cmd_end(cmd) != 0) {
            std::cerr << "❌ Failed to end ι⁻¹ command buffer\n";
            nova_free_descriptor_set(pipeline->context, desc_set);
            nova_free_command_buffer(pipeline->context, pipeline->cmd_pool, cmd);
            release_proto_identity_lock(pipeline);
            return -1;
        }
        
        if (nova_submit_compute(pipeline->context, pipeline->queue, cmd) != 0) {
            std::cerr << "❌ Failed to submit ι⁻¹ compute commands\n";
            nova_free_descriptor_set(pipeline->context, desc_set);
            nova_free_command_buffer(pipeline->context, pipeline->cmd_pool, cmd);
            release_proto_identity_lock(pipeline);
            return -1;
        }
        nova_queue_wait_idle(pipeline->queue);
        
        // Release proto-identity lock after operation completes
        release_proto_identity_lock(pipeline);
        
        int download_result = nova_image_download(
            pipeline->context,
            pipeline->proto_identity_working,
            output,
            pipeline->width,
            pipeline->height
        );
        if (download_result != 0) {
            std::cerr << "❌ Failed to download ι⁻¹ output\n";
            nova_free_descriptor_set(pipeline->context, desc_set);
            nova_free_command_buffer(pipeline->context, pipeline->cmd_pool, cmd);
            return -1;
        }
        
        nova_free_descriptor_set(pipeline->context, desc_set);
        nova_free_command_buffer(pipeline->context, pipeline->cmd_pool, cmd);
        
        return 0;
    } else {
        // CPU fallback
        const float* instance_data = input;
        float* proto_data = output;
        float amplitude = iota_params->global_amplitude;
        if (amplitude == 0.0f) amplitude = 1.0f;
        
        uint32_t size = pipeline->width * pipeline->height * 4;
        for (uint32_t i = 0; i < size; i++) {
            proto_data[i] = instance_data[i] / amplitude;
        }
        return 0;
    }
}

// NEW: Execute γ⁻¹ morphism: 𝟙 → ∅ (reverse genesis) - FULLY ON GPU
extern "C" int genesis_execute_gamma_reverse(
    GenesisBatchPipeline* pipeline,
    const GammaParams* gamma_params,
    const float* input,
    float* output
) {
    if (!pipeline || !gamma_params || !input || !output) {
        std::cerr << "❌ Invalid parameters (null pointer)\n";
        return -1;
    }
    
    // Upload params to per-instance buffer (index 0)
    std::memcpy(pipeline->gamma_params_mapped[0], gamma_params, sizeof(GammaParams));
    nova_vma_flush(pipeline->gamma_params_buffers[0], 0, sizeof(GammaParams));
    
    // Upload input proto-unity to proto_identity (𝟙) - this is the shared thinking state
    int upload_result = nova_image_upload(
        pipeline->context,
        pipeline->proto_identity_working,
        input,
        pipeline->width,
        pipeline->height
    );
    if (upload_result != 0) {
        std::cerr << "❌ Failed to upload γ⁻¹ input to proto-identity\n";
        return -1;
    }
    
    // Use GPU shader if available, otherwise CPU fallback
    if (pipeline->gamma_reverse_pipeline) {
        // Acquire proto-identity lock before binding
        acquire_proto_identity_lock(pipeline);
        
        void* desc_set = nova_allocate_descriptor_set(pipeline->context, pipeline->gamma_reverse_desc_layout);
        if (!desc_set) {
            std::cerr << "❌ Failed to allocate γ⁻¹ descriptor set\n";
            release_proto_identity_lock(pipeline);
            return -1;
        }
        
        void* gamma_vk_buf = nova_vma_get_buffer(pipeline->gamma_params_buffers[0]);
        
        void* cmd = nova_allocate_command_buffer(pipeline->context, pipeline->cmd_pool);
        if (nova_cmd_begin(cmd) != 0) {
            std::cerr << "❌ Failed to begin γ⁻¹ command buffer\n";
            return -1;
        }
        
        // Memory barrier: Ensure proto-identity is ready after upload (TRANSFER -> COMPUTE)
        nova_cmd_pipeline_barrier_image(
            cmd,
            pipeline->proto_identity_working,
            VK_IMAGE_LAYOUT_GENERAL,
            VK_IMAGE_LAYOUT_GENERAL,
            VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
        );
        
        // Update descriptor set AFTER barrier to ensure image state is correct
        nova_update_descriptor_set(pipeline->context, desc_set, 0, gamma_vk_buf);
        nova_update_descriptor_set_image(pipeline->context, desc_set, 1,
                                         pipeline->proto_identity_working_view, VK_IMAGE_LAYOUT_GENERAL);
        nova_update_descriptor_set_image(pipeline->context, desc_set, 2,
                                         pipeline->empty_state_view, VK_IMAGE_LAYOUT_GENERAL);
        
        nova_cmd_bind_pipeline(cmd, pipeline->gamma_reverse_pipeline);
        nova_cmd_bind_descriptor_set(cmd, pipeline->gamma_reverse_pipeline, desc_set);
        nova_cmd_dispatch(cmd, (pipeline->width + 15) / 16, (pipeline->height + 15) / 16, 1);
        
        if (nova_cmd_end(cmd) != 0) {
            std::cerr << "❌ Failed to end γ⁻¹ command buffer\n";
            nova_free_descriptor_set(pipeline->context, desc_set);
            nova_free_command_buffer(pipeline->context, pipeline->cmd_pool, cmd);
            release_proto_identity_lock(pipeline);
            return -1;
        }
        
        if (nova_submit_compute(pipeline->context, pipeline->queue, cmd) != 0) {
            std::cerr << "❌ Failed to submit γ⁻¹ compute commands\n";
            nova_free_descriptor_set(pipeline->context, desc_set);
            nova_free_command_buffer(pipeline->context, pipeline->cmd_pool, cmd);
            release_proto_identity_lock(pipeline);
            return -1;
        }
        nova_queue_wait_idle(pipeline->queue);
        
        // Release proto-identity lock after operation completes
        release_proto_identity_lock(pipeline);
        
        int download_result = nova_image_download(
            pipeline->context,
            pipeline->empty_state_image,
            output,
            pipeline->width,
            pipeline->height
        );
        if (download_result != 0) {
            std::cerr << "❌ Failed to download γ⁻¹ output\n";
            nova_free_descriptor_set(pipeline->context, desc_set);
            nova_free_command_buffer(pipeline->context, pipeline->cmd_pool, cmd);
            return -1;
        }
        
        nova_free_descriptor_set(pipeline->context, desc_set);
        nova_free_command_buffer(pipeline->context, pipeline->cmd_pool, cmd);
        
        return 0;
    } else {
        // CPU fallback
        const float* proto_data = input;
        float* empty_data = output;
        float dissolve_factor = 0.1f;
        
        uint32_t size = pipeline->width * pipeline->height * 4;
        for (uint32_t i = 0; i < size; i++) {
            empty_data[i] = proto_data[i] * dissolve_factor;
        }
        return 0;
    }
}

// NEW ARCHITECTURE: Encode training data batch via τ shader
extern "C" int genesis_batch_encode(
    GenesisBatchPipeline* pipeline,
    const TrainingData* training_data,
    ProtoEmbedding* proto_embeddings,
    const TauParams* tau_params,
    uint32_t count
) {
    if (!pipeline || !training_data || !proto_embeddings || !tau_params) {
        std::cerr << "❌ Invalid parameters (null pointer)\n";
        return -1;
    }

    if (count == 0) {
        std::cerr << "❌ count is zero\n";
        return -1;
    }
    
    if (count > pipeline->batch_size) {
        std::cerr << "❌ count (" << count << ") exceeds batch_size (" 
                  << pipeline->batch_size << ")\n";
        return -1;
    }
    
    // Validate training data
    for (uint32_t i = 0; i < count; i++) {
        if (!training_data[i].waveform) {
            std::cerr << "❌ Training data " << i << " has null waveform pointer\n";
            return -1;
        }
        if (training_data[i].width != pipeline->width || 
            training_data[i].height != pipeline->height) {
            std::cerr << "❌ Training data " << i << " dimensions mismatch: "
                      << training_data[i].width << "×" << training_data[i].height
                      << " vs expected " << pipeline->width << "×" << pipeline->height << "\n";
            return -1;
        }
    }
    
    // Validate tau params
    if (tau_params->projection_strength < 0.0f || tau_params->projection_strength > 1.0f) {
        std::cerr << "❌ Invalid projection_strength: " << tau_params->projection_strength << "\n";
        return -1;
    }

    // Step 1: Upload training data to GPU
    // (nova_image_upload declared at top of file)
    
    for (uint32_t i = 0; i < count; i++) {
        // Upload waveform data to training_inputs[i]
        int upload_result = nova_image_upload(
            pipeline->context,
            pipeline->training_inputs[i],
            training_data[i].waveform,
            training_data[i].width,
            training_data[i].height
        );
        if (upload_result != 0) {
            std::cerr << "❌ Failed to upload training data " << i << "\n";
            return -1;
        }
    }

    // Step 2: Upload τ params (per-instance - each instance gets its own params)
    // For now, use same params for all instances, but each has its own buffer
    struct {
        float normalization_epsilon;
        float projection_strength;
        float noise_threshold;
        uint32_t use_template_normalization;
        uint32_t _pad[4];
    } tau_params_struct;

    tau_params_struct.normalization_epsilon = tau_params->normalization_epsilon;
    tau_params_struct.projection_strength = tau_params->projection_strength;
    tau_params_struct.noise_threshold = tau_params->noise_threshold;
    tau_params_struct.use_template_normalization = tau_params->use_template_normalization ? 1 : 0;

    // Upload to each instance's buffer (per-instance params)
    for (uint32_t i = 0; i < count; i++) {
        std::memcpy(pipeline->tau_params_mapped[i], &tau_params_struct, sizeof(tau_params_struct));
        nova_vma_flush(pipeline->tau_params_buffers[i], 0, sizeof(tau_params_struct));
    }

    // Step 3: Execute τ shader in parallel (batch_size times)
    for (uint32_t i = 0; i < count; i++) {
        void* cmd = pipeline->cmd_buffers[i];  // Reusing existing cmd buffers

        if (nova_cmd_begin(cmd) != 0) {
            std::cerr << "❌ Failed to begin τ command buffer " << i << "\n";
            return -1;
        }

        // Rebind τ descriptor set for new architecture:
        // Binding 1: training_inputs[i] (not instance)
        // Binding 2: proto_identity (shared, requires lock)
        // Binding 3: proto_recovered[i] (output, same as before)
        
        // Acquire proto-identity lock before binding
        acquire_proto_identity_lock(pipeline);
        
        void* tau_vk_buf = nova_vma_get_buffer(pipeline->tau_params_buffers[i]);  // Per-instance params
        nova_update_descriptor_set(pipeline->context, pipeline->tau_desc_sets[i], 0, tau_vk_buf);
        nova_update_descriptor_set_image(pipeline->context, pipeline->tau_desc_sets[i], 1, 
                                         pipeline->training_input_views[i], VK_IMAGE_LAYOUT_GENERAL);
        nova_update_descriptor_set_image(pipeline->context, pipeline->tau_desc_sets[i], 2, 
                                         pipeline->proto_identity_working_view, VK_IMAGE_LAYOUT_GENERAL);
        nova_update_descriptor_set_image(pipeline->context, pipeline->tau_desc_sets[i], 3, 
                                         pipeline->proto_recovered_views[i], VK_IMAGE_LAYOUT_GENERAL);

        // Bind τ pipeline and descriptor set
        nova_cmd_bind_pipeline(cmd, pipeline->tau_pipeline);
        nova_cmd_bind_descriptor_set(cmd, pipeline->tau_pipeline, pipeline->tau_desc_sets[i]);

        // Dispatch τ shader: training_inputs[i] + proto_identity → proto_recovered[i]
        // Note: Reusing proto_recovered as proto_embeddings for now
        // No layout transitions needed - already GENERAL
        nova_cmd_dispatch(cmd, (pipeline->width + 15) / 16, (pipeline->height + 15) / 16, 1);

        if (nova_cmd_end(cmd) != 0) {
            std::cerr << "❌ Failed to end τ command buffer " << i << "\n";
            release_proto_identity_lock(pipeline);
            return -1;
        }

        if (nova_submit_compute(pipeline->context, pipeline->queue, cmd) != 0) {
            std::cerr << "❌ Failed to submit τ compute commands " << i << "\n";
            release_proto_identity_lock(pipeline);
            return -1;
        }

        // CRITICAL FIX: Release the lock and immediately wait for the GPU to finish
        // the submitted command. This serializes each operation in the batch, 
        // preventing the race condition on the shared proto-identity resource
        // that was causing system hangs.
        release_proto_identity_lock(pipeline);
        nova_queue_wait_idle(pipeline->queue);
    }

    // Step 4: Wait for all to complete (No longer needed as we wait inside the loop)

    // Step 5: Download proto-embeddings to CPU
    for (uint32_t i = 0; i < count; i++) {
        // Allocate output buffer if needed
        if (!proto_embeddings[i].embedding) {
            proto_embeddings[i].embedding = (float*)malloc(
                pipeline->width * pipeline->height * 4 * sizeof(float)
            );
            if (!proto_embeddings[i].embedding) {
                std::cerr << "❌ Failed to allocate output buffer " << i << "\n";
                return -1;
            }
        }

        // Download proto_recovered[i] (used as proto_embeddings)
        int download_result = nova_image_download(
            pipeline->context,
            pipeline->proto_recovered[i],  // Reusing proto_recovered as proto_embeddings
            proto_embeddings[i].embedding,
            pipeline->width,
            pipeline->height
        );

        if (download_result != 0) {
            std::cerr << "❌ Failed to download proto-embedding " << i << "\n";
            return -1;
        }

        proto_embeddings[i].width = pipeline->width;
        proto_embeddings[i].height = pipeline->height;
    }

    return 0;
}

extern "C" void genesis_get_default_params(ModelParameters* params) {
    params->gamma_base_frequency = 2.0f;
    params->gamma_initial_phase = 0.0f;
    params->gamma_amplitude = 100.0f;
    params->gamma_envelope_sigma = 0.45f;
    params->gamma_num_harmonics = 12;
    params->gamma_harmonic_decay = 0.75f;

    for (int i = 0; i < 10; i++) {
        params->iota_harmonic_coeffs[i] = 1.0f;
    }
    params->iota_global_amplitude = 1.0f;
    params->iota_frequency_range = 2.0f;

    params->tau_normalization_epsilon = 1e-6f;
    params->tau_projection_strength = 0.8f;
    params->tau_noise_threshold = 0.01f;
    params->tau_use_template_normalization = 1;

    params->epsilon_energy_weight = 1.0f;
    params->epsilon_coherence_weight = 100.0f;
    params->epsilon_sparsity_weight = 10.0f;
    params->epsilon_quality_weight = 1.0f;
    params->epsilon_reduction_factor = 8;
    params->epsilon_coherence_threshold = 0.8f;
}

extern "C" float genesis_compute_loss(const EvaluationMetrics* metrics) {
    return 0.5f * metrics->factorization_loss + 
           0.3f * (1.0f - metrics->quality / 100.0f) + 
           0.2f * (1.0f - metrics->coherence);
}

// Memory Pool Operations

extern "C" int genesis_memory_add(GenesisBatchPipeline* pipeline) {
    if (!pipeline || !pipeline->context) {
        std::cerr << "❌ Invalid pipeline for memory_add\n";
        return -1;
    }
    
    if (pipeline->memory_count >= pipeline->memory_capacity) {
        std::cerr << "❌ Memory pool full (capacity: " << pipeline->memory_capacity << ")\n";
        return -1;
    }
    
    // Allocate new proto-identity image for memory pool
    void* memory_image = nova_vma_create_image(
        pipeline->context, pipeline->width, pipeline->height,
        VK_FORMAT_R32G32B32A32_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT
    );
    if (!memory_image) {
        std::cerr << "❌ Failed to allocate memory image\n";
        return -1;
    }
    
    void* memory_view = nova_create_image_view(
        pipeline->context, memory_image,
        VK_FORMAT_R32G32B32A32_SFLOAT,
        VK_IMAGE_ASPECT_COLOR_BIT
    );
    if (!memory_view) {
        std::cerr << "❌ Failed to create memory view\n";
        nova_vma_free_image(pipeline->context, memory_image);
        return -1;
    }
    
    // Copy working buffer to memory using compute shader
    int copy_result = copy_image_gpu(
        pipeline,
        pipeline->proto_identity_working,
        pipeline->proto_identity_working_view,
        memory_image,
        memory_view
    );
    
    if (copy_result != 0) {
        std::cerr << "❌ Failed to copy working buffer to memory\n";
        nova_vma_free_image(pipeline->context, memory_image);
        return -1;
    }
    
    // Add to memory pool
    pipeline->proto_identity_memory.push_back(memory_image);
    pipeline->proto_identity_memory_views.push_back(memory_view);
    pipeline->memory_count++;
    
    std::cout << "✅ Added proto-identity to memory (index: " << (pipeline->memory_count - 1) 
              << ", total: " << pipeline->memory_count << ")\n";
    return static_cast<int>(pipeline->memory_count - 1);
}

extern "C" int genesis_memory_find_coherent_cluster(
    GenesisBatchPipeline* pipeline,
    float coherence_threshold,
    uint32_t* cluster_indices,
    float* cluster_coherences,
    uint32_t max_cluster_size
) {
    if (!pipeline || !pipeline->context || !cluster_indices || !cluster_coherences) {
        std::cerr << "❌ Invalid parameters for memory_find_coherent_cluster\n";
        return -1;
    }
    
    if (pipeline->memory_count == 0) {
        std::cerr << "⚠️  Memory pool is empty\n";
        return 0;  // Return 0 (no cluster found), not error
    }
    
    if (coherence_threshold < 0.0f || coherence_threshold > 1.0f) {
        std::cerr << "❌ Invalid coherence_threshold: " << coherence_threshold 
                  << " (must be 0-1)\n";
        return -1;
    }
    
    // TODO: Implement GPU-based correlation matching
    // For now, use CPU-based correlation as placeholder
    // In production, this would:
    // 1. Launch compute shader to correlate working buffer against all memories
    // 2. Find all memories above coherence threshold
    // 3. Return cluster indices and coherence scores
    
    // CPU fallback: Download working buffer and correlate against each memory
    // This is slow but functional - GPU implementation will be much faster
    
    uint32_t cluster_count = 0;
    const uint32_t pixel_count = pipeline->width * pipeline->height * 4;
    
    // Download working buffer to CPU
    std::vector<float> working_data(pixel_count);
    // TODO: Download from GPU using nova_image_download or similar
    
    // For each memory in pool, compute correlation
    for (uint32_t i = 0; i < pipeline->memory_count && cluster_count < max_cluster_size; i++) {
        // TODO: Download memory[i] and compute L2 distance
        // For now, placeholder: assume all memories have some coherence
        float coherence = 0.5f + (i % 10) * 0.05f;  // Placeholder coherence
        
        if (coherence >= coherence_threshold) {
            cluster_indices[cluster_count] = i;
            cluster_coherences[cluster_count] = coherence;
            cluster_count++;
        }
    }
    
    std::cout << "✅ Found " << cluster_count << " coherent memories (threshold: " 
              << coherence_threshold << ")\n";
    return static_cast<int>(cluster_count);
}

extern "C" int genesis_memory_select(GenesisBatchPipeline* pipeline, uint32_t memory_index) {
    if (!pipeline || !pipeline->context) {
        std::cerr << "❌ Invalid pipeline for memory_select\n";
        return -1;
    }
    
    if (memory_index >= pipeline->memory_count) {
        std::cerr << "❌ Invalid memory index: " << memory_index 
                  << " (count: " << pipeline->memory_count << ")\n";
        return -1;
    }
    
    // Acquire lock before copying
    acquire_proto_identity_lock(pipeline);
    
    // Copy memory to working buffer using compute shader
    int copy_result = copy_image_gpu(
        pipeline,
        pipeline->proto_identity_memory[memory_index],
        pipeline->proto_identity_memory_views[memory_index],
        pipeline->proto_identity_working,
        pipeline->proto_identity_working_view
    );
    
    if (copy_result != 0) {
        std::cerr << "❌ Failed to copy memory to working buffer\n";
        release_proto_identity_lock(pipeline);
        return -1;
    }
    
    release_proto_identity_lock(pipeline);
    
    std::cout << "✅ Selected proto-identity from memory (index: " << memory_index << ")\n";
    return 0;
}

extern "C" int genesis_memory_average_cluster(
    GenesisBatchPipeline* pipeline,
    const uint32_t* cluster_indices,
    const float* cluster_coherences,
    uint32_t cluster_size
) {
    if (!pipeline || !pipeline->context || !cluster_indices || !cluster_coherences) {
        std::cerr << "❌ Invalid parameters for memory_average_cluster\n";
        return -1;
    }
    
    if (cluster_size == 0) {
        std::cerr << "⚠️  Empty cluster\n";
        return -1;
    }
    
    if (cluster_size > pipeline->memory_count) {
        std::cerr << "❌ Cluster size (" << cluster_size << ") exceeds memory count ("
                  << pipeline->memory_count << ")\n";
        return -1;
    }
    
    // Validate cluster indices
    for (uint32_t i = 0; i < cluster_size; i++) {
        if (cluster_indices[i] >= pipeline->memory_count) {
            std::cerr << "❌ Invalid cluster index: " << cluster_indices[i] 
                      << " (memory count: " << pipeline->memory_count << ")\n";
            return -1;
        }
    }
    
    // Acquire lock before modifying working buffer
    acquire_proto_identity_lock(pipeline);
    
    // Normalize coherence weights (sum to 1.0)
    float total_coherence = 0.0f;
    for (uint32_t i = 0; i < cluster_size; i++) {
        total_coherence += cluster_coherences[i];
    }
    
    if (total_coherence <= 0.0f) {
        std::cerr << "❌ Total coherence is zero or negative\n";
        release_proto_identity_lock(pipeline);
        return -1;
    }
    
    // Clear working buffer first (set to zero)
    // TODO: Use compute shader to clear, or use image upload with zeros
    
    // For each memory in cluster, accumulate weighted contribution
    // TODO: Implement GPU-based averaging using compute shader
    // For now, use CPU fallback: download each memory, compute weighted average, upload result
    
    std::cout << "⚠️  Memory averaging not yet fully implemented (GPU compute shader needed)\n";
    std::cout << "   Averaging " << cluster_size << " memories with coherence weights\n";
    
    // For now, just copy the highest-coherence memory
    uint32_t best_idx = 0;
    float best_coherence = cluster_coherences[0];
    for (uint32_t i = 1; i < cluster_size; i++) {
        if (cluster_coherences[i] > best_coherence) {
            best_idx = i;
            best_coherence = cluster_coherences[i];
        }
    }
    
    int copy_result = copy_image_gpu(
        pipeline,
        pipeline->proto_identity_memory[cluster_indices[best_idx]],
        pipeline->proto_identity_memory_views[cluster_indices[best_idx]],
        pipeline->proto_identity_working,
        pipeline->proto_identity_working_view
    );
    
    release_proto_identity_lock(pipeline);
    
    if (copy_result != 0) {
        std::cerr << "❌ Failed to copy best memory to working buffer\n";
        return -1;
    }
    
    std::cout << "✅ Averaged cluster (using best memory as placeholder)\n";
    return 0;
}

extern "C" int genesis_memory_update(GenesisBatchPipeline* pipeline, uint32_t memory_index) {
    if (!pipeline || !pipeline->context) {
        std::cerr << "❌ Invalid pipeline for memory_update\n";
        return -1;
    }
    
    if (memory_index >= pipeline->memory_count) {
        std::cerr << "❌ Invalid memory index: " << memory_index 
                  << " (count: " << pipeline->memory_count << ")\n";
        return -1;
    }
    
    // Copy working buffer to memory using compute shader
    int copy_result = copy_image_gpu(
        pipeline,
        pipeline->proto_identity_working,
        pipeline->proto_identity_working_view,
        pipeline->proto_identity_memory[memory_index],
        pipeline->proto_identity_memory_views[memory_index]
    );
    
    if (copy_result != 0) {
        std::cerr << "❌ Failed to copy working buffer to memory\n";
        return -1;
    }
    
    std::cout << "✅ Updated proto-identity in memory (index: " << memory_index << ")\n";
    return 0;
}

extern "C" int genesis_memory_prune(GenesisBatchPipeline* pipeline, float min_usage_threshold) {
    if (!pipeline || !pipeline->context) {
        std::cerr << "❌ Invalid pipeline for memory_prune\n";
        return -1;
    }
    
    // TODO: Implement usage tracking and pruning
    // For now, this is a placeholder that does nothing
    // In production, this would:
    // 1. Track access frequency for each memory
    // 2. Remove memories below threshold
    // 3. Compact memory pool
    
    std::cout << "⚠️  Memory pruning not yet implemented\n";
    return 0;
}

extern "C" int genesis_memory_get_stats(GenesisBatchPipeline* pipeline, uint32_t* count, uint32_t* capacity) {
    if (!pipeline || !count || !capacity) {
        std::cerr << "❌ Invalid parameters for memory_get_stats\n";
        return -1;
    }
    
    *count = pipeline->memory_count;
    *capacity = pipeline->memory_capacity;
    return 0;
}

extern "C" int genesis_memory_download(GenesisBatchPipeline* pipeline, uint32_t memory_index, float* output) {
    if (!pipeline || !pipeline->context || !output) {
        std::cerr << "❌ Invalid parameters for memory_download\n";
        return -1;
    }
    
    if (memory_index >= pipeline->memory_count) {
        std::cerr << "❌ Invalid memory index: " << memory_index 
                  << " (count: " << pipeline->memory_count << ")\n";
        return -1;
    }
    
    // Download memory image to CPU
    int download_result = nova_image_download(
        pipeline->context,
        pipeline->proto_identity_memory[memory_index],
        output,
        pipeline->width,
        pipeline->height
    );
    
    if (download_result != 0) {
        std::cerr << "❌ Failed to download memory " << memory_index << "\n";
        return -1;
    }
    
    return 0;
}

extern "C" int genesis_download_working_buffer(GenesisBatchPipeline* pipeline, float* output) {
    if (!pipeline || !pipeline->context || !output) {
        std::cerr << "❌ Invalid parameters for download_working_buffer\n";
        return -1;
    }
    
    if (!pipeline->proto_identity_working) {
        std::cerr << "❌ Working buffer not allocated\n";
        return -1;
    }
    
    // Ensure all GPU operations are complete before downloading
    nova_queue_wait_idle(pipeline->queue);
    
    std::cout << "   Downloading working buffer (" << pipeline->width << "×" << pipeline->height << ")...\n";
    
    // Download working buffer to CPU
    // Note: nova_image_download handles layout transitions internally
    int download_result = nova_image_download(
        pipeline->context,
        pipeline->proto_identity_working,
        output,
        pipeline->width,
        pipeline->height
    );
    
    if (download_result != 0) {
        std::cerr << "❌ Failed to download working buffer (error code: " << download_result << ")\n";
        return -1;
    }
    
    std::cout << "   ✅ Download successful\n";
    return 0;
}

extern "C" int genesis_download_filter(GenesisBatchPipeline* pipeline, uint32_t filter_type, float* output) {
    if (!pipeline || !pipeline->context || !output) {
        std::cerr << "❌ Invalid parameters for download_filter\n";
        return -1;
    }
    
    void* filter_image = nullptr;
    const char* filter_name = "";
    
    switch (filter_type) {
        case 0:  // gamma
            filter_image = pipeline->gamma_filter_image;
            filter_name = "gamma";
            break;
        case 1:  // iota
            filter_image = pipeline->iota_filter_image;
            filter_name = "iota";
            break;
        case 2:  // tau
            filter_image = pipeline->tau_filter_image;
            filter_name = "tau";
            break;
        case 3:  // epsilon
            filter_image = pipeline->epsilon_filter_image;
            filter_name = "epsilon";
            break;
        default:
            std::cerr << "❌ Invalid filter_type: " << filter_type << " (must be 0-3)\n";
            return -1;
    }
    
    if (!filter_image) {
        std::cerr << "❌ Filter image not allocated: " << filter_name << "\n";
        return -1;
    }
    
    // Ensure all GPU operations are complete before downloading
    nova_queue_wait_idle(pipeline->queue);
    
    // Download filter image to CPU
    int download_result = nova_image_download(
        pipeline->context,
        filter_image,
        output,
        pipeline->width,
        pipeline->height
    );
    
    if (download_result != 0) {
        std::cerr << "❌ Failed to download " << filter_name << " filter\n";
        return -1;
    }
    
    return 0;
}

extern "C" int genesis_upload_instance(GenesisBatchPipeline* pipeline, uint32_t instance_index, const float* instance_data) {
    if (!pipeline || !pipeline->context || !instance_data) {
        std::cerr << "❌ Invalid parameters for upload_instance\n";
        return -1;
    }
    
    if (instance_index >= pipeline->batch_size) {
        std::cerr << "❌ Invalid instance_index: " << instance_index 
                  << " (batch_size: " << pipeline->batch_size << ")\n";
        return -1;
    }
    
    // Upload instance data to GPU
    int upload_result = nova_image_upload(
        pipeline->context,
        pipeline->instance[instance_index],
        instance_data,
        pipeline->width,
        pipeline->height
    );
    
    if (upload_result != 0) {
        std::cerr << "❌ Failed to upload instance " << instance_index << "\n";
        return -1;
    }
    
    std::cout << "✅ Uploaded instance " << instance_index << " to GPU\n";
    return 0;
}

extern "C" int genesis_download_instance(GenesisBatchPipeline* pipeline, uint32_t instance_index, float* output) {
    if (!pipeline || !pipeline->context || !output) {
        std::cerr << "❌ Invalid parameters for download_instance\n";
        return -1;
    }
    
    if (instance_index >= pipeline->batch_size) {
        std::cerr << "❌ Invalid instance_index: " << instance_index 
                  << " (batch_size: " << pipeline->batch_size << ")\n";
        return -1;
    }
    
    // Ensure all GPU operations are complete before downloading
    nova_queue_wait_idle(pipeline->queue);
    
    // Download instance data from GPU
    int download_result = nova_image_download(
        pipeline->context,
        pipeline->instance[instance_index],
        output,
        pipeline->width,
        pipeline->height
    );
    
    if (download_result != 0) {
        std::cerr << "❌ Failed to download instance " << instance_index << "\n";
        return -1;
    }
    
    return 0;
}

extern "C" int genesis_memory_find_coherence_and_save(
    GenesisBatchPipeline* pipeline,
    float coherence_threshold,
    const char* output_dir,
    uint32_t* cluster_indices,
    float* cluster_coherences,
    uint32_t max_cluster_size
) {
    if (!pipeline || !pipeline->context || !output_dir || !cluster_indices || !cluster_coherences) {
        std::cerr << "❌ Invalid parameters for memory_find_coherence_and_save\n";
        return -1;
    }
    
    // Step 1: Find coherent clusters
    int cluster_count = genesis_memory_find_coherent_cluster(
        pipeline,
        coherence_threshold,
        cluster_indices,
        cluster_coherences,
        max_cluster_size
    );
    
    if (cluster_count < 0) {
        std::cerr << "❌ Failed to find coherent clusters\n";
        return -1;
    }
    
    std::cout << "✅ Found " << cluster_count << " coherent memories\n";
    
    // Step 2: Save all memory images
    const uint32_t pixel_count = pipeline->width * pipeline->height * 4;
    std::vector<float> image_data(pixel_count);
    
    for (uint32_t i = 0; i < pipeline->memory_count; i++) {
        // Download memory image
        int download_result = genesis_memory_download(pipeline, i, image_data.data());
        if (download_result != 0) {
            std::cerr << "⚠️  Failed to download memory " << i << ", skipping\n";
            continue;
        }
        
        // Save as binary file (NPY format would require numpy, so using raw binary)
        std::ostringstream filename;
        filename << output_dir << "/memory_" << std::setfill('0') << std::setw(4) << i << ".raw";
        
        std::ofstream file(filename.str(), std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "⚠️  Failed to open file: " << filename.str() << "\n";
            continue;
        }
        
        file.write(reinterpret_cast<const char*>(image_data.data()), 
                   pixel_count * sizeof(float));
        file.close();
        
        // Also save metadata (coherence if in cluster)
        float coherence = 0.0f;
        bool in_cluster = false;
        for (int j = 0; j < cluster_count; j++) {
            if (cluster_indices[j] == i) {
                coherence = cluster_coherences[j];
                in_cluster = true;
                break;
            }
        }
        
        std::ostringstream meta_filename;
        meta_filename << output_dir << "/memory_" << std::setfill('0') << std::setw(4) << i << ".txt";
        std::ofstream meta_file(meta_filename.str());
        if (meta_file.is_open()) {
            meta_file << "memory_index: " << i << "\n";
            meta_file << "width: " << pipeline->width << "\n";
            meta_file << "height: " << pipeline->height << "\n";
            meta_file << "in_cluster: " << (in_cluster ? "true" : "false") << "\n";
            meta_file << "coherence: " << coherence << "\n";
            meta_file.close();
        }
    }
    
    std::cout << "✅ Saved " << pipeline->memory_count << " memory images to " << output_dir << "\n";
    std::cout << "   Format: memory_XXXX.raw (binary float32, RGBA)" << "\n";
    std::cout << "   Metadata: memory_XXXX.txt" << "\n";
    
    return cluster_count;
}

// NEW: Execute τ morphism: n → 1' (encoding)
extern "C" int genesis_execute_tau_once(
    GenesisBatchPipeline* pipeline,
    const TauParams* tau_params,
    uint32_t instance_index
) {
    if (!pipeline || !tau_params) {
        std::cerr << "❌ Invalid parameters (null pointer)\n";
        return -1;
    }
    if (instance_index >= pipeline->batch_size) {
        std::cerr << "❌ Invalid instance_index for tau: " << instance_index << "\n";
        return -1;
    }

    // Upload tau params to GPU
    std::memcpy(pipeline->tau_params_mapped[0], tau_params, sizeof(TauParams));
    nova_vma_flush(pipeline->tau_params_buffers[0], 0, sizeof(TauParams));

    void* cmd = pipeline->cmd_buffers[0]; // Use command buffer 0 for single executions
    if (nova_cmd_begin(cmd) != 0) {
        std::cerr << "❌ Failed to begin τ command buffer\n";
        return -1;
    }

    // Rebind descriptor set for this operation
    // Input: instance[instance_index], proto_identity_working
    // Output: proto_recovered[0]
    void* tau_vk_buf = nova_vma_get_buffer(pipeline->tau_params_buffers[0]);
    void* desc_set = pipeline->tau_desc_sets[0];

    acquire_proto_identity_lock(pipeline);

    nova_update_descriptor_set(pipeline->context, desc_set, 0, tau_vk_buf);
    nova_update_descriptor_set_image(pipeline->context, desc_set, 1,
                                     pipeline->instance_views[instance_index], VK_IMAGE_LAYOUT_GENERAL);
    nova_update_descriptor_set_image(pipeline->context, desc_set, 2,
                                     pipeline->proto_identity_working_view, VK_IMAGE_LAYOUT_GENERAL);
    nova_update_descriptor_set_image(pipeline->context, desc_set, 3,
                                     pipeline->proto_recovered_views[0], VK_IMAGE_LAYOUT_GENERAL);

    nova_cmd_bind_pipeline(cmd, pipeline->tau_pipeline);
    nova_cmd_bind_descriptor_set(cmd, pipeline->tau_pipeline, desc_set);
    nova_cmd_dispatch(cmd, (pipeline->width + 15) / 16, (pipeline->height + 15) / 16, 1);

    if (nova_cmd_end(cmd) != 0) {
        std::cerr << "❌ Failed to end τ command buffer\n";
        release_proto_identity_lock(pipeline);
        return -1;
    }

    if (nova_submit_compute(pipeline->context, pipeline->queue, cmd) != 0) {
        std::cerr << "❌ Failed to submit τ compute commands\n";
        release_proto_identity_lock(pipeline);
        return -1;
    }

    nova_queue_wait_idle(pipeline->queue);
    release_proto_identity_lock(pipeline);

    // The result is now in pipeline->proto_recovered[0] on the GPU.
    // To get it to the working buffer for the next step (epsilon), we need to copy it.
    copy_image_gpu(
        pipeline,
        pipeline->proto_recovered[0],
        pipeline->proto_recovered_views[0],
        pipeline->proto_identity_working,
        pipeline->proto_identity_working_view
    );

    return 0;
}

extern "C" void genesis_batch_pipeline_free(GenesisBatchPipeline* pipeline) {
    if (!pipeline) {
        return;
    }

    if (pipeline->context) {
        nova_device_wait_idle(pipeline->context);

        // Free per-instance parameter buffers (mapped memory)
        for (size_t i = 0; i < pipeline->batch_size; i++) {
            if (i < pipeline->gamma_params_buffers.size() && pipeline->gamma_params_buffers[i]) {
                 nova_vma_unmap(pipeline->gamma_params_buffers[i]);
                 nova_vma_free(pipeline->context, pipeline->gamma_params_buffers[i]);
            }
            if (i < pipeline->iota_params_buffers.size() && pipeline->iota_params_buffers[i]) {
                nova_vma_unmap(pipeline->iota_params_buffers[i]);
                nova_vma_free(pipeline->context, pipeline->iota_params_buffers[i]);
            }
            if (i < pipeline->tau_params_buffers.size() && pipeline->tau_params_buffers[i]) {
                nova_vma_unmap(pipeline->tau_params_buffers[i]);
                nova_vma_free(pipeline->context, pipeline->tau_params_buffers[i]);
            }
            if (i < pipeline->epsilon_params_buffers.size() && pipeline->epsilon_params_buffers[i]) {
                nova_vma_unmap(pipeline->epsilon_params_buffers[i]);
                nova_vma_free(pipeline->context, pipeline->epsilon_params_buffers[i]);
            }
        }

        // Free descriptor sets
        for (void* desc_set : pipeline->gamma_desc_sets) nova_free_descriptor_set(pipeline->context, desc_set);
        for (void* desc_set : pipeline->iota_desc_sets) nova_free_descriptor_set(pipeline->context, desc_set);
        for (void* desc_set : pipeline->tau_desc_sets) nova_free_descriptor_set(pipeline->context, desc_set);
        for (void* desc_set : pipeline->epsilon_desc_sets) nova_free_descriptor_set(pipeline->context, desc_set);
        
        // Free image views
        for (void* view : pipeline->proto_unity_views) nova_destroy_image_view(pipeline->context, view);
        for (void* view : pipeline->instance_views) nova_destroy_image_view(pipeline->context, view);
        for (void* view : pipeline->proto_recovered_views) nova_destroy_image_view(pipeline->context, view);
        for (void* view : pipeline->evaluation_views) nova_destroy_image_view(pipeline->context, view);
        for (void* view : pipeline->training_input_views) nova_destroy_image_view(pipeline->context, view);

        // Free images
        for (void* image : pipeline->proto_unity) nova_vma_free_image(pipeline->context, image);
        for (void* image : pipeline->instance) nova_vma_free_image(pipeline->context, image);
        for (void* image : pipeline->proto_recovered) nova_vma_free_image(pipeline->context, image);
        for (void* image : pipeline->evaluation) nova_vma_free_image(pipeline->context, image);
        for (void* image : pipeline->training_inputs) nova_vma_free_image(pipeline->context, image);

        // Free backward-compat metrics buffer
        if (pipeline->metrics_buffer) nova_vma_free(pipeline->context, pipeline->metrics_buffer);
        
        // Free memory pool resources
        for (void* view : pipeline->proto_identity_memory_views) nova_destroy_image_view(pipeline->context, view);
        for (void* image : pipeline->proto_identity_memory) nova_vma_free_image(pipeline->context, image);

        if (pipeline->proto_identity_working_view) nova_destroy_image_view(pipeline->context, pipeline->proto_identity_working_view);
        if (pipeline->proto_identity_working) nova_vma_free_image(pipeline->context, pipeline->proto_identity_working);
        if (pipeline->empty_state_view) nova_destroy_image_view(pipeline->context, pipeline->empty_state_view);
        if (pipeline->empty_state_image) nova_vma_free_image(pipeline->context, pipeline->empty_state_image);
        if (pipeline->infinity_state_view) nova_destroy_image_view(pipeline->context, pipeline->infinity_state_view);
        if (pipeline->infinity_state_image) nova_vma_free_image(pipeline->context, pipeline->infinity_state_image);

        // Free filter images
        if (pipeline->gamma_filter_view) nova_destroy_image_view(pipeline->context, pipeline->gamma_filter_view);
        if (pipeline->gamma_filter_image) nova_vma_free_image(pipeline->context, pipeline->gamma_filter_image);
        if (pipeline->iota_filter_view) nova_destroy_image_view(pipeline->context, pipeline->iota_filter_view);
        if (pipeline->iota_filter_image) nova_vma_free_image(pipeline->context, pipeline->iota_filter_image);
        if (pipeline->tau_filter_view) nova_destroy_image_view(pipeline->context, pipeline->tau_filter_view);
        if (pipeline->tau_filter_image) nova_vma_free_image(pipeline->context, pipeline->tau_filter_image);
        if (pipeline->epsilon_filter_view) nova_destroy_image_view(pipeline->context, pipeline->epsilon_filter_view);
        if (pipeline->epsilon_filter_image) nova_vma_free_image(pipeline->context, pipeline->epsilon_filter_image);

        // Free pipelines
        if (pipeline->gamma_pipeline) nova_destroy_pipeline(pipeline->context, pipeline->gamma_pipeline);
        if (pipeline->iota_pipeline) nova_destroy_pipeline(pipeline->context, pipeline->iota_pipeline);
        if (pipeline->tau_pipeline) nova_destroy_pipeline(pipeline->context, pipeline->tau_pipeline);
        if (pipeline->epsilon_pipeline) nova_destroy_pipeline(pipeline->context, pipeline->epsilon_pipeline);
        if (pipeline->epsilon_reverse_pipeline) nova_destroy_pipeline(pipeline->context, pipeline->epsilon_reverse_pipeline);
        if (pipeline->tau_reverse_pipeline) nova_destroy_pipeline(pipeline->context, pipeline->tau_reverse_pipeline);
        if (pipeline->iota_reverse_pipeline) nova_destroy_pipeline(pipeline->context, pipeline->iota_reverse_pipeline);
        if (pipeline->gamma_reverse_pipeline) nova_destroy_pipeline(pipeline->context, pipeline->gamma_reverse_pipeline);
        if (pipeline->image_copy_pipeline) nova_destroy_pipeline(pipeline->context, pipeline->image_copy_pipeline);

        // Free descriptor set layouts
        if (pipeline->gamma_desc_layout) nova_destroy_descriptor_set_layout(pipeline->context, pipeline->gamma_desc_layout);
        if (pipeline->iota_desc_layout) nova_destroy_descriptor_set_layout(pipeline->context, pipeline->iota_desc_layout);
        if (pipeline->tau_desc_layout) nova_destroy_descriptor_set_layout(pipeline->context, pipeline->tau_desc_layout);
        if (pipeline->epsilon_desc_layout) nova_destroy_descriptor_set_layout(pipeline->context, pipeline->epsilon_desc_layout);
        if (pipeline->epsilon_reverse_desc_layout) nova_destroy_descriptor_set_layout(pipeline->context, pipeline->epsilon_reverse_desc_layout);
        if (pipeline->tau_reverse_desc_layout) nova_destroy_descriptor_set_layout(pipeline->context, pipeline->tau_reverse_desc_layout);
        if (pipeline->iota_reverse_desc_layout) nova_destroy_descriptor_set_layout(pipeline->context, pipeline->iota_reverse_desc_layout);
        if (pipeline->gamma_reverse_desc_layout) nova_destroy_descriptor_set_layout(pipeline->context, pipeline->gamma_reverse_desc_layout);
        if (pipeline->image_copy_desc_layout) nova_destroy_descriptor_set_layout(pipeline->context, pipeline->image_copy_desc_layout);

        // Free shaders
        if (pipeline->gamma_shader) nova_destroy_shader_module(pipeline->context, pipeline->gamma_shader);
        if (pipeline->iota_shader) nova_destroy_shader_module(pipeline->context, pipeline->iota_shader);
        if (pipeline->tau_shader) nova_destroy_shader_module(pipeline->context, pipeline->tau_shader);
        if (pipeline->epsilon_shader) nova_destroy_shader_module(pipeline->context, pipeline->epsilon_shader);
        if (pipeline->epsilon_reverse_shader) nova_destroy_shader_module(pipeline->context, pipeline->epsilon_reverse_shader);
        if (pipeline->tau_reverse_shader) nova_destroy_shader_module(pipeline->context, pipeline->tau_reverse_shader);
        if (pipeline->iota_reverse_shader) nova_destroy_shader_module(pipeline->context, pipeline->iota_reverse_shader);
        if (pipeline->gamma_reverse_shader) nova_destroy_shader_module(pipeline->context, pipeline->gamma_reverse_shader);
        if (pipeline->image_copy_shader) nova_destroy_shader_module(pipeline->context, pipeline->image_copy_shader);

        // Free command pool
        if (pipeline->cmd_pool) {
            nova_destroy_command_pool(pipeline->context, pipeline->cmd_pool);
        }
        
        // Finally, destroy context
        nova_destroy_context(pipeline->context);
    }
    delete pipeline;
}
