#include "pipeline_internal.h"

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
    nova_update_descriptor_set_image(pipeline->context, desc_set, 0, src_view, 1 /* VK_IMAGE_LAYOUT_GENERAL */);
    nova_update_descriptor_set_image(pipeline->context, desc_set, 1, dst_view, 1 /* VK_IMAGE_LAYOUT_GENERAL */);
    
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
        cmd, src_image, 1, 1,
        0x00000800, 0x00000800,
        0x00000020, 0x00000020
    );
    nova_cmd_pipeline_barrier_image(
        cmd, dst_image, 1, 1,
        0x00000800, 0x00000800,
        0x00000040, 0x00000040
    );
    
    // Bind pipeline and descriptor set
    nova_cmd_bind_pipeline(cmd, pipeline->image_copy_pipeline);
    nova_cmd_bind_descriptor_set(cmd, pipeline->image_copy_pipeline, desc_set);
    
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
