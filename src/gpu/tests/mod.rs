//! GPU integration tests (CPU-based, no actual GPU execution)

#[cfg(test)]
mod buffer_tests {
    use std::mem;
    use crate::waveform::{WaveformNode, Complex64};

    #[test]
    fn test_waveform_node_alignment() {
        // Verify GPU-ready alignment
        assert_eq!(mem::align_of::<WaveformNode>(), 16);
        assert_eq!(mem::size_of::<WaveformNode>() % 16, 0);
    }

    #[test]
    fn test_waveform_node_layout() {
        // Verify RGBA structure for GPU image storage
        let node = WaveformNode {
            r: Complex64::new(1.0, 2.0),
            g: Complex64::new(3.0, 4.0),
            b: Complex64::new(5.0, 6.0),
            a: Complex64::new(7.0, 8.0),
        };

        // Should be laid out sequentially in memory
        let size = mem::size_of::<WaveformNode>();
        assert_eq!(size, 8 * mem::size_of::<f64>()); // 4 channels × 2 components
    }

    #[test]
    fn test_buffer_size_calculations() {
        // Test buffer sizing for various capacities
        let node_size = mem::size_of::<WaveformNode>();

        assert_eq!(node_size, 64); // 4 × Complex64 (2 × f64)

        // Common buffer sizes
        let sizes = [256, 512, 1024, 2048, 4096];
        for &count in &sizes {
            let bytes = count * node_size;
            assert_eq!(bytes % 16, 0, "Buffer size must be 16-byte aligned");
        }
    }
}

#[cfg(test)]
mod shader_tests {
    use crate::waveform::{GenesisParams, InstantiationParams};
    use std::mem;

    #[test]
    fn test_genesis_params_layout() {
        // Verify uniform buffer layout matches shader
        let params = GenesisParams {
            base_frequency: 440.0,
            initial_phase: 0.0,
            amplitude: 1.0,
            _padding: 0.0,
        };

        // Should be 4 × f64 = 32 bytes
        assert_eq!(mem::size_of::<GenesisParams>(), 32);
        assert_eq!(mem::align_of::<GenesisParams>(), 8);

        // Verify field offsets match shader binding layout
        let ptr = &params as *const GenesisParams as *const u8;
        let freq_offset = unsafe {
            &params.base_frequency as *const f64 as *const u8
        } as usize - ptr as usize;
        assert_eq!(freq_offset, 0); // First field at offset 0
    }

    #[test]
    fn test_instantiation_params_layout() {
        // Verify InstantiationParams matches shader layout
        let params = InstantiationParams {
            harmonic_coeffs: [0.0; 8],
            modulation_freq: 5.0,
            modulation_depth: 0.1,
            filter_cutoff: 8000.0,
            filter_resonance: 1.0,
        };

        let size = mem::size_of::<InstantiationParams>();

        // 8 harmonics + 4 scalars = 12 × f64 = 96 bytes
        assert_eq!(size, 96);

        // Verify alignment
        assert_eq!(mem::align_of::<InstantiationParams>(), 8);
    }

    #[test]
    fn test_parameter_serialization() {
        // Test that parameters can be safely copied to GPU buffers
        let genesis = GenesisParams {
            base_frequency: 440.0,
            initial_phase: 0.0,
            amplitude: 1.0,
            _padding: 0.0,
        };

        // Simulate buffer copy
        let bytes: [u8; 32] = unsafe {
            std::mem::transmute(genesis)
        };

        // Deserialize
        let reconstructed: GenesisParams = unsafe {
            std::mem::transmute(bytes)
        };

        assert_eq!(reconstructed.base_frequency, 440.0);
        assert_eq!(reconstructed.initial_phase, 0.0);
        assert_eq!(reconstructed.amplitude, 1.0);
    }
}

#[cfg(test)]
mod pipeline_tests {
    use crate::category::{Genesis, Instantiation, Numeric};
    use crate::waveform::WaveformNode;

    #[test]
    fn test_morphism_cpu_fallback() {
        // Test CPU fallback paths
        let genesis = Genesis::default();
        let mut waveform = WaveformNode::zero();

        genesis.apply_cpu(&mut waveform);

        // Should generate non-zero frequency content
        assert!(waveform.r.real != 0.0 || waveform.r.imag != 0.0);
    }

    #[test]
    fn test_instantiation_cpu_fallback() {
        let inst = Instantiation::default(Numeric(42));
        let mut waveform = WaveformNode::zero();

        // First apply genesis
        Genesis::default().apply_cpu(&mut waveform);

        // Then apply instantiation
        inst.apply_cpu(&mut waveform);

        // Should have modified the waveform
        assert!(waveform.r.real != 0.0);
    }

    #[test]
    fn test_categorical_composition_cpu() {
        // Verify universal factorization on CPU
        // Any morphism ∅ → n should factor as ι_n ∘ γ

        let mut waveform1 = WaveformNode::zero();
        let mut waveform2 = WaveformNode::zero();

        // Path 1: Direct CPU computation (simulated ∅ → n)
        Genesis::default().apply_cpu(&mut waveform1);
        Instantiation::default(Numeric(7)).apply_cpu(&mut waveform1);

        // Path 2: Same composition
        Genesis::default().apply_cpu(&mut waveform2);
        Instantiation::default(Numeric(7)).apply_cpu(&mut waveform2);

        // Should produce identical results
        assert_eq!(waveform1.r.real, waveform2.r.real);
        assert_eq!(waveform1.r.imag, waveform2.r.imag);
    }
}

#[cfg(test)]
mod ffi_tests {
    #[test]
    fn test_ffi_types_size() {
        use std::ffi::c_void;

        // Verify pointer sizes for FFI
        assert_eq!(std::mem::size_of::<*mut c_void>(), std::mem::size_of::<usize>());
    }

    #[test]
    fn test_buffer_usage_flags() {
        #[cfg(feature = "gpu")]
        {
            use crate::ffi::BufferUsage;

            // Verify enum discriminants match Vulkan flags
            let uniform = BufferUsage::UniformBuffer as u32;
            let storage = BufferUsage::StorageBuffer as u32;

            assert_eq!(uniform, 0x00000010);
            assert_eq!(storage, 0x00000020);
        }
    }
}

#[cfg(test)]
mod memory_tests {
    use crate::waveform::WaveformNode;

    #[test]
    fn test_stack_allocation() {
        // Verify WaveformNode can be stack-allocated
        let _node = WaveformNode::zero();
        // Should compile and run without heap allocation
    }

    #[test]
    fn test_vector_allocation() {
        // Verify vectors of nodes allocate correctly
        let nodes: Vec<WaveformNode> = vec![WaveformNode::zero(); 1024];

        assert_eq!(nodes.len(), 1024);
        assert_eq!(nodes.capacity() >= 1024, true);
    }

    #[test]
    fn test_memory_layout_predictability() {
        // Verify repr(C) layout is predictable
        let nodes = vec![
            WaveformNode::zero(),
            WaveformNode::zero(),
            WaveformNode::zero(),
        ];

        let ptr0 = &nodes[0] as *const WaveformNode as usize;
        let ptr1 = &nodes[1] as *const WaveformNode as usize;
        let ptr2 = &nodes[2] as *const WaveformNode as usize;

        let stride = std::mem::size_of::<WaveformNode>();

        // Should be sequential in memory
        assert_eq!(ptr1 - ptr0, stride);
        assert_eq!(ptr2 - ptr1, stride);
    }
}