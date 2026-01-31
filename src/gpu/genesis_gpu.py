#!/usr/bin/env python3
"""
Python wrapper for the Genesis Rust GPU pipeline.
"""

import ctypes
import numpy as np
from pathlib import Path
from typing import List, Tuple

# --- Ctypes structures for FFI ---

class CGammaParams(ctypes.Structure):
    _fields_ = [
        ("base_frequency", ctypes.c_float),
        ("initial_phase", ctypes.c_float),
        ("amplitude", ctypes.c_float),
        ("envelope_sigma", ctypes.c_float),
        ("num_harmonics", ctypes.c_uint32),
        ("harmonic_decay", ctypes.c_float),
        ("_pad", ctypes.c_uint32 * 2),
    ]

class CTauParams(ctypes.Structure):
    _fields_ = [
        ("normalization_epsilon", ctypes.c_float),
        ("projection_strength", ctypes.c_float),
        ("noise_threshold", ctypes.c_float),
        ("use_template_normalization", ctypes.c_uint32),
        ("_pad", ctypes.c_uint32 * 4),
    ]

class CIotaParams(ctypes.Structure):
    _fields_ = [
        ("harmonic_coeffs", ctypes.c_float * 10),
        ("global_amplitude", ctypes.c_float),
        ("frequency_range", ctypes.c_float),
        ("_pad", ctypes.c_uint32 * 1),
    ]

class CEpsilonParams(ctypes.Structure):
    _fields_ = [
        ("energy_weight", ctypes.c_float),
        ("coherence_weight", ctypes.c_float),
        ("sparsity_weight", ctypes.c_float),
        ("quality_weight", ctypes.c_float),
        ("reduction_factor", ctypes.c_uint32),
        ("coherence_threshold", ctypes.c_float),
        ("_pad", ctypes.c_uint32 * 2),
    ]

class CTrainingData(ctypes.Structure):
    _fields_ = [
        ("waveform", ctypes.POINTER(ctypes.c_float)),
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
    ]

class CProtoEmbedding(ctypes.Structure):
    _fields_ = [
        ("embedding", ctypes.POINTER(ctypes.c_float)),
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
    ]

class GenesisBatchPipeline:
    def __init__(self, batch_size: int = 16, width: int = 512, height: int = 512, lib_path: str = None):
        if lib_path is None:
            possible_paths = [
                "./target/release/libgenesis.so",
                "target/release/libgenesis.so",
            ]
            for path in possible_paths:
                if Path(path).exists():
                    lib_path = path
                    break
        
        if lib_path is None:
            raise FileNotFoundError("Could not find libgenesis.so. Please compile the Rust code first.")

        self.lib = ctypes.CDLL(lib_path)
        self.width = width
        self.height = height

        # --- Function signatures ---
        self.lib.genesis_pipeline_init.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
        self.lib.genesis_pipeline_init.restype = ctypes.c_void_p

        self.lib.genesis_pipeline_free.argtypes = [ctypes.c_void_p]

        self.lib.genesis_execute_gamma_once.argtypes = [ctypes.c_void_p, ctypes.POINTER(CGammaParams)]
        self.lib.genesis_execute_gamma_once.restype = ctypes.c_int32
        
        self.lib.genesis_download_working_buffer.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
        self.lib.genesis_download_working_buffer.restype = ctypes.c_int32

        self.lib.genesis_upload_instance.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_float)]
        self.lib.genesis_upload_instance.restype = ctypes.c_int32
        
        # ... other function signatures ...

        self.pipeline = self.lib.genesis_pipeline_init(batch_size, width, height)
        if not self.pipeline:
            raise RuntimeError("Failed to initialize Rust pipeline")

    def __del__(self):
        if hasattr(self, 'pipeline') and self.pipeline:
            self.lib.genesis_pipeline_free(self.pipeline)

    def execute_gamma_once(self, gamma_params: dict):
        c_params = CGammaParams(**gamma_params)
        result = self.lib.genesis_execute_gamma_once(self.pipeline, ctypes.byref(c_params))
        if result != 0:
            raise RuntimeError("Failed to execute gamma")
            
    def download_working_buffer(self) -> np.ndarray:
        output = np.zeros((self.height, self.width, 4), dtype=np.float32)
        result = self.lib.genesis_download_working_buffer(self.pipeline, output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        if result != 0:
            raise RuntimeError("Failed to download working buffer")
        return output

    def upload_instance(self, instance_index: int, instance_data: np.ndarray):
        result = self.lib.genesis_upload_instance(self.pipeline, instance_index, instance_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        if result != 0:
            raise RuntimeError("Failed to upload instance")

    # ... other wrapper methods ...

if __name__ == '__main__':
    pipeline = GenesisBatchPipeline()
    print("Genesis Rust pipeline initialized.")
    # Add more example usage here
    print("Genesis Rust pipeline shut down.")