"""
Refactoring Verification Tests
Tests to verify that Genesis application works after refactoring.
"""

import os
import sys
import traceback
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_imports():
    """Test 1: Verify all critical imports work"""
    print("\n" + "="*80)
    print("TEST 1: Import Verification")
    print("="*80)

    results = {}

    # Test memory module imports
    try:
        from src.memory.fm_modulation_base import FMModulationBase
        results['FMModulationBase'] = 'PASS'
        print("✓ FMModulationBase imported successfully")
    except Exception as e:
        results['FMModulationBase'] = f'FAIL: {str(e)}'
        print(f"✗ FMModulationBase import failed: {e}")
        traceback.print_exc()

    # Test clustering module imports
    try:
        from src.clustering import ProtoUnityClusterer
        results['ProtoUnityClusterer'] = 'PASS'
        print("✓ ProtoUnityClusterer imported successfully")
    except Exception as e:
        results['ProtoUnityClusterer'] = f'FAIL: {str(e)}'
        print(f"✗ ProtoUnityClusterer import failed: {e}")
        traceback.print_exc()

    try:
        from src.clustering_core import kmeans_iteration
        results['kmeans_iteration'] = 'PASS'
        print("✓ kmeans_iteration imported successfully")
    except Exception as e:
        results['kmeans_iteration'] = f'FAIL: {str(e)}'
        print(f"✗ kmeans_iteration import failed: {e}")
        traceback.print_exc()

    # Additional critical imports
    try:
        from src.memory.voxel_cloud import VoxelCloud
        results['VoxelCloud'] = 'PASS'
        print("✓ VoxelCloud imported successfully")
    except Exception as e:
        results['VoxelCloud'] = f'FAIL: {str(e)}'
        print(f"✗ VoxelCloud import failed: {e}")

    try:
        from src.memory.frequency_field import FrequencyField
        results['FrequencyField'] = 'PASS'
        print("✓ FrequencyField imported successfully")
    except Exception as e:
        results['FrequencyField'] = f'FAIL: {str(e)}'
        print(f"✗ FrequencyField import failed: {e}")

    return results


def test_fm_modulation_base_class():
    """Test 2: Verify FMModulationBase functionality"""
    print("\n" + "="*80)
    print("TEST 2: FMModulationBase Functionality")
    print("="*80)

    try:
        from src.memory.fm_modulation_base import FMModulationBase

        # Test instantiation
        fm_base = FMModulationBase(
            base_frequency=440.0,
            num_harmonics=8,
            epsilon=1e-8
        )
        print("✓ FMModulationBase instantiation successful")

        # Test generate_harmonic_series
        harmonics = fm_base.generate_harmonic_series()
        assert harmonics.shape == (8,), f"Expected shape (8,), got {harmonics.shape}"
        assert harmonics[0] == 440.0, f"Expected base freq 440.0, got {harmonics[0]}"
        print(f"✓ generate_harmonic_series works: {harmonics[:3]}")

        # Test compute_modulation_index
        carrier_freq = np.array([440.0, 880.0])
        modulator_freq = np.array([220.0, 440.0])
        mod_index = fm_base.compute_modulation_index(carrier_freq, modulator_freq)
        assert mod_index.shape == (2,), f"Expected shape (2,), got {mod_index.shape}"
        print(f"✓ compute_modulation_index works: {mod_index}")

        # Test compute_bessel_component
        bessel = fm_base.compute_bessel_component(1.0, 1)
        assert isinstance(bessel, (float, np.floating)), f"Expected float, got {type(bessel)}"
        print(f"✓ compute_bessel_component works: {bessel}")

        return 'PASS'
    except Exception as e:
        print(f"✗ FMModulationBase test failed: {e}")
        traceback.print_exc()
        return f'FAIL: {str(e)}'


# REMOVED: test_fm_modulation() - module deleted (broken imports)
# REMOVED: test_fm_modulation_stratified() - module deleted (broken imports)


def test_clustering():
    """Test 5: Verify clustering module functionality"""
    print("\n" + "="*80)
    print("TEST 5: Clustering Module Functionality")
    print("="*80)

    try:
        from src.clustering import ProtoUnityClusterer
        from src.clustering_core import kmeans_iteration

        # Test kmeans_iteration
        data = np.random.randn(100, 64)
        centroids = np.random.randn(4, 64)
        assignments, new_centroids = kmeans_iteration(data, centroids)
        assert assignments.shape == (100,), f"Expected shape (100,), got {assignments.shape}"
        assert new_centroids.shape == (4, 64), f"Expected shape (4, 64), got {new_centroids.shape}"
        print(f"✓ kmeans_iteration works: assignments {assignments.shape}, centroids {new_centroids.shape}")

        # Test ProtoUnityClusterer instantiation
        clusterer = ProtoUnityClusterer(
            n_clusters=4,
            embedding_dim=64,
            max_iter=10
        )
        print("✓ ProtoUnityClusterer instantiation successful")

        # Test fit method
        clusterer.fit(data)
        assert hasattr(clusterer, 'centroids_'), "Clusterer should have centroids_ after fit"
        print(f"✓ ProtoUnityClusterer.fit works: centroids shape {clusterer.centroids_.shape}")

        return 'PASS'
    except Exception as e:
        print(f"✗ Clustering test failed: {e}")
        traceback.print_exc()
        return f'FAIL: {str(e)}'


def main():
    """Run all verification tests"""
    print("\n" + "="*80)
    print("GENESIS REFACTORING VERIFICATION TEST SUITE")
    print("="*80)

    results = {}

    # Run all tests
    import_results = test_imports()
    results['imports'] = import_results

    results['fm_base'] = test_fm_modulation_base_class()
    results['clustering'] = test_clustering()

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    total_tests = 0
    passed_tests = 0

    # Import results
    print("\nImport Tests:")
    for module, result in import_results.items():
        total_tests += 1
        if result == 'PASS':
            passed_tests += 1
            print(f"  ✓ {module}")
        else:
            print(f"  ✗ {module}: {result}")

    # Functionality results
    print("\nFunctionality Tests:")
    for test_name, result in results.items():
        if test_name == 'imports':
            continue
        total_tests += 1
        if result == 'PASS':
            passed_tests += 1
            print(f"  ✓ {test_name}")
        else:
            print(f"  ✗ {test_name}: {result}")

    print(f"\n{'='*80}")
    print(f"FINAL RESULT: {passed_tests}/{total_tests} tests passed")
    print(f"{'='*80}\n")

    return 0 if passed_tests == total_tests else 1


if __name__ == '__main__':
    sys.exit(main())
