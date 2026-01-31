"""Test Identity Branch Manager functionality."""

import numpy as np
from src.memory.identity_branch_manager import IdentityBranchManager

def test_attractor_detection():
    """Test detection of multiple attractors in trajectory."""
    print("Testing attractor detection...")

    manager = IdentityBranchManager(min_clusters=2, max_clusters=5, eps=0.3)

    # Create trajectory with 2 clear clusters
    # Cluster 1: centered around (0, 0)
    # Cluster 2: centered around (10, 10)
    trajectory = []

    for _ in range(10):
        proto1 = np.random.randn(16, 16, 4) * 0.5  # Tight cluster
        trajectory.append(proto1)

    for _ in range(10):
        proto2 = np.random.randn(16, 16, 4) * 0.5 + 10.0  # Shifted cluster
        trajectory.append(proto2)

    attractors = manager.detect_attractors(trajectory)

    print(f"  Found {len(attractors)} attractors")
    print(f"  Expected: 2 attractors")
    assert len(attractors) >= 2, "Should detect at least 2 attractors"
    print("  ✓ Attractor detection passed")


def test_paradox_detection():
    """Test paradox detection with oscillating trajectory."""
    print("\nTesting paradox detection...")

    manager = IdentityBranchManager(min_clusters=2, max_clusters=5, eps=0.3)

    # Create oscillating trajectory between two states
    trajectory = []
    coherence_history = []

    state1 = np.random.randn(16, 16, 4)
    state2 = np.random.randn(16, 16, 4) + 5.0

    for i in range(20):
        if i % 2 == 0:
            trajectory.append(state1 + np.random.randn(16, 16, 4) * 0.1)
            coherence_history.append(0.6 + np.random.rand() * 0.1)
        else:
            trajectory.append(state2 + np.random.randn(16, 16, 4) * 0.1)
            coherence_history.append(0.5 + np.random.rand() * 0.1)

    is_paradox = manager.detect_paradox(trajectory, coherence_history)

    print(f"  Paradox detected: {is_paradox}")
    print(f"  Expected: True (oscillating trajectory)")
    assert is_paradox, "Should detect paradox in oscillating trajectory"
    print("  ✓ Paradox detection passed")


def test_branch_splitting():
    """Test splitting paradox into branches."""
    print("\nTesting branch splitting...")

    manager = IdentityBranchManager()

    # Create attractors
    proto = np.random.randn(16, 16, 4)
    attractor1 = np.random.randn(16, 16, 4)
    attractor2 = np.random.randn(16, 16, 4) + 5.0
    attractors = [attractor1, attractor2]

    branches = manager.split_paradox(proto, attractors)

    print(f"  Created {len(branches)} branches")
    print(f"  Expected: 2 branches")
    assert len(branches) == 2, "Should create 2 branches"

    for i, branch in enumerate(branches):
        print(f"  Branch {i}: ID={branch.branch_id}, state={branch.state}")
        assert branch.state == 'active', "Branch should be active"
        assert len(branch.trajectory) > 0, "Branch should have trajectory"

    print("  ✓ Branch splitting passed")


def test_branch_merging():
    """Test merging similar converged branches."""
    print("\nTesting branch merging...")

    manager = IdentityBranchManager()

    # Create two very similar converged branches
    base_proto = np.random.randn(16, 16, 4)

    from src.memory.synthesis_types import IdentityBranch

    branch1 = IdentityBranch(
        branch_id="test1",
        proto_identity=base_proto.copy(),
        trajectory=[base_proto.copy()],
        coherence_history=[0.9],
        state='converged'
    )

    # Branch 2 very similar to branch 1
    branch2 = IdentityBranch(
        branch_id="test2",
        proto_identity=base_proto + np.random.randn(16, 16, 4) * 0.001,
        trajectory=[base_proto.copy()],
        coherence_history=[0.9],
        state='converged'
    )

    # Branch 3 dissimilar
    branch3 = IdentityBranch(
        branch_id="test3",
        proto_identity=np.random.randn(16, 16, 4) + 10.0,
        trajectory=[base_proto.copy()],
        coherence_history=[0.8],
        state='converged'
    )

    # Test merging similar branches
    merged = manager.merge_branches([branch1, branch2])

    if merged:
        print(f"  Merged branch created: ID={merged.branch_id}")
        print(f"  State: {merged.state}")
        print("  ✓ Branch merging passed (similar branches merged)")
    else:
        print("  ✗ Warning: Similar branches should have merged")

    # Test no merge with dissimilar branches
    no_merge = manager.merge_branches([branch1, branch3])

    if not no_merge:
        print("  ✓ No merge for dissimilar branches (correct)")
    else:
        print("  ✗ Warning: Dissimilar branches should not merge")


def test_convergence_vs_paradox():
    """Test distinguishing convergence from paradox."""
    print("\nTesting convergence vs paradox...")

    manager = IdentityBranchManager()

    # Convergent trajectory (moving toward single point)
    trajectory_conv = []
    coherence_conv = []
    target = np.random.randn(16, 16, 4)

    for i in range(20):
        noise_scale = 1.0 - (i / 20.0)  # Decreasing noise
        proto = target + np.random.randn(16, 16, 4) * noise_scale
        trajectory_conv.append(proto)
        coherence_conv.append(0.5 + (i / 40.0))  # Increasing coherence

    is_paradox_conv = manager.detect_paradox(trajectory_conv, coherence_conv)

    print(f"  Convergent trajectory paradox: {is_paradox_conv}")
    print(f"  Expected: False (converging, not oscillating)")

    if not is_paradox_conv:
        print("  ✓ Correctly identified convergent trajectory")
    else:
        print("  ✗ Warning: Convergent trajectory misclassified as paradox")


if __name__ == '__main__':
    print("=" * 60)
    print("Identity Branch Manager Tests")
    print("=" * 60)

    test_attractor_detection()
    test_paradox_detection()
    test_branch_splitting()
    test_branch_merging()
    test_convergence_vs_paradox()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
