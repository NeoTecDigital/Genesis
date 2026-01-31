"""Dirac gamma matrices for spinor field theory."""

import numpy as np
from typing import Dict, Optional, Tuple


class GammaMatrices:
    """
    Dirac gamma matrices in Dirac-Pauli representation.

    Properties:
    - {γ^μ, γ^ν} = 2g^{μν}I (anticommutation)
    - (γ⁰)² = I, (γⁱ)² = -I
    - γ⁵ = iγ⁰γ¹γ²γ³
    """

    def __init__(self, representation: str = "dirac"):
        """
        Initialize gamma matrices and cache products.

        Args:
            representation: Matrix representation ("dirac" or "weyl")
                           Currently only "dirac" is implemented
        """
        if representation != "dirac":
            raise NotImplementedError(f"Representation '{representation}' not implemented")

        self.representation = representation

        # Pauli matrices (2×2)
        self.sigma_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        self.sigma_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
        self.sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
        self.I2 = np.eye(2, dtype=np.complex128)

        # Construct gamma matrices (4×4)
        self.gamma = self._construct_gammas()
        self.gamma5 = self._construct_gamma5()

        # Add gamma5 to the dictionary
        self.gamma[5] = self.gamma5

        # Cache commonly used products
        self._gamma_products = self._cache_products()

        # Validate construction
        if not self.validate():
            raise ValueError("Gamma matrices failed validation")

    def _construct_gammas(self) -> Dict[int, np.ndarray]:
        """
        Construct γ⁰, γ¹, γ², γ³ in Dirac-Pauli basis.

        Returns:
            Dictionary mapping index to 4×4 gamma matrix
        """
        gamma = {}

        # γ⁰ = [[I₂, 0], [0, -I₂]]
        gamma[0] = np.block([
            [self.I2, np.zeros((2, 2), dtype=np.complex128)],
            [np.zeros((2, 2), dtype=np.complex128), -self.I2]
        ])

        # γ¹ = [[0, σₓ], [-σₓ, 0]]
        gamma[1] = np.block([
            [np.zeros((2, 2), dtype=np.complex128), self.sigma_x],
            [-self.sigma_x, np.zeros((2, 2), dtype=np.complex128)]
        ])

        # γ² = [[0, σᵧ], [-σᵧ, 0]]
        gamma[2] = np.block([
            [np.zeros((2, 2), dtype=np.complex128), self.sigma_y],
            [-self.sigma_y, np.zeros((2, 2), dtype=np.complex128)]
        ])

        # γ³ = [[0, σᵤ], [-σᵤ, 0]]
        gamma[3] = np.block([
            [np.zeros((2, 2), dtype=np.complex128), self.sigma_z],
            [-self.sigma_z, np.zeros((2, 2), dtype=np.complex128)]
        ])

        return gamma

    def _construct_gamma5(self) -> np.ndarray:
        """
        Construct γ⁵ = iγ⁰γ¹γ²γ³.

        Returns:
            4×4 gamma5 matrix
        """
        # Compute product
        gamma5 = 1j * self.gamma[0] @ self.gamma[1] @ self.gamma[2] @ self.gamma[3]

        # In Dirac-Pauli representation, this should give [[0, I₂], [I₂, 0]]
        # Let's verify and use the exact form for numerical stability
        expected = np.block([
            [np.zeros((2, 2), dtype=np.complex128), self.I2],
            [self.I2, np.zeros((2, 2), dtype=np.complex128)]
        ])

        # Use expected form if close enough (avoid numerical errors)
        if np.allclose(gamma5, expected, atol=1e-10):
            gamma5 = expected

        return gamma5

    def _cache_products(self) -> Dict[Tuple[int, int], np.ndarray]:
        """
        Cache commonly used gamma matrix products.

        Returns:
            Dictionary of (μ, ν) -> γ^μ γ^ν products
        """
        products = {}

        # Cache all unique products
        for mu in range(4):
            for nu in range(4):
                if mu <= nu:  # Only store unique pairs
                    products[(mu, nu)] = self.gamma[mu] @ self.gamma[nu]

        # Add some useful triple products
        products[(0, 1, 2)] = self.gamma[0] @ self.gamma[1] @ self.gamma[2]
        products[(1, 2, 3)] = self.gamma[1] @ self.gamma[2] @ self.gamma[3]

        return products

    def apply(self, matrix: np.ndarray, spinor: np.ndarray) -> np.ndarray:
        """
        Apply 4×4 matrix to spinor field.

        Args:
            matrix: 4×4 gamma matrix
            spinor: (..., 4) spinor field

        Returns:
            (..., 4) result of matrix·spinor at each point
        """
        # Use einsum for efficient vectorized multiplication
        # matrix: (4, 4), spinor: (..., 4) -> result: (..., 4)
        return np.einsum('ab,...b->...a', matrix, spinor, optimize='optimal')

    def anticommutator(self, mu: int, nu: int) -> np.ndarray:
        """
        Compute anticommutator {γ^μ, γ^ν} = γ^μγ^ν + γ^νγ^μ.

        Args:
            mu: First index (0-3)
            nu: Second index (0-3)

        Returns:
            4×4 anticommutator matrix
        """
        return self.gamma[mu] @ self.gamma[nu] + self.gamma[nu] @ self.gamma[mu]

    def get_product(self, mu: int, nu: int) -> np.ndarray:
        """
        Get cached product γ^μ γ^ν.

        Args:
            mu: First index
            nu: Second index

        Returns:
            4×4 product matrix
        """
        key = (mu, nu) if mu <= nu else (nu, mu)
        if key in self._gamma_products:
            result = self._gamma_products[key]
            # If reversed order, need to compute
            if mu > nu:
                result = self.gamma[mu] @ self.gamma[nu]
        else:
            # Compute on the fly if not cached
            result = self.gamma[mu] @ self.gamma[nu]

        return result

    def validate(self, tol: float = 1e-10) -> bool:
        """
        Validate anticommutation relations and other properties.

        Args:
            tol: Numerical tolerance

        Returns:
            True if all validations pass
        """
        # Minkowski metric signature (+, -, -, -) for standard Dirac-Pauli
        # This gives {γ^μ, γ^ν} = 2g^{μν}I
        metric = np.diag([1.0, -1.0, -1.0, -1.0])
        I4 = np.eye(4, dtype=np.complex128)

        # Check anticommutation relations: {γ^μ, γ^ν} = 2g^{μν}I
        for mu in range(4):
            for nu in range(4):
                anticomm = self.anticommutator(mu, nu)
                expected = 2 * metric[mu, nu] * I4
                if not np.allclose(anticomm, expected, atol=tol):
                    print(f"Anticommutation failed for γ^{mu}, γ^{nu}")
                    return False

        # Check (γ⁵)² = I
        gamma5_squared = self.gamma5 @ self.gamma5
        if not np.allclose(gamma5_squared, I4, atol=tol):
            print("γ⁵ squared != I")
            return False

        # Check {γ⁵, γ^μ} = 0 for all μ
        for mu in range(4):
            anticomm = self.gamma5 @ self.gamma[mu] + self.gamma[mu] @ self.gamma5
            if not np.allclose(anticomm, np.zeros((4, 4)), atol=tol):
                print(f"γ⁵ doesn't anticommute with γ^{mu}")
                return False

        # Check hermiticity: γ⁰ is hermitian, γⁱ are anti-hermitian
        if not np.allclose(self.gamma[0], self.gamma[0].conj().T, atol=tol):
            print("γ⁰ not hermitian")
            return False

        for i in range(1, 4):
            if not np.allclose(self.gamma[i], -self.gamma[i].conj().T, atol=tol):
                print(f"γ^{i} not anti-hermitian")
                return False

        return True

    def chiral_projectors(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get left and right chiral projection operators.

        Returns:
            P_L = (I - γ⁵)/2, P_R = (I + γ⁵)/2
        """
        I4 = np.eye(4, dtype=np.complex128)
        P_L = (I4 - self.gamma5) / 2
        P_R = (I4 + self.gamma5) / 2
        return P_L, P_R