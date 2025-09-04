"""
Complex mathematics utilities for RF signal processing.

This module provides specialized mathematical operations for complex-valued
data commonly used in RF and electromagnetic field calculations.
"""

import torch
import numpy as np
from typing import Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


class ComplexMath:
    """
    Utility class for complex mathematical operations.
    
    This class provides static methods for common complex number operations
    used in RF signal processing and electromagnetic field calculations.
    """
    
    @staticmethod
    def complex_to_polar(complex_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert complex values to polar form (magnitude, phase).
        
        Args:
            complex_values: Complex tensor (any shape)
            
        Returns:
            Tuple of (magnitude, phase) tensors
        """
        magnitude = torch.abs(complex_values)
        phase = torch.angle(complex_values)
        return magnitude, phase
    
    @staticmethod
    def polar_to_complex(magnitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        """
        Convert polar form to complex values.
        
        Args:
            magnitude: Magnitude tensor
            phase: Phase tensor (in radians)
            
        Returns:
            Complex tensor
        """
        return magnitude * torch.exp(1j * phase)
    
    @staticmethod
    def db_to_linear(db_values: torch.Tensor) -> torch.Tensor:
        """Convert dB values to linear scale."""
        return 10 ** (db_values / 10)
    
    @staticmethod
    def linear_to_db(linear_values: torch.Tensor, epsilon: float = 1e-12) -> torch.Tensor:
        """Convert linear values to dB scale."""
        return 10 * torch.log10(torch.abs(linear_values) + epsilon)
    
    @staticmethod
    def phase_unwrap(phase: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """
        Unwrap phase values to remove 2π discontinuities.
        
        Args:
            phase: Phase tensor in radians
            dim: Dimension along which to unwrap
            
        Returns:
            Unwrapped phase tensor
        """
        # Convert to numpy for unwrapping (PyTorch doesn't have native unwrap)
        phase_np = phase.detach().cpu().numpy()
        unwrapped_np = np.unwrap(phase_np, axis=dim)
        return torch.tensor(unwrapped_np, device=phase.device, dtype=phase.dtype)
    
    @staticmethod
    def complex_interpolate(
        values1: torch.Tensor,
        values2: torch.Tensor,
        alpha: float
    ) -> torch.Tensor:
        """
        Interpolate between two complex tensors.
        
        Args:
            values1: First complex tensor
            values2: Second complex tensor
            alpha: Interpolation factor [0, 1]
            
        Returns:
            Interpolated complex tensor
        """
        # Interpolate in polar coordinates to handle phase properly
        mag1, phase1 = ComplexMath.complex_to_polar(values1)
        mag2, phase2 = ComplexMath.complex_to_polar(values2)
        
        # Handle phase wrapping for interpolation
        phase_diff = phase2 - phase1
        phase_diff = torch.angle(torch.exp(1j * phase_diff))  # Wrap to [-π, π]
        
        interp_mag = (1 - alpha) * mag1 + alpha * mag2
        interp_phase = phase1 + alpha * phase_diff
        
        return ComplexMath.polar_to_complex(interp_mag, interp_phase)
    
    @staticmethod
    def complex_gradient(
        complex_field: torch.Tensor,
        positions: torch.Tensor,
        method: str = 'central'
    ) -> torch.Tensor:
        """
        Compute spatial gradient of complex field.
        
        Args:
            complex_field: Complex field values (N,)
            positions: Spatial positions (N, 3)
            method: Gradient method ('central', 'forward', 'backward')
            
        Returns:
            Complex gradient tensor (N, 3)
        """
        if complex_field.requires_grad and positions.requires_grad:
            # Use automatic differentiation if available
            gradients = torch.autograd.grad(
                outputs=complex_field.sum(),
                inputs=positions,
                create_graph=True,
                retain_graph=True
            )[0]
            return gradients
        else:
            # Numerical gradient computation
            return ComplexMath._numerical_gradient(complex_field, positions, method)
    
    @staticmethod
    def _numerical_gradient(
        complex_field: torch.Tensor,
        positions: torch.Tensor,
        method: str
    ) -> torch.Tensor:
        """Compute numerical gradient of complex field."""
        n_points, n_dims = positions.shape
        gradients = torch.zeros((n_points, n_dims), dtype=complex_field.dtype, device=positions.device)
        
        h = 1e-6  # Step size
        
        for dim in range(n_dims):
            if method == 'central':
                # Central difference
                pos_forward = positions.clone()
                pos_backward = positions.clone()
                pos_forward[:, dim] += h
                pos_backward[:, dim] -= h
                
                # This is a simplified version - in practice, you'd need to re-evaluate
                # the field at these new positions
                gradients[:, dim] = (complex_field - complex_field) / (2 * h)  # Placeholder
            
            elif method == 'forward':
                # Forward difference
                pos_forward = positions.clone()
                pos_forward[:, dim] += h
                gradients[:, dim] = (complex_field - complex_field) / h  # Placeholder
            
            elif method == 'backward':
                # Backward difference
                pos_backward = positions.clone()
                pos_backward[:, dim] -= h
                gradients[:, dim] = (complex_field - complex_field) / h  # Placeholder
        
        return gradients
    
    @staticmethod
    def complex_correlation(
        signal1: torch.Tensor,
        signal2: torch.Tensor,
        mode: str = 'full'
    ) -> torch.Tensor:
        """
        Compute complex cross-correlation between two signals.
        
        Args:
            signal1: First complex signal
            signal2: Second complex signal
            mode: Correlation mode ('full', 'valid', 'same')
            
        Returns:
            Complex correlation result
        """
        # Use FFT-based correlation for efficiency
        n1, n2 = len(signal1), len(signal2)
        
        if mode == 'full':
            n_out = n1 + n2 - 1
        elif mode == 'valid':
            n_out = max(n1, n2) - min(n1, n2) + 1
        else:  # 'same'
            n_out = max(n1, n2)
        
        # Pad signals for FFT
        n_fft = 2 ** int(np.ceil(np.log2(n_out)))
        
        fft1 = torch.fft.fft(signal1, n=n_fft)
        fft2 = torch.fft.fft(torch.conj(torch.flip(signal2, [0])), n=n_fft)
        
        correlation = torch.fft.ifft(fft1 * fft2)[:n_out]
        
        return correlation
    
    @staticmethod
    def complex_pca(
        complex_data: torch.Tensor,
        n_components: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform Principal Component Analysis on complex data.
        
        Args:
            complex_data: Complex data matrix (N, D)
            n_components: Number of components to keep
            
        Returns:
            Tuple of (transformed_data, components, explained_variance)
        """
        # Center the data
        mean_data = torch.mean(complex_data, dim=0, keepdim=True)
        centered_data = complex_data - mean_data
        
        # Compute covariance matrix
        cov_matrix = torch.matmul(centered_data.conj().T, centered_data) / (complex_data.shape[0] - 1)
        
        # Eigendecomposition
        eigenvals, eigenvecs = torch.linalg.eigh(cov_matrix)
        
        # Sort by eigenvalues (descending)
        idx = torch.argsort(eigenvals.real, descending=True)
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]
        
        # Select components
        if n_components is not None:
            eigenvals = eigenvals[:n_components]
            eigenvecs = eigenvecs[:, :n_components]
        
        # Transform data
        transformed_data = torch.matmul(centered_data, eigenvecs)
        
        return transformed_data, eigenvecs, eigenvals.real
    
    @staticmethod
    def complex_svd(complex_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform Singular Value Decomposition on complex matrix.
        
        Args:
            complex_matrix: Complex matrix (M, N)
            
        Returns:
            Tuple of (U, S, Vh) where A = U @ diag(S) @ Vh
        """
        U, S, Vh = torch.linalg.svd(complex_matrix, full_matrices=False)
        return U, S, Vh
    
    @staticmethod
    def fresnel_coefficients(
        theta_i: torch.Tensor,
        n1: Union[float, complex] = 1.0,
        n2: Union[float, complex] = 1.5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Fresnel reflection coefficients for electromagnetic waves.
        
        Args:
            theta_i: Incident angles in radians
            n1: Refractive index of first medium
            n2: Refractive index of second medium
            
        Returns:
            Tuple of (r_s, r_p) for s and p polarizations
        """
        # Convert to complex if needed
        n1 = complex(n1)
        n2 = complex(n2)
        
        cos_theta_i = torch.cos(theta_i)
        sin_theta_i = torch.sin(theta_i)
        
        # Snell's law for transmitted angle
        sin_theta_t = (n1 / n2) * sin_theta_i
        cos_theta_t = torch.sqrt(1 - sin_theta_t**2 + 0j)  # Add small imaginary part for stability
        
        # Fresnel coefficients
        r_s = (n1 * cos_theta_i - n2 * cos_theta_t) / (n1 * cos_theta_i + n2 * cos_theta_t)
        r_p = (n2 * cos_theta_i - n1 * cos_theta_t) / (n2 * cos_theta_i + n1 * cos_theta_t)
        
        return r_s, r_p
    
    @staticmethod
    def poynting_vector(
        e_field: torch.Tensor,
        h_field: torch.Tensor,
        mu0: float = 4e-7 * np.pi
    ) -> torch.Tensor:
        """
        Compute Poynting vector from E and H fields.
        
        Args:
            e_field: Electric field (N, 3) complex
            h_field: Magnetic field (N, 3) complex  
            mu0: Permeability of free space
            
        Returns:
            Poynting vector (N, 3) real
        """
        # S = (1/μ₀) * Re(E × H*)
        cross_product = torch.cross(e_field, torch.conj(h_field), dim=-1)
        poynting = torch.real(cross_product) / mu0
        
        return poynting
    
    @staticmethod
    def field_intensity(complex_field: torch.Tensor) -> torch.Tensor:
        """
        Compute field intensity (|E|²).
        
        Args:
            complex_field: Complex field values
            
        Returns:
            Field intensity (real, positive)
        """
        return torch.abs(complex_field) ** 2
    
    @staticmethod
    def phase_velocity(
        frequency: float,
        wavelength: float,
        medium_properties: Optional[dict] = None
    ) -> float:
        """
        Calculate phase velocity in a medium.
        
        Args:
            frequency: Frequency in Hz
            wavelength: Wavelength in meters
            medium_properties: Optional medium properties (permittivity, permeability)
            
        Returns:
            Phase velocity in m/s
        """
        if medium_properties is None:
            # Free space
            return frequency * wavelength
        else:
            eps_r = medium_properties.get('permittivity', 1.0)
            mu_r = medium_properties.get('permeability', 1.0)
            c = 3e8  # Speed of light
            return c / np.sqrt(eps_r * mu_r)
