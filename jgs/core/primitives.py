"""
Complex Gaussian primitive implementation for RF signal processing.

This module defines the fundamental building blocks of the complex-valued
Gaussian Splatting system - individual Gaussian primitives that represent
localized RF field distributions.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ComplexGaussianPrimitive:
    """
    A single complex-valued Gaussian primitive for RF field representation.
    
    This class represents a 3D Gaussian distribution with complex amplitude,
    used as a building block for representing RF electromagnetic fields.
    """
    
    def __init__(
        self,
        position: torch.Tensor,
        complex_value: torch.Tensor,
        scale: torch.Tensor,
        rotation: torch.Tensor,
        opacity: float = 1.0
    ):
        """
        Initialize a complex Gaussian primitive.
        
        Args:
            position: 3D center position (3,)
            complex_value: Complex amplitude value
            scale: Scaling factors for each axis (3,)
            rotation: Rotation quaternion (w, x, y, z) (4,)
            opacity: Opacity/strength factor [0, 1]
        """
        self.position = position
        self.complex_value = complex_value
        self.scale = scale
        self.rotation = rotation
        self.opacity = opacity
        
        # Precompute rotation matrix from quaternion
        self.rotation_matrix = self._quaternion_to_rotation_matrix(rotation)
        
        # Precompute covariance matrix
        self.covariance_matrix = self._compute_covariance_matrix()
        self.inv_covariance = torch.inverse(self.covariance_matrix)
        self.det_covariance = torch.det(self.covariance_matrix)
    
    def _quaternion_to_rotation_matrix(self, q: torch.Tensor) -> torch.Tensor:
        """
        Convert quaternion to rotation matrix.
        
        Args:
            q: Quaternion (w, x, y, z) (4,)
            
        Returns:
            Rotation matrix (3, 3)
        """
        # Normalize quaternion
        q = F.normalize(q, dim=0)
        w, x, y, z = q[0], q[1], q[2], q[3]
        
        # Compute rotation matrix elements
        R = torch.zeros((3, 3), device=q.device, dtype=q.dtype)
        
        R[0, 0] = 1 - 2 * (y*y + z*z)
        R[0, 1] = 2 * (x*y - w*z)
        R[0, 2] = 2 * (x*z + w*y)
        
        R[1, 0] = 2 * (x*y + w*z)
        R[1, 1] = 1 - 2 * (x*x + z*z)
        R[1, 2] = 2 * (y*z - w*x)
        
        R[2, 0] = 2 * (x*z - w*y)
        R[2, 1] = 2 * (y*z + w*x)
        R[2, 2] = 1 - 2 * (x*x + y*y)
        
        return R
    
    def _compute_covariance_matrix(self) -> torch.Tensor:
        """
        Compute the covariance matrix from scale and rotation.
        
        Returns:
            Covariance matrix (3, 3)
        """
        # Create diagonal scale matrix
        S = torch.diag(self.scale ** 2)
        
        # Apply rotation: Σ = R * S * R^T
        covariance = self.rotation_matrix @ S @ self.rotation_matrix.T
        
        return covariance
    
    def evaluate(
        self, 
        query_points: torch.Tensor,
        frequency: Optional[float] = None
    ) -> torch.Tensor:
        """
        Evaluate the Gaussian primitive at query points.
        
        Args:
            query_points: Points to evaluate at (N, 3)
            frequency: Optional frequency for phase calculations
            
        Returns:
            Complex field values at query points (N,)
        """
        # Compute relative positions
        diff = query_points - self.position.unsqueeze(0)  # (N, 3)
        
        # Compute Mahalanobis distance squared
        # d² = (x - μ)ᵀ Σ⁻¹ (x - μ)
        mahalanobis_sq = torch.sum(diff @ self.inv_covariance * diff, dim=1)  # (N,)
        
        # Compute Gaussian weight
        normalization = 1.0 / torch.sqrt((2 * np.pi) ** 3 * self.det_covariance)
        gaussian_weight = normalization * torch.exp(-0.5 * mahalanobis_sq)
        
        # Apply opacity
        gaussian_weight *= self.opacity
        
        # Apply complex amplitude
        if frequency is not None:
            # Add frequency-dependent phase shift
            distance = torch.norm(diff, dim=1)
            phase_shift = torch.exp(1j * 2 * np.pi * frequency * distance / 3e8)  # c = 3e8 m/s
            complex_amplitude = self.complex_value * phase_shift
        else:
            complex_amplitude = self.complex_value
        
        # Combine Gaussian weight with complex amplitude
        result = gaussian_weight * complex_amplitude
        
        return result
    
    def evaluate_gradient(
        self, 
        query_points: torch.Tensor,
        frequency: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate the Gaussian primitive and its spatial gradient.
        
        Args:
            query_points: Points to evaluate at (N, 3)
            frequency: Optional frequency for phase calculations
            
        Returns:
            Tuple of (field_values, gradient) where gradient is (N, 3)
        """
        # Enable gradient computation
        query_points.requires_grad_(True)
        
        # Evaluate the primitive
        field_values = self.evaluate(query_points, frequency)
        
        # Compute gradient
        if field_values.requires_grad:
            gradient = torch.autograd.grad(
                outputs=field_values.sum(),
                inputs=query_points,
                create_graph=True,
                retain_graph=True
            )[0]
        else:
            gradient = torch.zeros_like(query_points)
        
        return field_values, gradient
    
    def get_bounding_box(self, sigma_threshold: float = 3.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get axis-aligned bounding box containing significant field values.
        
        Args:
            sigma_threshold: Number of standard deviations to include
            
        Returns:
            Tuple of (min_bounds, max_bounds) each of shape (3,)
        """
        # Get eigenvalues and eigenvectors of covariance matrix
        eigenvals, eigenvecs = torch.linalg.eigh(self.covariance_matrix)
        
        # Compute extent in each principal direction
        extents = sigma_threshold * torch.sqrt(eigenvals)
        
        # Transform to world coordinates
        world_extents = torch.abs(eigenvecs @ torch.diag(extents))
        max_extent = torch.max(world_extents, dim=1)[0]
        
        min_bounds = self.position - max_extent
        max_bounds = self.position + max_extent
        
        return min_bounds, max_bounds
    
    def update_position(self, new_position: torch.Tensor):
        """Update the position of the primitive."""
        self.position = new_position
    
    def update_complex_value(self, new_complex_value: torch.Tensor):
        """Update the complex amplitude of the primitive."""
        self.complex_value = new_complex_value
    
    def update_scale(self, new_scale: torch.Tensor):
        """Update the scale of the primitive."""
        self.scale = new_scale
        self.covariance_matrix = self._compute_covariance_matrix()
        self.inv_covariance = torch.inverse(self.covariance_matrix)
        self.det_covariance = torch.det(self.covariance_matrix)
    
    def update_rotation(self, new_rotation: torch.Tensor):
        """Update the rotation of the primitive."""
        self.rotation = new_rotation
        self.rotation_matrix = self._quaternion_to_rotation_matrix(new_rotation)
        self.covariance_matrix = self._compute_covariance_matrix()
        self.inv_covariance = torch.inverse(self.covariance_matrix)
        self.det_covariance = torch.det(self.covariance_matrix)
    
    def clone(self) -> 'ComplexGaussianPrimitive':
        """Create a copy of this primitive."""
        return ComplexGaussianPrimitive(
            position=self.position.clone(),
            complex_value=self.complex_value.clone(),
            scale=self.scale.clone(),
            rotation=self.rotation.clone(),
            opacity=self.opacity
        )
    
    def to_dict(self) -> dict:
        """Convert primitive to dictionary representation."""
        return {
            'position': self.position.detach().cpu().numpy(),
            'complex_value': self.complex_value.detach().cpu().numpy(),
            'scale': self.scale.detach().cpu().numpy(),
            'rotation': self.rotation.detach().cpu().numpy(),
            'opacity': self.opacity
        }
    
    @classmethod
    def from_dict(cls, data: dict, device: torch.device) -> 'ComplexGaussianPrimitive':
        """Create primitive from dictionary representation."""
        return cls(
            position=torch.tensor(data['position'], device=device),
            complex_value=torch.tensor(data['complex_value'], device=device),
            scale=torch.tensor(data['scale'], device=device),
            rotation=torch.tensor(data['rotation'], device=device),
            opacity=data['opacity']
        )
