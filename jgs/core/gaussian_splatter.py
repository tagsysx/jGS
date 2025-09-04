"""
Complex-valued Gaussian Splatting implementation for RF signal processing.

This module implements the core Gaussian Splatting algorithm adapted for 
complex-valued RF signals, enabling efficient representation and manipulation
of electromagnetic field data.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Union
import logging

from ..utils.complex_math import ComplexMath
from .primitives import ComplexGaussianPrimitive

logger = logging.getLogger(__name__)


class ComplexGaussianSplatter(nn.Module):
    """
    Complex-valued Gaussian Splatting model for RF signal processing.
    
    This class implements a differentiable renderer for complex-valued RF signals
    using Gaussian primitives to represent electromagnetic field distributions.
    """
    
    def __init__(
        self,
        positions: Union[np.ndarray, torch.Tensor],
        complex_values: Union[np.ndarray, torch.Tensor],
        scales: Optional[Union[np.ndarray, torch.Tensor]] = None,
        rotations: Optional[Union[np.ndarray, torch.Tensor]] = None,
        device: str = 'cuda',
        dtype: torch.dtype = torch.complex64
    ):
        """
        Initialize the Complex Gaussian Splatting model.
        
        Args:
            positions: 3D positions of Gaussian primitives (N, 3)
            complex_values: Complex amplitudes at each position (N,) or (N, C)
            scales: Scaling factors for each Gaussian (N, 3), defaults to unit scales
            rotations: Rotation quaternions for each Gaussian (N, 4), defaults to identity
            device: Device to run computations on ('cuda' or 'cpu')
            dtype: Data type for complex computations
        """
        super().__init__()
        
        self.device = torch.device(device)
        self.dtype = dtype
        self.complex_math = ComplexMath()
        
        # Convert inputs to tensors
        self.positions = self._to_tensor(positions, torch.float32)
        self.complex_values = self._to_tensor(complex_values, dtype)
        
        # Initialize scales and rotations if not provided
        n_gaussians = self.positions.shape[0]
        
        if scales is None:
            scales = torch.ones((n_gaussians, 3), device=self.device, dtype=torch.float32)
        else:
            scales = self._to_tensor(scales, torch.float32)
        
        if rotations is None:
            # Identity quaternions (w, x, y, z)
            rotations = torch.zeros((n_gaussians, 4), device=self.device, dtype=torch.float32)
            rotations[:, 0] = 1.0  # w component
        else:
            rotations = self._to_tensor(rotations, torch.float32)
        
        # Register as parameters for optimization
        self.register_parameter('_positions', nn.Parameter(self.positions))
        self.register_parameter('_complex_values', nn.Parameter(self.complex_values))
        self.register_parameter('_scales', nn.Parameter(scales))
        self.register_parameter('_rotations', nn.Parameter(rotations))
        
        # Initialize primitives
        self._update_primitives()
        
        logger.info(f"Initialized ComplexGaussianSplatter with {n_gaussians} primitives on {device}")
    
    def _to_tensor(self, data: Union[np.ndarray, torch.Tensor], dtype: torch.dtype) -> torch.Tensor:
        """Convert input data to tensor on the correct device."""
        if isinstance(data, np.ndarray):
            tensor = torch.from_numpy(data).to(dtype=dtype, device=self.device)
        else:
            tensor = data.to(dtype=dtype, device=self.device)
        return tensor
    
    def _update_primitives(self):
        """Update internal Gaussian primitives based on current parameters."""
        self.primitives = []
        for i in range(self._positions.shape[0]):
            primitive = ComplexGaussianPrimitive(
                position=self._positions[i],
                complex_value=self._complex_values[i],
                scale=self._scales[i],
                rotation=self._rotations[i]
            )
            self.primitives.append(primitive)
    
    @property
    def positions(self) -> torch.Tensor:
        """Get current positions of Gaussian primitives."""
        return self._positions
    
    @property
    def complex_values(self) -> torch.Tensor:
        """Get current complex values of Gaussian primitives."""
        return self._complex_values
    
    @property
    def scales(self) -> torch.Tensor:
        """Get current scales of Gaussian primitives."""
        return self._scales
    
    @property
    def rotations(self) -> torch.Tensor:
        """Get current rotations of Gaussian primitives."""
        return self._rotations
    
    def render(
        self, 
        query_points: Union[np.ndarray, torch.Tensor],
        frequency: Optional[float] = None
    ) -> torch.Tensor:
        """
        Render complex RF field at specified query points.
        
        Args:
            query_points: 3D points to evaluate field at (M, 3)
            frequency: Optional frequency for phase calculations
            
        Returns:
            Complex field values at query points (M,) or (M, C)
        """
        query_points = self._to_tensor(query_points, torch.float32)
        
        # Initialize output tensor
        output_shape = (query_points.shape[0],) + self._complex_values.shape[1:]
        rendered_field = torch.zeros(output_shape, dtype=self.dtype, device=self.device)
        
        # Evaluate each Gaussian primitive
        for i, primitive in enumerate(self.primitives):
            contribution = primitive.evaluate(query_points, frequency)
            rendered_field += contribution
        
        return rendered_field
    
    def forward(self, query_points: torch.Tensor, frequency: Optional[float] = None) -> torch.Tensor:
        """Forward pass for neural network compatibility."""
        return self.render(query_points, frequency)
    
    def add_primitive(
        self,
        position: torch.Tensor,
        complex_value: torch.Tensor,
        scale: Optional[torch.Tensor] = None,
        rotation: Optional[torch.Tensor] = None
    ):
        """
        Add a new Gaussian primitive to the model.
        
        Args:
            position: 3D position (3,)
            complex_value: Complex amplitude
            scale: Scaling factors (3,), defaults to unit scale
            rotation: Rotation quaternion (4,), defaults to identity
        """
        if scale is None:
            scale = torch.ones(3, device=self.device, dtype=torch.float32)
        if rotation is None:
            rotation = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device, dtype=torch.float32)
        
        # Expand parameter tensors
        new_positions = torch.cat([self._positions, position.unsqueeze(0)], dim=0)
        new_complex_values = torch.cat([self._complex_values, complex_value.unsqueeze(0)], dim=0)
        new_scales = torch.cat([self._scales, scale.unsqueeze(0)], dim=0)
        new_rotations = torch.cat([self._rotations, rotation.unsqueeze(0)], dim=0)
        
        # Update parameters
        self._positions = nn.Parameter(new_positions)
        self._complex_values = nn.Parameter(new_complex_values)
        self._scales = nn.Parameter(new_scales)
        self._rotations = nn.Parameter(new_rotations)
        
        self._update_primitives()
        
        logger.info(f"Added new primitive. Total primitives: {len(self.primitives)}")
    
    def remove_primitive(self, index: int):
        """Remove a Gaussian primitive by index."""
        if not (0 <= index < len(self.primitives)):
            raise ValueError(f"Invalid primitive index: {index}")
        
        # Create mask to exclude the specified index
        mask = torch.ones(self._positions.shape[0], dtype=torch.bool, device=self.device)
        mask[index] = False
        
        # Update parameters
        self._positions = nn.Parameter(self._positions[mask])
        self._complex_values = nn.Parameter(self._complex_values[mask])
        self._scales = nn.Parameter(self._scales[mask])
        self._rotations = nn.Parameter(self._rotations[mask])
        
        self._update_primitives()
        
        logger.info(f"Removed primitive {index}. Total primitives: {len(self.primitives)}")
    
    def get_field_magnitude(self, query_points: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Get magnitude of the complex field at query points."""
        field = self.render(query_points)
        return torch.abs(field)
    
    def get_field_phase(self, query_points: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Get phase of the complex field at query points."""
        field = self.render(query_points)
        return torch.angle(field)
    
    def save_state(self, filepath: str):
        """Save model state to file."""
        state = {
            'positions': self._positions.detach().cpu(),
            'complex_values': self._complex_values.detach().cpu(),
            'scales': self._scales.detach().cpu(),
            'rotations': self._rotations.detach().cpu(),
            'device': str(self.device),
            'dtype': str(self.dtype)
        }
        torch.save(state, filepath)
        logger.info(f"Saved model state to {filepath}")
    
    def load_state(self, filepath: str):
        """Load model state from file."""
        state = torch.load(filepath, map_location=self.device)
        
        self._positions = nn.Parameter(state['positions'].to(self.device))
        self._complex_values = nn.Parameter(state['complex_values'].to(self.device))
        self._scales = nn.Parameter(state['scales'].to(self.device))
        self._rotations = nn.Parameter(state['rotations'].to(self.device))
        
        self._update_primitives()
        logger.info(f"Loaded model state from {filepath}")
