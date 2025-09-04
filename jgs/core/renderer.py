"""
Complex renderer for RF field visualization and computation.

This module provides rendering capabilities for complex-valued RF fields
using the Gaussian Splatting primitives.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, List, Union
import logging

logger = logging.getLogger(__name__)


class ComplexRenderer(nn.Module):
    """
    Renderer for complex-valued RF fields using Gaussian Splatting.
    
    This class handles the rendering pipeline for converting Gaussian primitives
    into complex field values at arbitrary query points in 3D space.
    """
    
    def __init__(
        self,
        device: str = 'cuda',
        dtype: torch.dtype = torch.complex64,
        batch_size: int = 10000
    ):
        """
        Initialize the complex renderer.
        
        Args:
            device: Device to run computations on
            dtype: Data type for complex computations
            batch_size: Batch size for processing large point clouds
        """
        super().__init__()
        
        self.device = torch.device(device)
        self.dtype = dtype
        self.batch_size = batch_size
        
        logger.info(f"Initialized ComplexRenderer on {device}")
    
    def render_field(
        self,
        primitives: List,
        query_points: torch.Tensor,
        frequency: Optional[float] = None,
        use_batching: bool = True
    ) -> torch.Tensor:
        """
        Render complex RF field at query points.
        
        Args:
            primitives: List of ComplexGaussianPrimitive objects
            query_points: Points to evaluate field at (N, 3)
            frequency: Optional frequency for phase calculations
            use_batching: Whether to use batching for large point sets
            
        Returns:
            Complex field values at query points (N,)
        """
        n_points = query_points.shape[0]
        
        if use_batching and n_points > self.batch_size:
            return self._render_batched(primitives, query_points, frequency)
        else:
            return self._render_direct(primitives, query_points, frequency)
    
    def _render_direct(
        self,
        primitives: List,
        query_points: torch.Tensor,
        frequency: Optional[float] = None
    ) -> torch.Tensor:
        """Direct rendering without batching."""
        n_points = query_points.shape[0]
        field_values = torch.zeros(n_points, dtype=self.dtype, device=self.device)
        
        for primitive in primitives:
            contribution = primitive.evaluate(query_points, frequency)
            field_values += contribution
        
        return field_values
    
    def _render_batched(
        self,
        primitives: List,
        query_points: torch.Tensor,
        frequency: Optional[float] = None
    ) -> torch.Tensor:
        """Batched rendering for large point sets."""
        n_points = query_points.shape[0]
        field_values = torch.zeros(n_points, dtype=self.dtype, device=self.device)
        
        for i in range(0, n_points, self.batch_size):
            end_idx = min(i + self.batch_size, n_points)
            batch_points = query_points[i:end_idx]
            
            batch_field = self._render_direct(primitives, batch_points, frequency)
            field_values[i:end_idx] = batch_field
        
        return field_values
    
    def render_magnitude(
        self,
        primitives: List,
        query_points: torch.Tensor,
        frequency: Optional[float] = None
    ) -> torch.Tensor:
        """
        Render field magnitude at query points.
        
        Args:
            primitives: List of ComplexGaussianPrimitive objects
            query_points: Points to evaluate field at (N, 3)
            frequency: Optional frequency for phase calculations
            
        Returns:
            Field magnitude values at query points (N,)
        """
        field_values = self.render_field(primitives, query_points, frequency)
        return torch.abs(field_values)
    
    def render_phase(
        self,
        primitives: List,
        query_points: torch.Tensor,
        frequency: Optional[float] = None
    ) -> torch.Tensor:
        """
        Render field phase at query points.
        
        Args:
            primitives: List of ComplexGaussianPrimitive objects
            query_points: Points to evaluate field at (N, 3)
            frequency: Optional frequency for phase calculations
            
        Returns:
            Field phase values at query points (N,)
        """
        field_values = self.render_field(primitives, query_points, frequency)
        return torch.angle(field_values)
    
    def render_power(
        self,
        primitives: List,
        query_points: torch.Tensor,
        frequency: Optional[float] = None
    ) -> torch.Tensor:
        """
        Render field power (|E|²) at query points.
        
        Args:
            primitives: List of ComplexGaussianPrimitive objects
            query_points: Points to evaluate field at (N, 3)
            frequency: Optional frequency for phase calculations
            
        Returns:
            Field power values at query points (N,)
        """
        field_values = self.render_field(primitives, query_points, frequency)
        return torch.abs(field_values) ** 2
    
    def render_grid(
        self,
        primitives: List,
        bounds: Tuple[torch.Tensor, torch.Tensor],
        resolution: Union[int, Tuple[int, int, int]],
        frequency: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Render field on a regular 3D grid.
        
        Args:
            primitives: List of ComplexGaussianPrimitive objects
            bounds: Tuple of (min_bounds, max_bounds) each of shape (3,)
            resolution: Grid resolution (single int or tuple of 3 ints)
            frequency: Optional frequency for phase calculations
            
        Returns:
            Tuple of (grid_points, field_values)
        """
        min_bounds, max_bounds = bounds
        
        if isinstance(resolution, int):
            resolution = (resolution, resolution, resolution)
        
        # Create grid points
        x = torch.linspace(min_bounds[0], max_bounds[0], resolution[0], device=self.device)
        y = torch.linspace(min_bounds[1], max_bounds[1], resolution[1], device=self.device)
        z = torch.linspace(min_bounds[2], max_bounds[2], resolution[2], device=self.device)
        
        X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
        grid_points = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=1)
        
        # Render field at grid points
        field_values = self.render_field(primitives, grid_points, frequency)
        
        # Reshape to grid
        field_grid = field_values.reshape(resolution)
        
        return grid_points.reshape(resolution + (3,)), field_grid
    
    def render_slice(
        self,
        primitives: List,
        plane_normal: torch.Tensor,
        plane_point: torch.Tensor,
        bounds_2d: Tuple[torch.Tensor, torch.Tensor],
        resolution: Union[int, Tuple[int, int]],
        frequency: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Render field on a 2D slice through 3D space.
        
        Args:
            primitives: List of ComplexGaussianPrimitive objects
            plane_normal: Normal vector of the slice plane (3,)
            plane_point: Point on the slice plane (3,)
            bounds_2d: 2D bounds for the slice
            resolution: 2D resolution (single int or tuple of 2 ints)
            frequency: Optional frequency for phase calculations
            
        Returns:
            Tuple of (slice_points, field_values)
        """
        if isinstance(resolution, int):
            resolution = (resolution, resolution)
        
        # Create orthonormal basis for the plane
        normal = plane_normal / torch.norm(plane_normal)
        
        # Find two orthogonal vectors in the plane
        if torch.abs(normal[0]) < 0.9:
            u = torch.cross(normal, torch.tensor([1.0, 0.0, 0.0], device=self.device))
        else:
            u = torch.cross(normal, torch.tensor([0.0, 1.0, 0.0], device=self.device))
        u = u / torch.norm(u)
        
        v = torch.cross(normal, u)
        v = v / torch.norm(v)
        
        # Create 2D grid in plane coordinates
        min_bounds_2d, max_bounds_2d = bounds_2d
        u_coords = torch.linspace(min_bounds_2d[0], max_bounds_2d[0], resolution[0], device=self.device)
        v_coords = torch.linspace(min_bounds_2d[1], max_bounds_2d[1], resolution[1], device=self.device)
        
        U, V = torch.meshgrid(u_coords, v_coords, indexing='ij')
        
        # Convert to 3D coordinates
        slice_points = (plane_point.unsqueeze(0).unsqueeze(0) + 
                       U.unsqueeze(-1) * u.unsqueeze(0).unsqueeze(0) + 
                       V.unsqueeze(-1) * v.unsqueeze(0).unsqueeze(0))
        
        slice_points_flat = slice_points.reshape(-1, 3)
        
        # Render field at slice points
        field_values = self.render_field(primitives, slice_points_flat, frequency)
        field_slice = field_values.reshape(resolution)
        
        return slice_points, field_slice
    
    def compute_field_gradient(
        self,
        primitives: List,
        query_points: torch.Tensor,
        frequency: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute field values and spatial gradients.
        
        Args:
            primitives: List of ComplexGaussianPrimitive objects
            query_points: Points to evaluate field at (N, 3)
            frequency: Optional frequency for phase calculations
            
        Returns:
            Tuple of (field_values, gradients) where gradients is (N, 3)
        """
        query_points.requires_grad_(True)
        
        field_values = self.render_field(primitives, query_points, frequency)
        
        # Compute gradient for each component
        gradients = torch.zeros(query_points.shape, dtype=torch.float32, device=self.device)
        
        for i in range(query_points.shape[0]):
            if field_values[i].requires_grad:
                grad = torch.autograd.grad(
                    outputs=field_values[i],
                    inputs=query_points,
                    retain_graph=True,
                    create_graph=True
                )[0]
                if grad is not None:
                    gradients[i] = grad[i].real  # Take real part of gradient
        
        return field_values, gradients
    
    def forward(
        self,
        primitives: List,
        query_points: torch.Tensor,
        frequency: Optional[float] = None
    ) -> torch.Tensor:
        """Forward pass for neural network compatibility."""
        return self.render_field(primitives, query_points, frequency)
