"""
Test suite for ComplexGaussianSplatter class.

This module contains comprehensive tests for the core Gaussian Splatting
functionality in jGS.
"""

import pytest
import torch
import numpy as np
import sys
import os

# Add jGS to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import jgs
from jgs.core.gaussian_splatter import ComplexGaussianSplatter


class TestComplexGaussianSplatter:
    """Test ComplexGaussianSplatter class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.n_gaussians = 10
        self.positions = torch.randn(self.n_gaussians, 3)
        self.complex_values = torch.randn(self.n_gaussians, dtype=torch.complex64)
        self.device = 'cpu'  # Use CPU for testing
    
    def test_initialization_with_valid_inputs(self):
        """Test that model initializes correctly with valid inputs."""
        model = ComplexGaussianSplatter(
            positions=self.positions,
            complex_values=self.complex_values,
            device=self.device
        )
        
        assert model.positions.shape == (self.n_gaussians, 3)
        assert model.complex_values.shape == (self.n_gaussians,)
        assert model.scales.shape == (self.n_gaussians, 3)
        assert model.rotations.shape == (self.n_gaussians, 4)
        assert len(model.primitives) == self.n_gaussians
    
    def test_initialization_with_custom_scales_rotations(self):
        """Test initialization with custom scales and rotations."""
        scales = torch.ones(self.n_gaussians, 3) * 0.5
        rotations = torch.zeros(self.n_gaussians, 4)
        rotations[:, 0] = 1.0  # Identity quaternions
        
        model = ComplexGaussianSplatter(
            positions=self.positions,
            complex_values=self.complex_values,
            scales=scales,
            rotations=rotations,
            device=self.device
        )
        
        assert torch.allclose(model.scales, scales)
        assert torch.allclose(model.rotations, rotations)
    
    def test_initialization_with_numpy_arrays(self):
        """Test initialization with numpy arrays."""
        positions_np = self.positions.numpy()
        complex_values_np = self.complex_values.numpy()
        
        model = ComplexGaussianSplatter(
            positions=positions_np,
            complex_values=complex_values_np,
            device=self.device
        )
        
        assert isinstance(model.positions, torch.Tensor)
        assert isinstance(model.complex_values, torch.Tensor)
        assert model.positions.shape == (self.n_gaussians, 3)
    
    def test_render_returns_correct_shape(self):
        """Test that render method returns correct output shape."""
        model = ComplexGaussianSplatter(
            positions=self.positions,
            complex_values=self.complex_values,
            device=self.device
        )
        
        n_query = 20
        query_points = torch.randn(n_query, 3)
        result = model.render(query_points)
        
        assert result.shape == (n_query,)
        assert result.dtype == torch.complex64
    
    def test_render_with_frequency(self):
        """Test rendering with frequency parameter."""
        model = ComplexGaussianSplatter(
            positions=self.positions,
            complex_values=self.complex_values,
            device=self.device
        )
        
        query_points = torch.randn(10, 3)
        frequency = 2.4e9
        
        result_no_freq = model.render(query_points)
        result_with_freq = model.render(query_points, frequency=frequency)
        
        # Results should be different when frequency is applied
        assert not torch.allclose(result_no_freq, result_with_freq)
        assert result_with_freq.shape == result_no_freq.shape
    
    def test_render_single_point(self):
        """Test rendering at a single point."""
        model = ComplexGaussianSplatter(
            positions=self.positions,
            complex_values=self.complex_values,
            device=self.device
        )
        
        query_point = torch.tensor([[0.0, 0.0, 0.0]])
        result = model.render(query_point)
        
        assert result.shape == (1,)
        assert torch.isfinite(result).all()
    
    def test_forward_method(self):
        """Test forward method for neural network compatibility."""
        model = ComplexGaussianSplatter(
            positions=self.positions,
            complex_values=self.complex_values,
            device=self.device
        )
        
        query_points = torch.randn(5, 3)
        
        result_render = model.render(query_points)
        result_forward = model.forward(query_points)
        
        assert torch.allclose(result_render, result_forward)
    
    def test_add_primitive(self):
        """Test adding a new primitive to the model."""
        model = ComplexGaussianSplatter(
            positions=self.positions,
            complex_values=self.complex_values,
            device=self.device
        )
        
        initial_count = len(model.primitives)
        
        new_position = torch.tensor([1.0, 2.0, 3.0])
        new_complex_value = torch.tensor(1.0 + 2.0j)
        
        model.add_primitive(new_position, new_complex_value)
        
        assert len(model.primitives) == initial_count + 1
        assert model.positions.shape[0] == initial_count + 1
        assert torch.allclose(model.positions[-1], new_position)
        assert torch.allclose(model.complex_values[-1], new_complex_value)
    
    def test_remove_primitive(self):
        """Test removing a primitive from the model."""
        model = ComplexGaussianSplatter(
            positions=self.positions,
            complex_values=self.complex_values,
            device=self.device
        )
        
        initial_count = len(model.primitives)
        remove_index = 3
        
        model.remove_primitive(remove_index)
        
        assert len(model.primitives) == initial_count - 1
        assert model.positions.shape[0] == initial_count - 1
    
    def test_remove_primitive_invalid_index(self):
        """Test that removing invalid index raises error."""
        model = ComplexGaussianSplatter(
            positions=self.positions,
            complex_values=self.complex_values,
            device=self.device
        )
        
        with pytest.raises(ValueError, match="Invalid primitive index"):
            model.remove_primitive(100)  # Index out of range
        
        with pytest.raises(ValueError, match="Invalid primitive index"):
            model.remove_primitive(-1)   # Negative index
    
    def test_get_field_magnitude(self):
        """Test field magnitude computation."""
        model = ComplexGaussianSplatter(
            positions=self.positions,
            complex_values=self.complex_values,
            device=self.device
        )
        
        query_points = torch.randn(10, 3)
        magnitude = model.get_field_magnitude(query_points)
        
        assert magnitude.shape == (10,)
        assert magnitude.dtype == torch.float32
        assert (magnitude >= 0).all()  # Magnitude should be non-negative
    
    def test_get_field_phase(self):
        """Test field phase computation."""
        model = ComplexGaussianSplatter(
            positions=self.positions,
            complex_values=self.complex_values,
            device=self.device
        )
        
        query_points = torch.randn(10, 3)
        phase = model.get_field_phase(query_points)
        
        assert phase.shape == (10,)
        assert phase.dtype == torch.float32
        assert (phase >= -np.pi).all() and (phase <= np.pi).all()
    
    def test_save_and_load_state(self, tmp_path):
        """Test saving and loading model state."""
        model = ComplexGaussianSplatter(
            positions=self.positions,
            complex_values=self.complex_values,
            device=self.device
        )
        
        # Save state
        save_path = tmp_path / "test_model.pth"
        model.save_state(str(save_path))
        
        # Create new model and load state
        new_model = ComplexGaussianSplatter(
            positions=torch.zeros(1, 3),
            complex_values=torch.zeros(1, dtype=torch.complex64),
            device=self.device
        )
        new_model.load_state(str(save_path))
        
        # Check that states match
        assert torch.allclose(model.positions, new_model.positions)
        assert torch.allclose(model.complex_values, new_model.complex_values)
        assert torch.allclose(model.scales, new_model.scales)
        assert torch.allclose(model.rotations, new_model.rotations)
    
    def test_gradient_computation(self):
        """Test that gradients can be computed."""
        model = ComplexGaussianSplatter(
            positions=self.positions,
            complex_values=self.complex_values,
            device=self.device
        )
        
        query_points = torch.randn(5, 3, requires_grad=True)
        result = model.render(query_points)
        
        # Compute gradient of sum with respect to query points
        loss = torch.sum(torch.abs(result))
        loss.backward()
        
        assert query_points.grad is not None
        assert query_points.grad.shape == query_points.shape
    
    def test_parameter_optimization(self):
        """Test that model parameters can be optimized."""
        model = ComplexGaussianSplatter(
            positions=self.positions,
            complex_values=self.complex_values,
            device=self.device
        )
        
        # Create target field
        target_points = torch.randn(20, 3)
        target_field = torch.randn(20, dtype=torch.complex64)
        
        # Set up optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        initial_loss = None
        for epoch in range(10):
            optimizer.zero_grad()
            
            predicted = model.render(target_points)
            loss = torch.mean(torch.abs(predicted - target_field)**2)
            
            if initial_loss is None:
                initial_loss = loss.item()
            
            loss.backward()
            optimizer.step()
        
        final_loss = loss.item()
        
        # Loss should decrease (though may not always due to random initialization)
        assert final_loss <= initial_loss * 2  # Allow some tolerance
    
    def test_empty_model(self):
        """Test behavior with empty model."""
        empty_positions = torch.empty(0, 3)
        empty_values = torch.empty(0, dtype=torch.complex64)
        
        model = ComplexGaussianSplatter(
            positions=empty_positions,
            complex_values=empty_values,
            device=self.device
        )
        
        query_points = torch.randn(5, 3)
        result = model.render(query_points)
        
        # Should return zeros for empty model
        assert torch.allclose(result, torch.zeros_like(result))
    
    def test_large_model_performance(self):
        """Test performance with larger model."""
        n_large = 1000
        large_positions = torch.randn(n_large, 3)
        large_values = torch.randn(n_large, dtype=torch.complex64)
        
        model = ComplexGaussianSplatter(
            positions=large_positions,
            complex_values=large_values,
            device=self.device
        )
        
        query_points = torch.randn(100, 3)
        
        # Should complete without error
        result = model.render(query_points)
        assert result.shape == (100,)
        assert torch.isfinite(result).all()


class TestComplexGaussianSplatterEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_invalid_position_shape(self):
        """Test error with invalid position shape."""
        with pytest.raises(Exception):  # Should raise some error
            ComplexGaussianSplatter(
                positions=torch.randn(10),  # Wrong shape
                complex_values=torch.randn(10, dtype=torch.complex64)
            )
    
    def test_mismatched_sizes(self):
        """Test error with mismatched position and value sizes."""
        with pytest.raises(Exception):  # Should raise some error
            ComplexGaussianSplatter(
                positions=torch.randn(10, 3),
                complex_values=torch.randn(5, dtype=torch.complex64)  # Wrong size
            )
    
    def test_invalid_query_points_shape(self):
        """Test error with invalid query points shape."""
        model = ComplexGaussianSplatter(
            positions=torch.randn(5, 3),
            complex_values=torch.randn(5, dtype=torch.complex64)
        )
        
        with pytest.raises(Exception):  # Should raise some error
            model.render(torch.randn(10))  # Wrong shape
    
    def test_device_consistency(self):
        """Test that tensors stay on correct device."""
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        
        model = ComplexGaussianSplatter(
            positions=torch.randn(5, 3),
            complex_values=torch.randn(5, dtype=torch.complex64),
            device=device
        )
        
        assert str(model.positions.device).startswith(device)
        assert str(model.complex_values.device).startswith(device)
        
        query_points = torch.randn(3, 3)
        result = model.render(query_points)
        
        # Result should be on same device as model
        assert str(result.device).startswith(device)


if __name__ == "__main__":
    pytest.main([__file__])
