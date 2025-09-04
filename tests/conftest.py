"""
Pytest configuration and shared fixtures for jGS tests.

This module provides common test fixtures and configuration
for the jGS test suite.
"""

import pytest
import torch
import numpy as np
import tempfile
import os


@pytest.fixture
def device():
    """Fixture providing the device to use for tests."""
    return 'cpu'  # Use CPU for tests to ensure compatibility


@pytest.fixture
def sample_positions():
    """Fixture providing sample 3D positions."""
    torch.manual_seed(42)  # For reproducibility
    return torch.randn(20, 3)


@pytest.fixture
def sample_complex_values():
    """Fixture providing sample complex values."""
    torch.manual_seed(42)  # For reproducibility
    return torch.randn(20, dtype=torch.complex64)


@pytest.fixture
def sample_query_points():
    """Fixture providing sample query points."""
    torch.manual_seed(123)  # Different seed for variety
    return torch.randn(10, 3)


@pytest.fixture
def small_model_data():
    """Fixture providing data for a small test model."""
    torch.manual_seed(42)
    return {
        'positions': torch.randn(5, 3),
        'complex_values': torch.randn(5, dtype=torch.complex64),
        'scales': torch.ones(5, 3) * 0.5,
        'rotations': torch.tensor([
            [1.0, 0.0, 0.0, 0.0],  # Identity quaternions
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0]
        ])
    }


@pytest.fixture
def temp_directory():
    """Fixture providing a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def sample_rf_data():
    """Fixture providing sample RF measurement data."""
    np.random.seed(42)
    
    # Generate measurement positions
    n_measurements = 100
    positions = np.random.uniform(-2, 2, (n_measurements, 3))
    
    # Generate synthetic complex field (simple dipole-like pattern)
    r = np.linalg.norm(positions, axis=1)
    theta = np.arccos(positions[:, 2] / (r + 1e-8))  # Angle from z-axis
    
    # Simple dipole pattern: sin(theta) / r
    amplitude = np.sin(theta) / (r + 0.1)
    phase = -2 * np.pi * r / 0.125  # Assume wavelength = 0.125m
    
    complex_field = amplitude * np.exp(1j * phase)
    
    return {
        'positions': torch.tensor(positions, dtype=torch.float32),
        'complex_field': torch.tensor(complex_field, dtype=torch.complex64),
        'frequency': 2.4e9
    }


@pytest.fixture
def antenna_test_data():
    """Fixture providing test data for antenna patterns."""
    # Create test points on a sphere
    n_points = 50
    theta = np.linspace(0, np.pi, n_points)
    phi = np.linspace(0, 2*np.pi, n_points)
    
    THETA, PHI = np.meshgrid(theta, phi)
    
    # Convert to Cartesian (unit sphere)
    r = 5.0  # Distance
    x = r * np.sin(THETA) * np.cos(PHI)
    y = r * np.sin(THETA) * np.sin(PHI)
    z = r * np.cos(THETA)
    
    positions = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)
    
    return {
        'positions': torch.tensor(positions, dtype=torch.float32),
        'theta': THETA,
        'phi': PHI,
        'frequency': 2.4e9,
        'antenna_position': np.array([0.0, 0.0, 0.0])
    }


@pytest.fixture(autouse=True)
def set_random_seeds():
    """Automatically set random seeds for reproducible tests."""
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)


@pytest.fixture
def optimization_test_data():
    """Fixture providing data for optimization tests."""
    torch.manual_seed(42)
    
    # Create target field (simple Gaussian)
    n_points = 50
    positions = torch.randn(n_points, 3) * 2.0
    
    # Target: single Gaussian at origin
    distances = torch.norm(positions, dim=1)
    target_field = torch.exp(-distances**2) * torch.exp(1j * distances)
    
    return {
        'positions': positions,
        'target_field': target_field,
        'test_positions': torch.randn(20, 3) * 1.5
    }


class TestConfig:
    """Test configuration constants."""
    
    # Tolerances for numerical comparisons
    FLOAT_TOLERANCE = 1e-6
    COMPLEX_TOLERANCE = 1e-6
    
    # Default test parameters
    DEFAULT_FREQUENCY = 2.4e9
    DEFAULT_WAVELENGTH = 3e8 / DEFAULT_FREQUENCY
    
    # Test data sizes
    SMALL_SIZE = 10
    MEDIUM_SIZE = 100
    LARGE_SIZE = 1000


@pytest.fixture
def test_config():
    """Fixture providing test configuration."""
    return TestConfig()


def pytest_configure(config):
    """Pytest configuration hook."""
    # Add custom markers
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )
    config.addinivalue_line(
        "markers", "integration: marks integration tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Mark GPU tests
        if "cuda" in item.name.lower() or "gpu" in item.name.lower():
            item.add_marker(pytest.mark.gpu)
        
        # Mark slow tests
        if "large" in item.name.lower() or "performance" in item.name.lower():
            item.add_marker(pytest.mark.slow)
        
        # Mark integration tests
        if "integration" in item.name.lower() or item.fspath.basename.startswith("test_integration"):
            item.add_marker(pytest.mark.integration)


@pytest.fixture
def skip_if_no_cuda():
    """Fixture to skip tests if CUDA is not available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


# Helper functions for tests
def assert_complex_allclose(actual, expected, rtol=1e-5, atol=1e-8):
    """Assert that complex tensors are close."""
    assert torch.allclose(actual.real, expected.real, rtol=rtol, atol=atol)
    assert torch.allclose(actual.imag, expected.imag, rtol=rtol, atol=atol)


def assert_tensor_properties(tensor, expected_shape, expected_dtype=None, device=None):
    """Assert tensor has expected properties."""
    assert tensor.shape == expected_shape, f"Expected shape {expected_shape}, got {tensor.shape}"
    
    if expected_dtype is not None:
        assert tensor.dtype == expected_dtype, f"Expected dtype {expected_dtype}, got {tensor.dtype}"
    
    if device is not None:
        assert str(tensor.device).startswith(device), f"Expected device {device}, got {tensor.device}"


def create_test_gaussian_model(n_gaussians=10, device='cpu'):
    """Helper function to create a test Gaussian model."""
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    
    import jgs
    
    torch.manual_seed(42)
    positions = torch.randn(n_gaussians, 3)
    complex_values = torch.randn(n_gaussians, dtype=torch.complex64)
    
    return jgs.ComplexGaussianSplatter(
        positions=positions,
        complex_values=complex_values,
        device=device
    )
