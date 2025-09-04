#!/usr/bin/env python3
"""
Antenna Array Example for jGS.

This example demonstrates how to model and analyze antenna arrays
using complex-valued Gaussian Splatting techniques.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
import os

# Add jGS to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jgs
from jgs.rf.antenna_patterns import DipoleAntenna, ArrayAntenna
from jgs.core.optimization import ComplexOptimizer
from jgs.visualization.plotter import RFPlotter


def create_linear_array(n_elements=8, spacing=0.5, frequency=2.4e9):
    """Create a linear antenna array."""
    print(f"Creating linear array with {n_elements} elements...")
    
    # Create individual dipole elements
    elements = []
    element_positions = []
    
    for i in range(n_elements):
        # Position elements along x-axis
        position = np.array([i * spacing, 0.0, 0.0])
        element_positions.append(position)
        
        # Create dipole antenna
        dipole = DipoleAntenna(
            frequency=frequency,
            position=position,
            orientation=np.array([0, 0, 0])  # Vertical orientation
        )
        elements.append(dipole)
    
    # Create array with uniform weighting
    array_antenna = ArrayAntenna(
        element_antennas=elements,
        device='cpu'
    )
    
    print(f"Array created with {spacing:.2f}λ spacing")
    return array_antenna, element_positions


def simulate_array_pattern(array_antenna, resolution=100):
    """Simulate the array radiation pattern."""
    print("Simulating array radiation pattern...")
    
    # Create observation points on a sphere
    theta = np.linspace(0, np.pi, resolution)
    phi = np.linspace(0, 2*np.pi, resolution)
    
    THETA, PHI = np.meshgrid(theta, phi)
    
    # Convert to Cartesian coordinates (unit sphere)
    r = 10.0  # Far field distance
    x = r * np.sin(THETA) * np.cos(PHI)
    y = r * np.sin(THETA) * np.sin(PHI)
    z = r * np.cos(THETA)
    
    # Stack coordinates
    positions = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)
    positions_tensor = torch.tensor(positions, dtype=torch.float32)
    
    # Compute array pattern
    pattern = array_antenna.compute_pattern(positions_tensor)
    pattern_2d = pattern.reshape(resolution, resolution)
    
    return THETA, PHI, pattern_2d


def apply_beam_steering(array_antenna, steering_angles):
    """Demonstrate beam steering capabilities."""
    print("Demonstrating beam steering...")
    
    patterns = {}
    
    for name, (theta_s, phi_s) in steering_angles.items():
        print(f"  Steering to {name}: θ={np.degrees(theta_s):.1f}°, φ={np.degrees(phi_s):.1f}°")
        
        # Set beam steering
        array_antenna.set_beam_steering((theta_s, phi_s))
        
        # Compute pattern
        theta, phi, pattern = simulate_array_pattern(array_antenna, resolution=50)
        patterns[name] = (theta, phi, pattern)
    
    return patterns


def fit_gaussian_model_to_array(array_antenna, n_gaussians=30):
    """Fit Gaussian Splatting model to array pattern."""
    print(f"Fitting Gaussian model with {n_gaussians} primitives...")
    
    # Generate measurement points
    n_measurements = 500
    positions = np.random.uniform(-5, 5, (n_measurements, 3))
    positions[:, 2] = np.abs(positions[:, 2])  # Keep z positive (upper hemisphere)
    positions_tensor = torch.tensor(positions, dtype=torch.float32)
    
    # Compute true field from array
    true_field = array_antenna.compute_pattern(positions_tensor)
    
    # Add noise
    noise_level = 0.05
    noise = noise_level * (torch.randn_like(true_field) + 1j * torch.randn_like(true_field))
    noisy_field = true_field + noise
    
    # Initialize Gaussian model
    gaussian_positions = torch.randn(n_gaussians, 3) * 2.0
    initial_values = torch.randn(n_gaussians, dtype=torch.complex64) * 0.1
    
    model = jgs.ComplexGaussianSplatter(
        positions=gaussian_positions,
        complex_values=initial_values,
        device='cpu'
    )
    
    # Optimize
    optimizer = ComplexOptimizer(model, learning_rate=5e-3)
    history = optimizer.fit(
        query_points=positions_tensor,
        target_values=noisy_field,
        num_epochs=300,
        verbose=True,
        loss_function='complex_mse'
    )
    
    return model, history, positions_tensor, true_field


def visualize_results(array_antenna, model, patterns, history):
    """Create comprehensive visualizations."""
    print("Creating visualizations...")
    
    plotter = RFPlotter()
    
    # 1. Array geometry
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot element positions
    for i, antenna in enumerate(array_antenna.element_antennas):
        pos = antenna.position.cpu().numpy()
        ax1.scatter(pos[0], pos[1], s=100, c='red', marker='o', label='Element' if i == 0 else "")
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('Linear Antenna Array Geometry')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_aspect('equal')
    plt.tight_layout()
    plt.show()
    
    # 2. Beam steering patterns
    fig2, axes = plt.subplots(2, 2, figsize=(12, 10), subplot_kw=dict(projection='polar'))
    axes = axes.flatten()
    
    for i, (name, (theta, phi, pattern)) in enumerate(patterns.items()):
        if i < 4:  # Only plot first 4 patterns
            # Take a slice at phi=0 for polar plot
            pattern_slice = np.abs(pattern[:, 0])  # phi=0 slice
            theta_slice = theta[:, 0]
            
            axes[i].plot(theta_slice, pattern_slice / np.max(pattern_slice))
            axes[i].set_title(f'Pattern: {name}')
            axes[i].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.show()
    
    # 3. Training history
    fig3 = plotter.plot_training_history(history, title="Gaussian Model Training")
    plt.show()
    
    # 4. Model comparison
    test_points = torch.randn(100, 3) * 3.0
    test_points[:, 2] = torch.abs(test_points[:, 2])  # Upper hemisphere
    
    array_field = array_antenna.compute_pattern(test_points)
    model_field = model.render(test_points)
    
    fig4 = plotter.plot_field_comparison(
        positions=test_points,
        field1=array_field,
        field2=model_field,
        labels=("Array Pattern", "Gaussian Model"),
        comparison_type='magnitude'
    )
    plt.show()


def analyze_array_performance(array_antenna, model):
    """Analyze array and model performance."""
    print("\n=== Array Performance Analysis ===")
    
    # Compute directivity
    directivity = array_antenna.element_antennas[0].compute_directivity()
    print(f"Single element directivity: {directivity:.2f} ({10*np.log10(directivity):.1f} dB)")
    
    # Array factor analysis
    test_points = torch.tensor([[10.0, 0.0, 0.0]], dtype=torch.float32)  # Broadside
    array_factor = array_antenna.compute_array_factor(test_points)
    print(f"Array factor at broadside: {torch.abs(array_factor[0]):.2f}")
    
    # Model accuracy
    eval_points = torch.randn(200, 3) * 4.0
    eval_points[:, 2] = torch.abs(eval_points[:, 2])
    
    true_field = array_antenna.compute_pattern(eval_points)
    pred_field = model.render(eval_points)
    
    mse = torch.mean(torch.abs(pred_field - true_field)**2).item()
    correlation = torch.corrcoef(torch.stack([
        torch.real(pred_field.flatten()), 
        torch.real(true_field.flatten())
    ]))[0, 1].item()
    
    print(f"Model MSE: {mse:.6f}")
    print(f"Model correlation: {correlation:.4f}")


def main():
    """Main execution function."""
    print("=== jGS Antenna Array Example ===")
    print("Modeling antenna arrays with complex-valued Gaussian Splatting\n")
    
    # Parameters
    frequency = 2.4e9  # 2.4 GHz
    n_elements = 6
    spacing = 0.5  # wavelength units
    
    try:
        # Step 1: Create antenna array
        array_antenna, element_positions = create_linear_array(
            n_elements=n_elements, 
            spacing=spacing, 
            frequency=frequency
        )
        
        # Step 2: Demonstrate beam steering
        steering_angles = {
            'Broadside': (np.pi/2, 0),           # θ=90°, φ=0°
            'Endfire': (np.pi/2, np.pi/2),      # θ=90°, φ=90°
            '30° Elevation': (np.pi/3, 0),       # θ=60°, φ=0°
            '45° Azimuth': (np.pi/2, np.pi/4)   # θ=90°, φ=45°
        }
        
        patterns = apply_beam_steering(array_antenna, steering_angles)
        
        # Step 3: Fit Gaussian model
        model, history, positions, true_field = fit_gaussian_model_to_array(array_antenna)
        
        # Step 4: Analyze performance
        analyze_array_performance(array_antenna, model)
        
        # Step 5: Visualize results
        visualize_results(array_antenna, model, patterns, history)
        
        print("\n=== Example completed successfully! ===")
        
        # Save results
        model.save_state("antenna_array_model.pth")
        print("Model saved to 'antenna_array_model.pth'")
        
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
