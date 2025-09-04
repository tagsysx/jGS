#!/usr/bin/env python3
"""
Basic example of complex-valued Gaussian Splatting for RF signals.

This script demonstrates the fundamental usage of jGS for representing
and reconstructing complex-valued RF field data.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
import os

# Add jGS to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jgs
from jgs.rf.antenna_patterns import DipoleAntenna
from jgs.core.optimization import ComplexOptimizer
from jgs.visualization.plotter import RFPlotter


def generate_synthetic_data(n_points: int = 1000, frequency: float = 2.4e9):
    """Generate synthetic RF field data from a dipole antenna."""
    print("Generating synthetic RF field data...")
    
    # Create dipole antenna
    antenna_pos = np.array([0.0, 0.0, 0.0])
    dipole = DipoleAntenna(frequency=frequency, position=antenna_pos)
    
    # Generate random measurement points
    np.random.seed(42)  # For reproducibility
    positions = np.random.uniform(-2, 2, (n_points, 3))
    positions = torch.tensor(positions, dtype=torch.float32)
    
    # Compute true field from dipole
    true_field = dipole.compute_pattern(positions)
    
    # Add some noise
    noise_level = 0.1
    noise = noise_level * (torch.randn_like(true_field) + 1j * torch.randn_like(true_field))
    noisy_field = true_field + noise
    
    print(f"Generated {n_points} field measurements at {frequency/1e9:.1f} GHz")
    
    return positions, noisy_field, true_field


def fit_gaussian_splatting(positions, field_data, n_gaussians: int = 50):
    """Fit Gaussian Splatting model to field data."""
    print(f"Fitting Gaussian Splatting model with {n_gaussians} primitives...")
    
    # Initialize Gaussian positions randomly within the measurement region
    gaussian_positions = torch.randn(n_gaussians, 3) * 1.5
    
    # Initialize complex values
    initial_values = torch.randn(n_gaussians, dtype=torch.complex64) * 0.1
    
    # Create Gaussian Splatting model
    model = jgs.ComplexGaussianSplatter(
        positions=gaussian_positions,
        complex_values=initial_values,
        device='cpu'  # Use CPU for this example
    )
    
    # Create optimizer
    optimizer = ComplexOptimizer(
        model=model,
        learning_rate=1e-3,
        optimizer_type='adam'
    )
    
    # Fit the model
    history = optimizer.fit(
        query_points=positions,
        target_values=field_data,
        num_epochs=500,
        verbose=True,
        loss_function='complex_mse'
    )
    
    print("Training completed!")
    return model, history


def evaluate_reconstruction(model, positions, true_field, test_positions=None):
    """Evaluate the reconstruction quality."""
    print("Evaluating reconstruction quality...")
    
    # Render field at original positions
    predicted_field = model.render(positions)
    
    # Compute metrics
    mse = torch.mean(torch.abs(predicted_field - true_field) ** 2).item()
    mae = torch.mean(torch.abs(predicted_field - true_field)).item()
    
    # Correlation coefficient
    pred_flat = predicted_field.flatten()
    true_flat = true_field.flatten()
    correlation = torch.corrcoef(torch.stack([
        torch.real(pred_flat), torch.real(true_flat)
    ]))[0, 1].item()
    
    print(f"Reconstruction Metrics:")
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  Correlation: {correlation:.4f}")
    
    return {
        'mse': mse,
        'mae': mae,
        'correlation': correlation,
        'predicted_field': predicted_field
    }


def visualize_results(positions, true_field, predicted_field, history):
    """Create visualization plots."""
    print("Creating visualization plots...")
    
    plotter = RFPlotter()
    
    # Plot field comparison
    fig1 = plotter.plot_field_comparison(
        positions=positions,
        field1=true_field,
        field2=predicted_field,
        labels=("True Field", "Reconstructed Field"),
        comparison_type='magnitude'
    )
    plt.show()
    
    # Plot training history
    fig2 = plotter.plot_training_history(history)
    plt.show()
    
    # Plot field magnitude
    fig3 = plotter.plot_field_magnitude(
        positions=positions,
        complex_field=predicted_field,
        title="Reconstructed Field Magnitude"
    )
    plt.show()
    
    print("Visualization complete!")


def main():
    """Main execution function."""
    print("=== jGS Basic Example ===")
    print("Complex-valued Gaussian Splatting for RF Signal Processing\n")
    
    # Parameters
    frequency = 2.4e9  # 2.4 GHz
    n_points = 500     # Number of measurement points
    n_gaussians = 30   # Number of Gaussian primitives
    
    try:
        # Step 1: Generate synthetic data
        positions, noisy_field, true_field = generate_synthetic_data(n_points, frequency)
        
        # Step 2: Fit Gaussian Splatting model
        model, history = fit_gaussian_splatting(positions, noisy_field, n_gaussians)
        
        # Step 3: Evaluate reconstruction
        results = evaluate_reconstruction(model, positions, true_field)
        
        # Step 4: Visualize results
        visualize_results(positions, true_field, results['predicted_field'], history)
        
        print("\n=== Example completed successfully! ===")
        
        # Save model for later use
        model.save_state("basic_example_model.pth")
        print("Model saved to 'basic_example_model.pth'")
        
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
