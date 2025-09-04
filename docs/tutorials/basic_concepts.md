# Basic Concepts of Complex-valued Gaussian Splatting for RF

This tutorial introduces the fundamental concepts behind jGS and how Gaussian Splatting is adapted for complex-valued RF signal processing.

## Table of Contents
1. [From Computer Graphics to RF](#from-computer-graphics-to-rf)
2. [Complex-valued Fields](#complex-valued-fields)
3. [Gaussian Primitives for RF](#gaussian-primitives-for-rf)
4. [Mathematical Foundation](#mathematical-foundation)
5. [Rendering Process](#rendering-process)
6. [Optimization and Learning](#optimization-and-learning)

## From Computer Graphics to RF

### Traditional Gaussian Splatting
Gaussian Splatting was originally developed for 3D scene representation in computer graphics:
- **Goal**: Represent 3D scenes using Gaussian primitives
- **Data**: RGB colors and opacity
- **Rendering**: Project 3D Gaussians to 2D images

### jGS Adaptation for RF
We adapt this technique for electromagnetic field representation:
- **Goal**: Represent RF fields using complex-valued Gaussians
- **Data**: Complex amplitudes (magnitude + phase)
- **Rendering**: Evaluate field at arbitrary 3D points

```python
# Traditional Gaussian Splatting (graphics)
gaussian = {
    'position': [x, y, z],
    'color': [r, g, b],
    'opacity': alpha,
    'scale': [sx, sy, sz],
    'rotation': quaternion
}

# jGS Complex Gaussian (RF)
gaussian = {
    'position': [x, y, z],
    'complex_value': magnitude * exp(1j * phase),
    'scale': [sx, sy, sz],
    'rotation': quaternion
}
```

## Complex-valued Fields

### Why Complex Numbers?
RF electromagnetic fields are naturally complex due to their wave nature:

1. **Magnitude**: Field strength or amplitude
2. **Phase**: Temporal/spatial phase relationship
3. **Frequency**: Determines wavelength and propagation

```python
import numpy as np
import matplotlib.pyplot as plt

# Example: Complex sinusoidal field
t = np.linspace(0, 2*np.pi, 100)
frequency = 2.4e9  # 2.4 GHz
field = np.exp(1j * 2 * np.pi * frequency * t)

magnitude = np.abs(field)  # Always 1 for this example
phase = np.angle(field)    # Linear phase ramp

# Visualization
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
ax1.plot(t, magnitude, label='Magnitude')
ax1.set_ylabel('|E|')
ax1.legend()

ax2.plot(t, phase, label='Phase', color='red')
ax2.set_ylabel('∠E (rad)')
ax2.set_xlabel('Time')
ax2.legend()
plt.tight_layout()
```

### Field Properties
Complex fields encode multiple physical quantities:

```python
# Given a complex field E
E = 2.5 * np.exp(1j * np.pi/4)  # Magnitude=2.5, Phase=45°

# Extract properties
magnitude = np.abs(E)           # Field strength
phase = np.angle(E)             # Phase in radians
power = np.abs(E)**2            # Power density
real_part = np.real(E)          # In-phase component
imag_part = np.imag(E)          # Quadrature component

print(f"Magnitude: {magnitude:.2f}")
print(f"Phase: {np.degrees(phase):.1f}°")
print(f"Power: {power:.2f}")
```

## Gaussian Primitives for RF

### 3D Gaussian Distribution
Each primitive represents a localized field distribution:

```python
import torch
from jgs.core.primitives import ComplexGaussianPrimitive

# Create a Gaussian primitive
position = torch.tensor([1.0, 0.5, 0.0])      # Center position
complex_value = torch.tensor(2.0 + 1j * 1.5)  # Complex amplitude
scale = torch.tensor([0.3, 0.3, 0.2])         # Size in each dimension
rotation = torch.tensor([1.0, 0.0, 0.0, 0.0]) # Identity quaternion

primitive = ComplexGaussianPrimitive(
    position=position,
    complex_value=complex_value,
    scale=scale,
    rotation=rotation
)

# Evaluate at query points
query_points = torch.tensor([[1.0, 0.5, 0.0],  # At center
                            [1.5, 0.5, 0.0]])   # Offset
field_values = primitive.evaluate(query_points)
print(f"Field at center: {field_values[0]}")
print(f"Field at offset: {field_values[1]}")
```

### Primitive Parameters

1. **Position** (μ): 3D center coordinates
2. **Complex Value** (A): Complex amplitude A = |A|e^(jφ)
3. **Scale** (σ): Standard deviations [σₓ, σᵧ, σᵤ]
4. **Rotation** (R): Orientation quaternion

The Gaussian function becomes:
```
G(x) = A * exp(-½(x-μ)ᵀ Σ⁻¹ (x-μ))
```

Where Σ is the covariance matrix derived from scale and rotation.

## Mathematical Foundation

### Field Superposition
The total field is the sum of all Gaussian contributions:

```python
# Mathematical representation
def total_field(query_points, primitives):
    """
    E_total(x) = Σᵢ Aᵢ * exp(-½(x-μᵢ)ᵀ Σᵢ⁻¹ (x-μᵢ))
    """
    total = torch.zeros(len(query_points), dtype=torch.complex64)
    
    for primitive in primitives:
        contribution = primitive.evaluate(query_points)
        total += contribution
    
    return total
```

### Covariance Matrix
The covariance matrix Σ determines the Gaussian shape:

```python
def compute_covariance(scale, rotation_matrix):
    """
    Σ = R * S * Rᵀ
    where S = diag(σₓ², σᵧ², σᵤ²)
    """
    S = torch.diag(scale ** 2)
    covariance = rotation_matrix @ S @ rotation_matrix.T
    return covariance
```

### Frequency-dependent Phase
For propagating waves, add frequency-dependent phase:

```python
def evaluate_with_frequency(self, query_points, frequency=None):
    # Base Gaussian evaluation
    field = self.evaluate_base(query_points)
    
    if frequency is not None:
        # Add propagation phase
        distance = torch.norm(query_points - self.position, dim=1)
        k = 2 * np.pi * frequency / 3e8  # Wave number
        phase_shift = torch.exp(-1j * k * distance)
        field *= phase_shift
    
    return field
```

## Rendering Process

### Forward Rendering
Given Gaussian primitives, compute field at any point:

```python
import jgs

# Create model with multiple primitives
model = jgs.ComplexGaussianSplatter(
    positions=torch.randn(50, 3),           # 50 random positions
    complex_values=torch.randn(50, dtype=torch.complex64),
    device='cuda'
)

# Render field at query points
query_points = torch.tensor([[0, 0, 0], [1, 1, 1]], dtype=torch.float32)
rendered_field = model.render(query_points, frequency=2.4e9)

print(f"Field at origin: {rendered_field[0]}")
print(f"Field at (1,1,1): {rendered_field[1]}")
```

### Batch Processing
For efficiency, process multiple points simultaneously:

```python
# Generate grid of points
x = torch.linspace(-2, 2, 50)
y = torch.linspace(-2, 2, 50)
z = torch.zeros(1)

X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
grid_points = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=1)

# Render entire grid at once
field_grid = model.render(grid_points)
field_2d = field_grid.reshape(50, 50)

# Visualize
import matplotlib.pyplot as plt
plt.imshow(torch.abs(field_2d), extent=[-2, 2, -2, 2])
plt.colorbar(label='|E|')
plt.title('Field Magnitude')
```

## Optimization and Learning

### Fitting to Measurements
The key advantage of jGS is the ability to fit Gaussian parameters to measurement data:

```python
from jgs.core.optimization import ComplexOptimizer

# Measurement data
measurement_points = torch.randn(100, 3)  # 100 measurement locations
measured_field = torch.randn(100, dtype=torch.complex64)  # Measured values

# Initialize model
model = jgs.ComplexGaussianSplatter(
    positions=torch.randn(20, 3),  # 20 Gaussians
    complex_values=torch.randn(20, dtype=torch.complex64)
)

# Set up optimizer
optimizer = ComplexOptimizer(model, learning_rate=1e-3)

# Fit model to measurements
history = optimizer.fit(
    query_points=measurement_points,
    target_values=measured_field,
    num_epochs=1000,
    loss_function='complex_mse'
)

print(f"Final loss: {history['train_loss'][-1]:.6f}")
```

### Loss Functions
Different loss functions for different objectives:

```python
# Complex MSE: |E_pred - E_true|²
loss_mse = torch.mean(torch.abs(predicted - target)**2)

# Magnitude + Phase loss
mag_loss = torch.mean((torch.abs(predicted) - torch.abs(target))**2)
phase_diff = torch.angle(predicted) - torch.angle(target)
phase_loss = torch.mean(torch.angle(torch.exp(1j * phase_diff))**2)
total_loss = mag_loss + phase_loss
```

## Key Advantages

1. **Continuous Representation**: Evaluate field anywhere in 3D space
2. **Differentiable**: Enables gradient-based optimization
3. **Compact**: Few parameters represent complex field distributions
4. **Flexible**: Handles arbitrary measurement geometries
5. **Physically Meaningful**: Preserves complex field properties

## Limitations and Considerations

1. **Smoothness Assumption**: Fields must be reasonably smooth
2. **Gaussian Basis**: May not capture sharp discontinuities well
3. **Parameter Count**: Need sufficient Gaussians for complex fields
4. **Local Minima**: Optimization may get stuck in poor solutions

## Next Steps

Now that you understand the basic concepts:

1. **Try Field Reconstruction**: [Field Reconstruction Tutorial](field_reconstruction.md)
2. **Learn Antenna Modeling**: [Antenna Modeling Tutorial](antenna_modeling.md)
3. **Explore Optimization**: [Advanced Topics](advanced_topics.md)

---

**Previous**: [Getting Started](getting_started.md) | **Next**: [Antenna Modeling](antenna_modeling.md)
