# Basic Concepts of Complex-valued Gaussian Splatting for RF

This tutorial introduces the fundamental concepts behind jGS and how Gaussian Splatting is adapted for complex-valued RF signal processing.

## Table of Contents
1. [From Computer Graphics to RF](#from-computer-graphics-to-rf)
2. [Complex-valued Fields](#complex-valued-fields)
3. [Gaussian Primitives for RF](#gaussian-primitives-for-rf)
4. [Data Storage and Memory Management](#data-storage-and-memory-management)
5. [Mathematical Foundation](#mathematical-foundation)
6. [Rendering Process](#rendering-process)
7. [Optimization and Learning](#optimization-and-learning)
8. [Performance Considerations](#performance-considerations)

## From Computer Graphics to RF

### Traditional Gaussian Splatting
Gaussian Splatting was originally developed for 3D scene representation in computer graphics:
- **Goal**: Represent 3D scenes using Gaussian primitives
- **Data**: RGB colors and opacity
- **Rendering**: Project 3D Gaussians to 2D images

### jGS Adaptation for RF
We adapt this technique for electromagnetic field representation:
- **Goal**: Represent RF fields using complex-valued Gaussians
- **Data**: Complex radiance (magnitude + phase) with complex attenuation
- **Rendering**: Evaluate field at arbitrary 3D points
- **Physics**: Incorporates both radiance and attenuation effects

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
    'radiance': magnitude * exp(1j * phase),
    'scale': [sx, sy, sz],
    'rotation': quaternion,
    'attenuation': complex_attenuation_coefficient
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

### Complex Radiance and Attenuation

In jGS, we distinguish between two key physical quantities:

#### 1. Complex Radiance
The **complex radiance** represents the intrinsic electromagnetic field emission or scattering from a source:

```python
# Complex radiance combines magnitude and phase
radiance = magnitude * np.exp(1j * phase)

# Examples of different radiance types:
# Point source with uniform radiance
uniform_radiance = 1.0 + 0j

# Directional source with phase gradient
directional_radiance = 2.5 * np.exp(1j * np.pi/3)  # 2.5 magnitude, 60° phase

# Frequency-dependent radiance
frequency = 2.4e9  # 2.4 GHz
wavelength = 3e8 / frequency
k = 2 * np.pi / wavelength
radiance_with_propagation = np.exp(1j * k * distance)
```

#### 2. Complex Attenuation Coefficient
The **complex attenuation coefficient** modulates the radiance based on material properties and propagation effects:

```python
# Complex attenuation coefficient
attenuation = attenuation_magnitude * np.exp(1j * attenuation_phase)

# Physical interpretations:
# Real part: Amplitude attenuation (absorption, scattering)
# Imaginary part: Phase shift (dispersion, refraction)

# Examples:
# No attenuation (free space)
free_space = 1.0 + 0j

# Pure amplitude attenuation (lossy medium)
lossy_medium = 0.7 + 0j  # 30% power loss

# Pure phase shift (dispersive medium)
dispersive = 1.0 * np.exp(1j * np.pi/6)  # 30° phase shift, no loss

# Combined attenuation and phase shift
realistic_medium = 0.8 * np.exp(1j * np.pi/4)  # 20% loss + 45° phase shift
```

#### Combined Field Calculation
The total field contribution from a Gaussian primitive is:

```python
def calculate_field_contribution(radiance, attenuation, gaussian_weight):
    """
    Calculate the field contribution from a primitive.
    
    Args:
        radiance: Complex radiance value
        attenuation: Complex attenuation coefficient  
        gaussian_weight: Real-valued Gaussian spatial weight
        
    Returns:
        Complex field contribution
    """
    # Convert Gaussian weight to complex for proper multiplication
    complex_weight = gaussian_weight.to(dtype=torch.complex64)
    
    # Apply attenuation to the Gaussian weight
    attenuated_weight = complex_weight * attenuation
    
    # Combine with radiance
    field_contribution = attenuated_weight * radiance
    
    return field_contribution
```

#### Physical Examples

```python
# Example 1: Antenna with frequency-dependent attenuation
antenna_radiance = 5.0 * np.exp(1j * 0)  # 5V/m, 0° phase
frequency_attenuation = np.exp(-1j * 2 * np.pi * frequency * delay)
field_1 = antenna_radiance * frequency_attenuation

# Example 2: Scatterer in lossy medium
scatterer_radiance = 2.0 * np.exp(1j * np.pi/2)  # 2V/m, 90° phase
medium_loss = 0.6 + 0j  # 40% power loss
medium_dispersion = np.exp(1j * np.pi/8)  # 22.5° phase shift
combined_attenuation = medium_loss * medium_dispersion
field_2 = scatterer_radiance * combined_attenuation

# Example 3: Multi-path propagation
direct_path = 1.0 + 0j  # No attenuation
reflected_path = 0.8 * np.exp(1j * np.pi)  # 20% loss + 180° phase (reflection)
```

## Gaussian Primitives for RF

### 3D Gaussian Distribution
Each primitive represents a localized field distribution:

```python
import torch
from jgs.core.primitives import ComplexGaussianPrimitive

# Create a Gaussian primitive
position = torch.tensor([1.0, 0.5, 0.0])      # Center position
complex_radiance = torch.tensor(2.0 + 1j * 1.5)  # Complex radiance
scale = torch.tensor([0.3, 0.3, 0.2])         # Size in each dimension
rotation = torch.tensor([1.0, 0.0, 0.0, 0.0]) # Identity quaternion
attenuation = torch.tensor(0.9 + 0.1j)        # Complex attenuation coefficient

primitive = ComplexGaussianPrimitive(
    position=position,
    complex_value=complex_radiance,  # Represents radiance
    scale=scale,
    rotation=rotation,
    attenuation=attenuation
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
2. **Complex Radiance** (R): Complex radiance R = |R|e^(jφᵣ)
3. **Scale** (σ): Standard deviations [σₓ, σᵧ, σᵤ]
4. **Rotation** (Q): Orientation quaternion
5. **Attenuation** (α): Complex attenuation coefficient α = |α|e^(jφₐ)

The Gaussian function becomes:
```
G(x) = α * R * exp(-½(x-μ)ᵀ Σ⁻¹ (x-μ))
```

Where:
- R is the complex radiance (intrinsic field emission)
- α is the complex attenuation coefficient (medium/propagation effects)
- The product α * R gives the effective complex amplitude

Where Σ is the covariance matrix derived from scale and rotation.

## Data Storage and Memory Management

### How Capacity is Stored

The jGS system stores Gaussian primitive data in a highly optimized manner for both memory efficiency and computational performance:

#### Parameter Storage Structure

Each Gaussian primitive requires the following core parameters:

```python
# Per-primitive storage requirements:
position_data = torch.float32    # 3 values × 4 bytes = 12 bytes
complex_radiance = torch.complex64  # 1 complex × 8 bytes = 8 bytes
scale_data = torch.float32       # 3 values × 4 bytes = 12 bytes
rotation_data = torch.float32    # 4 quaternion × 4 bytes = 16 bytes
attenuation_data = torch.complex64  # 1 complex × 8 bytes = 8 bytes
# Total per primitive: 56 bytes

# For N primitives: 56N bytes of core parameters
```

#### Tensor-based Storage

The `ComplexGaussianSplatter` class stores all primitives in batched tensors for efficient GPU processing:

```python
class ComplexGaussianSplatter(nn.Module):
    def __init__(self, positions, complex_values, scales, rotations, ...):
        # Batch storage as PyTorch parameters (optimizable)
        self._positions = nn.Parameter(positions)      # Shape: (N, 3)
        self._complex_values = nn.Parameter(complex_values)  # Shape: (N,)
        self._scales = nn.Parameter(scales)            # Shape: (N, 3)
        self._rotations = nn.Parameter(rotations)      # Shape: (N, 4)
        
        # Derived data (computed on-demand)
        self.primitives = []  # List of ComplexGaussianPrimitive objects
```

#### Memory Layout Optimization

1. **Contiguous Memory**: All parameters are stored in contiguous GPU memory for optimal access patterns
2. **Batch Processing**: Operations are vectorized across all primitives simultaneously
3. **Lazy Computation**: Expensive derived quantities (covariance matrices) are computed only when needed

```python
# Example: Memory usage for 1000 primitives
n_primitives = 1000
core_memory = n_primitives * 56  # 56,000 bytes = 55 KB

# Additional derived data (computed per primitive):
rotation_matrix = 9 * 4  # 3x3 matrix = 36 bytes
covariance_matrix = 9 * 4  # 3x3 matrix = 36 bytes
inv_covariance = 9 * 4   # 3x3 matrix = 36 bytes
det_covariance = 1 * 4   # scalar = 4 bytes
# Derived per primitive: 112 bytes

total_memory = core_memory + (n_primitives * 112)  # 168 KB total
```

### Capacity Scaling

The system can handle varying numbers of primitives efficiently:

```python
# Small models (< 1K primitives): ~168 KB
small_model = ComplexGaussianSplatter(
    positions=torch.randn(500, 3),
    complex_values=torch.randn(500, dtype=torch.complex64)  # Now represents radiance
)

# Medium models (1K-10K primitives): ~1.68 MB
medium_model = ComplexGaussianSplatter(
    positions=torch.randn(5000, 3),
    complex_values=torch.randn(5000, dtype=torch.complex64)  # Radiance values
)

# Large models (10K-100K primitives): ~16.8 MB
large_model = ComplexGaussianSplatter(
    positions=torch.randn(50000, 3),
    complex_values=torch.randn(50000, dtype=torch.complex64)  # Radiance values
)
```

### Dynamic Memory Management

#### Adding/Removing Primitives

```python
# Adding primitives requires tensor reallocation
def add_primitive(self, position, complex_value, scale=None, rotation=None):
    """Add a new primitive to the model."""
    # Create new tensors with increased capacity
    new_positions = torch.cat([self._positions, position.unsqueeze(0)])
    new_complex_values = torch.cat([self._complex_values, complex_value.unsqueeze(0)])
    
    # Update parameters (triggers reallocation)
    self._positions = nn.Parameter(new_positions)
    self._complex_values = nn.Parameter(new_complex_values)
    
    # Rebuild primitive list
    self._update_primitives()
```

#### Memory-Efficient Updates

```python
# In-place parameter updates (no reallocation)
model._positions[idx] = new_position  # Efficient
model._complex_values[idx] = new_value  # Efficient

# Batch updates for multiple primitives
indices = torch.tensor([0, 5, 10])
model._positions[indices] = new_positions  # Vectorized update
```

### Precomputed Data Caching

For performance, each primitive caches expensive computations:

```python
class ComplexGaussianPrimitive:
    def __init__(self, ...):
        # Cache expensive matrix operations
        self.rotation_matrix = self._quaternion_to_rotation_matrix(rotation)
        self.covariance_matrix = self._compute_covariance_matrix()
        self.inv_covariance = torch.inverse(self.covariance_matrix)
        self.det_covariance = torch.det(self.covariance_matrix)
    
    def update_scale(self, new_scale):
        """Update scale and recompute dependent matrices."""
        self.scale = new_scale
        # Recompute cached values
        self.covariance_matrix = self._compute_covariance_matrix()
        self.inv_covariance = torch.inverse(self.covariance_matrix)
        self.det_covariance = torch.det(self.covariance_matrix)
```

### Serialization and Persistence

The system supports efficient saving/loading of model state:

```python
# Save model state
model.save_state('model_checkpoint.pth')

# Primitive-level serialization
primitive_dict = primitive.to_dict()
# Returns:
# {
#     'position': numpy array (3,),
#     'complex_value': complex numpy scalar,
#     'scale': numpy array (3,),
#     'rotation': numpy array (4,),
#     'opacity': float
# }

# Restore from dictionary
restored_primitive = ComplexGaussianPrimitive.from_dict(primitive_dict, device)
```

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

## Performance Considerations

### Computational Complexity

The rendering complexity scales as O(N×M) where:
- N = number of Gaussian primitives
- M = number of query points

```python
# Rendering performance analysis
def analyze_performance(n_primitives, n_query_points):
    """
    Theoretical operations for field evaluation:
    - Per primitive-point pair: ~50 FLOPs
    - Total: N × M × 50 FLOPs
    """
    operations = n_primitives * n_query_points * 50
    return operations

# Examples:
print(f"1K primitives, 1K points: {analyze_performance(1000, 1000):,} ops")
print(f"10K primitives, 10K points: {analyze_performance(10000, 10000):,} ops")
```

### GPU Acceleration Benefits

```python
# CPU vs GPU performance comparison
import time

# CPU rendering
model_cpu = ComplexGaussianSplatter(..., device='cpu')
start = time.time()
field_cpu = model_cpu.render(query_points)
cpu_time = time.time() - start

# GPU rendering
model_gpu = ComplexGaussianSplatter(..., device='cuda')
start = time.time()
field_gpu = model_gpu.render(query_points.cuda())
gpu_time = time.time() - start

speedup = cpu_time / gpu_time
print(f"GPU speedup: {speedup:.1f}x")
```

### Memory vs Accuracy Trade-offs

```python
# Adaptive primitive count based on field complexity
def estimate_required_primitives(field_complexity, accuracy_target):
    """
    Rule of thumb: More complex fields need more primitives
    - Simple fields (single source): 10-100 primitives
    - Medium complexity (multiple sources): 100-1000 primitives  
    - High complexity (interference patterns): 1000-10000 primitives
    """
    if field_complexity == 'simple':
        return min(100, accuracy_target * 1000)
    elif field_complexity == 'medium':
        return min(1000, accuracy_target * 5000)
    else:  # complex
        return min(10000, accuracy_target * 20000)
```

### Batch Processing Strategies

```python
# Efficient batch rendering for large point clouds
def render_large_pointcloud(model, points, batch_size=10000):
    """Render field at many points using batching."""
    n_points = points.shape[0]
    results = []
    
    for i in range(0, n_points, batch_size):
        batch_points = points[i:i+batch_size]
        batch_result = model.render(batch_points)
        results.append(batch_result)
    
    return torch.cat(results, dim=0)

# Memory-efficient grid rendering
def render_grid_efficient(model, grid_shape, bounds):
    """Render on 3D grid without storing all points in memory."""
    x_vals = torch.linspace(bounds[0][0], bounds[0][1], grid_shape[0])
    y_vals = torch.linspace(bounds[1][0], bounds[1][1], grid_shape[1])
    z_vals = torch.linspace(bounds[2][0], bounds[2][1], grid_shape[2])
    
    field_grid = torch.zeros(grid_shape, dtype=torch.complex64)
    
    # Process one z-slice at a time to save memory
    for k, z in enumerate(z_vals):
        X, Y = torch.meshgrid(x_vals, y_vals, indexing='ij')
        Z = torch.full_like(X, z)
        slice_points = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=1)
        
        slice_field = model.render(slice_points)
        field_grid[:, :, k] = slice_field.reshape(grid_shape[0], grid_shape[1])
    
    return field_grid
```

## Key Advantages

1. **Continuous Representation**: Evaluate field anywhere in 3D space
2. **Differentiable**: Enables gradient-based optimization
3. **Compact**: Few parameters represent complex field distributions
4. **Flexible**: Handles arbitrary measurement geometries
5. **Physically Meaningful**: Preserves complex field properties
6. **Scalable**: Efficient GPU acceleration for large models
7. **Memory Efficient**: Optimized tensor storage and caching

## Limitations and Considerations

1. **Smoothness Assumption**: Fields must be reasonably smooth
2. **Gaussian Basis**: May not capture sharp discontinuities well
3. **Parameter Count**: Need sufficient Gaussians for complex fields
4. **Local Minima**: Optimization may get stuck in poor solutions
5. **Memory Scaling**: Large models (>100K primitives) require significant GPU memory
6. **Computational Cost**: Rendering scales quadratically with primitives and query points

## Next Steps

Now that you understand the basic concepts:

1. **Try Field Reconstruction**: [Field Reconstruction Tutorial](field_reconstruction.md)
2. **Learn Antenna Modeling**: [Antenna Modeling Tutorial](antenna_modeling.md)
3. **Explore Optimization**: [Advanced Topics](advanced_topics.md)

---

**Previous**: [Getting Started](getting_started.md) | **Next**: [Antenna Modeling](antenna_modeling.md)
