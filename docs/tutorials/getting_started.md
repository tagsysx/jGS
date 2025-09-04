# Getting Started with jGS

Welcome to jGS (Complex-valued Gaussian Splatting for RF Signal Processing)! This tutorial will guide you through the installation process and your first steps with the library.

## What is jGS?

jGS is a novel adaptation of Gaussian Splatting techniques from computer graphics to handle complex-valued RF (Radio Frequency) signals. It enables efficient representation and manipulation of electromagnetic field data through differentiable rendering techniques.

### Key Applications
- RF field reconstruction from sparse measurements
- Antenna pattern modeling and analysis
- Electromagnetic field visualization
- Signal processing and analysis
- Wireless communication system modeling

## Installation

### Prerequisites

Before installing jGS, ensure you have:
- Python 3.8 or higher
- CUDA-compatible GPU (recommended for performance)
- Git (for cloning the repository)

### Step 1: Clone the Repository

```bash
git clone https://github.com/tagsysx/jGS.git
cd jGS
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Using conda
conda create -n jgs python=3.9
conda activate jgs

# Or using venv
python -m venv jgs_env
source jgs_env/bin/activate  # On Windows: jgs_env\Scripts\activate
```

### Step 3: Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt

# Install jGS in development mode
pip install -e .
```

### Step 4: Verify Installation

```python
import jgs
print(f"jGS version: {jgs.__version__}")

# Test basic functionality
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

## Your First jGS Program

Let's create a simple example to demonstrate the basic workflow:

```python
import numpy as np
import torch
import jgs
from jgs.rf.antenna_patterns import DipoleAntenna
from jgs.visualization.plotter import RFPlotter

# Step 1: Create synthetic RF field data
frequency = 2.4e9  # 2.4 GHz
antenna_pos = np.array([0.0, 0.0, 0.0])

# Create a dipole antenna
dipole = DipoleAntenna(frequency=frequency, position=antenna_pos)

# Generate measurement points
n_points = 100
positions = np.random.uniform(-1, 1, (n_points, 3))
positions = torch.tensor(positions, dtype=torch.float32)

# Compute field from antenna
field_data = dipole.compute_pattern(positions)

# Step 2: Create Gaussian Splatting model
n_gaussians = 20
gaussian_positions = torch.randn(n_gaussians, 3) * 0.5
initial_values = torch.randn(n_gaussians, dtype=torch.complex64) * 0.1

model = jgs.ComplexGaussianSplatter(
    positions=gaussian_positions,
    complex_values=initial_values,
    device='cpu'  # Use 'cuda' if available
)

# Step 3: Render field at query points
query_points = torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.float32)
rendered_field = model.render(query_points)

print(f"Rendered field at query point: {rendered_field}")

# Step 4: Visualize results
plotter = RFPlotter()
fig = plotter.plot_field_magnitude(positions, field_data, title="RF Field Magnitude")
```

## Understanding the Workflow

The typical jGS workflow consists of four main steps:

### 1. Data Preparation
- Load or generate RF field measurements
- Preprocess data (filtering, calibration, etc.)
- Convert to appropriate tensor format

### 2. Model Creation
- Initialize Gaussian primitive positions and parameters
- Create ComplexGaussianSplatter model
- Set up device (CPU/GPU) configuration

### 3. Training/Optimization (Optional)
- Define loss function and optimizer
- Fit model parameters to measurement data
- Monitor training progress

### 4. Analysis and Visualization
- Render fields at arbitrary points
- Compute field properties (magnitude, phase, power)
- Create visualizations and plots

## Key Concepts

### Complex-valued Fields
RF electromagnetic fields are inherently complex, containing both magnitude and phase information:
```python
# Complex field representation
field = magnitude * np.exp(1j * phase)
magnitude = np.abs(field)
phase = np.angle(field)
```

### Gaussian Primitives
Each Gaussian primitive represents a localized field distribution:
- **Position**: 3D center location
- **Complex Value**: Amplitude and phase
- **Scale**: Size in each dimension
- **Rotation**: Orientation in 3D space

### Rendering
The rendering process evaluates the combined field from all Gaussian primitives:
```python
# Render field at specific points
field_values = model.render(query_points, frequency=2.4e9)
```

## Next Steps

Now that you have jGS installed and understand the basics:

1. **Try the Examples**: Run the example scripts in the `examples/` directory
2. **Read the Tutorials**: Explore more detailed tutorials for specific use cases
3. **Check the API**: Review the API documentation for detailed function references
4. **Experiment**: Create your own RF field scenarios and models

## Common Issues and Solutions

### CUDA Out of Memory
```python
# Reduce batch size or use CPU
model = jgs.ComplexGaussianSplatter(..., device='cpu')
```

### Import Errors
```bash
# Ensure jGS is properly installed
pip install -e .

# Check Python path
python -c "import sys; print(sys.path)"
```

### Visualization Issues
```bash
# Install additional plotting dependencies
pip install matplotlib plotly
```

## Getting Help

- 📖 Check the [API Documentation](../api/)
- 💡 Browse [Examples](../examples/)
- 🐛 Report issues on [GitHub](https://github.com/tagsysx/jGS/issues)
- 💬 Join discussions in the repository

---

**Next Tutorial**: [Basic Concepts](basic_concepts.md) - Learn the theoretical foundations of jGS
