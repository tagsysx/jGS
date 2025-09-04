# jGS: Complex-Valued Gaussian Splatting for RF Signal Processing

A novel adaptation of Gaussian Splatting techniques for complex-valued RF signal representation and processing.

## Overview

This project extends traditional Gaussian Splatting from computer graphics to handle complex-valued RF signals, enabling efficient representation and manipulation of electromagnetic field data through differentiable rendering techniques.

## Features

- Complex-valued Gaussian primitives for RF signal representation
- Differentiable rendering for electromagnetic fields
- CUDA-accelerated processing for real-time performance
- Support for various RF signal formats and antenna patterns
- Integration with common RF simulation tools

## Installation

```bash
# Clone the repository
git clone https://github.com/tagsysx/jGS.git
cd jGS

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Quick Start

```python
import jgs
import numpy as np

# Create complex-valued RF signal data
signal_data = np.random.complex128((1000, 3))  # 1000 points, 3D positions
complex_values = np.random.complex128(1000)    # Complex amplitudes

# Initialize Gaussian Splatting model
model = jgs.ComplexGaussianSplatter(
    positions=signal_data,
    complex_values=complex_values,
    device='cuda'
)

# Render RF field at specific points
query_points = np.random.float32((100, 3))
rendered_field = model.render(query_points)
```

## Project Structure

```
jGS/
├── jgs/                    # Main package
│   ├── core/              # Core Gaussian splatting algorithms
│   ├── rf/                # RF-specific implementations
│   ├── cuda/              # CUDA kernels and GPU acceleration
│   ├── utils/             # Utility functions
│   └── visualization/     # Visualization tools
├── examples/              # Example scripts and notebooks
├── tests/                 # Test suite
├── docs/                  # Documentation
├── .temp/                 # Temporary files (ignored by git)
└── configs/               # Configuration files
```

## License

MIT License - see LICENSE file for details.

## Contributing

Please read CONTRIBUTING.md for guidelines on contributing to this project.

## Citation

If you use this work in your research, please cite:

```bibtex
@software{jgs2025,
  title={jGS: Complex-Valued Gaussian Splatting for RF Signal Processing},
  author={Your Name},
  year={2025},
  url={https://github.com/tagsysx/jGS}
}
```
