"""
jGS: Complex-Valued Gaussian Splatting for RF Signal Processing

A novel adaptation of Gaussian Splatting techniques for complex-valued RF signal 
representation and processing.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Core imports
from .core.gaussian_splatter import ComplexGaussianSplatter
from .core.renderer import ComplexRenderer
from .rf.signal_processor import RFSignalProcessor
from .rf.antenna_patterns import AntennaPattern
from .utils.complex_math import ComplexMath
from .visualization.plotter import RFPlotter

# Make key classes available at package level
__all__ = [
    "ComplexGaussianSplatter",
    "ComplexRenderer", 
    "RFSignalProcessor",
    "AntennaPattern",
    "ComplexMath",
    "RFPlotter",
]

# Package metadata
PACKAGE_INFO = {
    "name": "jGS",
    "version": __version__,
    "description": "Complex-Valued Gaussian Splatting for RF Signal Processing",
    "author": __author__,
    "email": __email__,
    "url": "https://github.com/tagsysx/jGS",
}
