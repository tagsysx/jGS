"""
RF-specific implementations for complex-valued Gaussian Splatting.

This module contains specialized implementations for RF signal processing,
including antenna patterns, signal processing utilities, and electromagnetic
field calculations.
"""

from .signal_processor import RFSignalProcessor
from .antenna_patterns import AntennaPattern, DipoleAntenna, PatchAntenna
from .propagation import PropagationModel, FreeSpaceModel
from .measurements import RFMeasurement, MeasurementSet

__all__ = [
    "RFSignalProcessor",
    "AntennaPattern",
    "DipoleAntenna", 
    "PatchAntenna",
    "PropagationModel",
    "FreeSpaceModel",
    "RFMeasurement",
    "MeasurementSet",
]
