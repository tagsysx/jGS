"""
Antenna pattern implementations for RF Gaussian Splatting.

This module provides various antenna pattern models that can be used
to generate synthetic RF field data or to incorporate antenna characteristics
into the Gaussian Splatting representation.
"""

import torch
import numpy as np
from typing import Optional, Tuple, Dict, Union
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class AntennaPattern(ABC):
    """
    Abstract base class for antenna patterns.
    
    This class defines the interface for antenna pattern implementations
    used in RF Gaussian Splatting applications.
    """
    
    def __init__(
        self,
        frequency: float,
        position: np.ndarray,
        orientation: Optional[np.ndarray] = None,
        device: str = 'cuda'
    ):
        """
        Initialize antenna pattern.
        
        Args:
            frequency: Operating frequency in Hz
            position: Antenna position (3,)
            orientation: Antenna orientation as Euler angles [roll, pitch, yaw] in radians
            device: Device for computations
        """
        self.frequency = frequency
        self.wavelength = 3e8 / frequency
        self.position = torch.tensor(position, dtype=torch.float32, device=device)
        self.device = torch.device(device)
        
        if orientation is None:
            orientation = np.array([0.0, 0.0, 0.0])
        self.orientation = torch.tensor(orientation, dtype=torch.float32, device=device)
        
        # Compute rotation matrix from Euler angles
        self.rotation_matrix = self._euler_to_rotation_matrix(self.orientation)
    
    def _euler_to_rotation_matrix(self, euler_angles: torch.Tensor) -> torch.Tensor:
        """Convert Euler angles to rotation matrix."""
        roll, pitch, yaw = euler_angles[0], euler_angles[1], euler_angles[2]
        
        # Rotation matrices for each axis
        Rx = torch.tensor([
            [1, 0, 0],
            [0, torch.cos(roll), -torch.sin(roll)],
            [0, torch.sin(roll), torch.cos(roll)]
        ], device=self.device, dtype=torch.float32)
        
        Ry = torch.tensor([
            [torch.cos(pitch), 0, torch.sin(pitch)],
            [0, 1, 0],
            [-torch.sin(pitch), 0, torch.cos(pitch)]
        ], device=self.device, dtype=torch.float32)
        
        Rz = torch.tensor([
            [torch.cos(yaw), -torch.sin(yaw), 0],
            [torch.sin(yaw), torch.cos(yaw), 0],
            [0, 0, 1]
        ], device=self.device, dtype=torch.float32)
        
        # Combined rotation matrix (ZYX order)
        R = Rz @ Ry @ Rx
        return R
    
    def _cartesian_to_spherical(self, positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert Cartesian coordinates to spherical coordinates relative to antenna.
        
        Args:
            positions: Cartesian positions (N, 3)
            
        Returns:
            Tuple of (r, theta, phi) where theta is elevation, phi is azimuth
        """
        # Translate to antenna coordinate system
        relative_pos = positions - self.position.unsqueeze(0)
        
        # Rotate to antenna frame
        rotated_pos = relative_pos @ self.rotation_matrix.T
        
        x, y, z = rotated_pos[:, 0], rotated_pos[:, 1], rotated_pos[:, 2]
        
        r = torch.sqrt(x**2 + y**2 + z**2)
        theta = torch.acos(torch.clamp(z / (r + 1e-8), -1, 1))  # Elevation from z-axis
        phi = torch.atan2(y, x)  # Azimuth in xy-plane
        
        return r, theta, phi
    
    @abstractmethod
    def compute_pattern(
        self,
        positions: torch.Tensor,
        polarization: str = 'linear'
    ) -> torch.Tensor:
        """
        Compute antenna pattern at given positions.
        
        Args:
            positions: Positions to evaluate pattern at (N, 3)
            polarization: Polarization type ('linear', 'circular')
            
        Returns:
            Complex field values (N,)
        """
        pass
    
    def compute_gain(
        self,
        positions: torch.Tensor,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Compute antenna gain pattern.
        
        Args:
            positions: Positions to evaluate gain at (N, 3)
            normalize: Whether to normalize to maximum gain
            
        Returns:
            Gain values in linear scale (N,)
        """
        field_pattern = self.compute_pattern(positions)
        gain = torch.abs(field_pattern) ** 2
        
        if normalize:
            gain = gain / torch.max(gain)
        
        return gain
    
    def compute_directivity(self, num_samples: int = 10000) -> float:
        """
        Compute antenna directivity by numerical integration.
        
        Args:
            num_samples: Number of samples for integration
            
        Returns:
            Directivity in linear scale
        """
        # Generate uniform samples on sphere
        u = torch.rand(num_samples, device=self.device)
        v = torch.rand(num_samples, device=self.device)
        
        theta = torch.acos(2 * u - 1)  # Elevation
        phi = 2 * np.pi * v  # Azimuth
        
        # Convert to Cartesian (unit sphere)
        x = torch.sin(theta) * torch.cos(phi)
        y = torch.sin(theta) * torch.sin(phi)
        z = torch.cos(theta)
        
        sphere_points = torch.stack([x, y, z], dim=1) + self.position.unsqueeze(0)
        
        # Compute pattern
        pattern = self.compute_pattern(sphere_points)
        power_pattern = torch.abs(pattern) ** 2
        
        # Numerical integration (average over sphere)
        avg_power = torch.mean(power_pattern)
        max_power = torch.max(power_pattern)
        
        directivity = max_power / avg_power
        return directivity.item()


class DipoleAntenna(AntennaPattern):
    """
    Electric dipole antenna pattern implementation.
    
    This class implements the radiation pattern of a short electric dipole
    antenna, which is commonly used in RF applications.
    """
    
    def __init__(
        self,
        frequency: float,
        position: np.ndarray,
        length: Optional[float] = None,
        orientation: Optional[np.ndarray] = None,
        device: str = 'cuda'
    ):
        """
        Initialize dipole antenna.
        
        Args:
            frequency: Operating frequency in Hz
            position: Antenna position (3,)
            length: Dipole length in meters (defaults to λ/2)
            orientation: Antenna orientation [roll, pitch, yaw] in radians
            device: Device for computations
        """
        super().__init__(frequency, position, orientation, device)
        
        if length is None:
            length = self.wavelength / 2  # Half-wave dipole
        self.length = length
        
        logger.info(f"Initialized dipole antenna at {frequency/1e9:.2f} GHz, length {length:.3f}m")
    
    def compute_pattern(
        self,
        positions: torch.Tensor,
        polarization: str = 'linear'
    ) -> torch.Tensor:
        """
        Compute dipole radiation pattern.
        
        The dipole pattern is given by:
        E(θ) ∝ sin(θ) * exp(-jkr) / r
        
        where θ is the angle from the dipole axis.
        """
        r, theta, phi = self._cartesian_to_spherical(positions)
        
        # Dipole pattern: sin(theta) where theta is angle from z-axis (dipole axis)
        pattern_amplitude = torch.sin(theta)
        
        # Add length factor for finite dipole
        if self.length < self.wavelength:
            # Short dipole approximation
            length_factor = self.length / self.wavelength
        else:
            # Finite length correction
            k = 2 * np.pi / self.wavelength
            beta_l = k * self.length / 2
            length_factor = torch.abs(torch.cos(beta_l * torch.cos(theta)) - torch.cos(beta_l)) / torch.sin(theta)
        
        pattern_amplitude *= length_factor
        
        # Free space propagation
        propagation_factor = torch.exp(-1j * 2 * np.pi * r / self.wavelength) / (r + 1e-8)
        
        # Combine amplitude and propagation
        field_pattern = pattern_amplitude * propagation_factor
        
        # Handle polarization
        if polarization == 'circular':
            # Add 90-degree phase shift for circular polarization
            field_pattern *= (1 + 1j) / np.sqrt(2)
        
        return field_pattern


class PatchAntenna(AntennaPattern):
    """
    Microstrip patch antenna pattern implementation.
    
    This class implements the radiation pattern of a rectangular microstrip
    patch antenna, commonly used in wireless communications.
    """
    
    def __init__(
        self,
        frequency: float,
        position: np.ndarray,
        width: Optional[float] = None,
        height: Optional[float] = None,
        substrate_height: float = 1.6e-3,
        dielectric_constant: float = 4.4,
        orientation: Optional[np.ndarray] = None,
        device: str = 'cuda'
    ):
        """
        Initialize patch antenna.
        
        Args:
            frequency: Operating frequency in Hz
            position: Antenna position (3,)
            width: Patch width in meters (defaults to calculated value)
            height: Patch height in meters (defaults to calculated value)
            substrate_height: Substrate thickness in meters
            dielectric_constant: Relative dielectric constant
            orientation: Antenna orientation [roll, pitch, yaw] in radians
            device: Device for computations
        """
        super().__init__(frequency, position, orientation, device)
        
        self.substrate_height = substrate_height
        self.dielectric_constant = dielectric_constant
        
        # Calculate effective dielectric constant
        self.eps_eff = (dielectric_constant + 1) / 2 + (dielectric_constant - 1) / 2 * \
                      (1 + 12 * substrate_height / (width if width else self.wavelength/2)) ** (-0.5)
        
        # Calculate patch dimensions if not provided
        if width is None:
            width = 3e8 / (2 * frequency * np.sqrt(self.eps_eff))
        if height is None:
            height = width * 0.75  # Typical aspect ratio
        
        self.width = width
        self.height = height
        
        logger.info(f"Initialized patch antenna: {width*1000:.1f}mm x {height*1000:.1f}mm")
    
    def compute_pattern(
        self,
        positions: torch.Tensor,
        polarization: str = 'linear'
    ) -> torch.Tensor:
        """
        Compute patch antenna radiation pattern.
        
        The patch pattern is approximated by:
        E(θ,φ) ∝ cos(θ) * sinc(kW sin(θ)cos(φ)/2) * sinc(kL sin(θ)sin(φ)/2)
        """
        r, theta, phi = self._cartesian_to_spherical(positions)
        
        k = 2 * np.pi / self.wavelength
        
        # Patch pattern factors
        cos_factor = torch.cos(theta)  # Cosine pattern in elevation
        
        # Sinc patterns in both dimensions
        kw_factor = k * self.width * torch.sin(theta) * torch.cos(phi) / 2
        kh_factor = k * self.height * torch.sin(theta) * torch.sin(phi) / 2
        
        # Sinc function: sinc(x) = sin(x)/x
        sinc_w = torch.sinc(kw_factor / np.pi)  # PyTorch sinc is normalized
        sinc_h = torch.sinc(kh_factor / np.pi)
        
        # Combined pattern
        pattern_amplitude = cos_factor * sinc_w * sinc_h
        
        # Ensure pattern is zero for theta > 90 degrees (back radiation)
        pattern_amplitude = pattern_amplitude * (theta < np.pi/2).float()
        
        # Free space propagation
        propagation_factor = torch.exp(-1j * k * r) / (r + 1e-8)
        
        # Combine amplitude and propagation
        field_pattern = pattern_amplitude * propagation_factor
        
        # Handle polarization
        if polarization == 'circular':
            # Add 90-degree phase shift for circular polarization
            field_pattern *= (1 + 1j) / np.sqrt(2)
        
        return field_pattern


class ArrayAntenna(AntennaPattern):
    """
    Antenna array pattern implementation.
    
    This class implements the radiation pattern of an antenna array
    by combining multiple individual antenna elements.
    """
    
    def __init__(
        self,
        element_antennas: list,
        element_weights: Optional[torch.Tensor] = None,
        device: str = 'cuda'
    ):
        """
        Initialize antenna array.
        
        Args:
            element_antennas: List of individual antenna elements
            element_weights: Complex weights for each element (N,)
            device: Device for computations
        """
        if not element_antennas:
            raise ValueError("At least one antenna element required")
        
        # Use first element's properties for base class
        first_antenna = element_antennas[0]
        super().__init__(
            first_antenna.frequency,
            first_antenna.position.cpu().numpy(),
            device=device
        )
        
        self.element_antennas = element_antennas
        
        if element_weights is None:
            element_weights = torch.ones(len(element_antennas), dtype=torch.complex64, device=device)
        self.element_weights = element_weights
        
        logger.info(f"Initialized antenna array with {len(element_antennas)} elements")
    
    def compute_pattern(
        self,
        positions: torch.Tensor,
        polarization: str = 'linear'
    ) -> torch.Tensor:
        """
        Compute array radiation pattern by superposition.
        
        The array pattern is the sum of individual element patterns
        weighted by their complex coefficients.
        """
        total_pattern = torch.zeros(positions.shape[0], dtype=torch.complex64, device=self.device)
        
        for i, antenna in enumerate(self.element_antennas):
            element_pattern = antenna.compute_pattern(positions, polarization)
            total_pattern += self.element_weights[i] * element_pattern
        
        return total_pattern
    
    def set_beam_steering(self, steering_angle: Tuple[float, float]):
        """
        Set beam steering angles for the array.
        
        Args:
            steering_angle: Tuple of (theta, phi) steering angles in radians
        """
        theta_s, phi_s = steering_angle
        k = 2 * np.pi / self.wavelength
        
        # Calculate phase shifts for beam steering
        for i, antenna in enumerate(self.element_antennas):
            # Position relative to array center
            pos_rel = antenna.position - self.position
            
            # Phase shift for steering
            phase_shift = k * (pos_rel[0] * np.sin(theta_s) * np.cos(phi_s) +
                              pos_rel[1] * np.sin(theta_s) * np.sin(phi_s) +
                              pos_rel[2] * np.cos(theta_s))
            
            # Update element weight
            magnitude = torch.abs(self.element_weights[i])
            self.element_weights[i] = magnitude * torch.exp(-1j * phase_shift)
        
        logger.info(f"Set beam steering to θ={np.degrees(theta_s):.1f}°, φ={np.degrees(phi_s):.1f}°")
    
    def compute_array_factor(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Compute array factor (without element patterns).
        
        Args:
            positions: Positions to evaluate at (N, 3)
            
        Returns:
            Array factor values (N,)
        """
        array_factor = torch.zeros(positions.shape[0], dtype=torch.complex64, device=self.device)
        k = 2 * np.pi / self.wavelength
        
        for i, antenna in enumerate(self.element_antennas):
            # Distance from each element to observation points
            distances = torch.norm(positions - antenna.position.unsqueeze(0), dim=1)
            
            # Phase contribution from this element
            phase_contrib = self.element_weights[i] * torch.exp(-1j * k * distances)
            array_factor += phase_contrib
        
        return array_factor
