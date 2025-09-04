"""
RF signal processing utilities for complex-valued Gaussian Splatting.

This module provides tools for processing RF measurements and converting
them into formats suitable for Gaussian Splatting representation.
"""

import torch
import numpy as np
from typing import Optional, Tuple, Dict, List, Union
import logging
from scipy import signal
from scipy.interpolate import griddata

logger = logging.getLogger(__name__)


class RFSignalProcessor:
    """
    Processor for RF signal measurements and field data.
    
    This class handles the preprocessing, filtering, and conversion of
    RF measurements into complex field representations suitable for
    Gaussian Splatting.
    """
    
    def __init__(
        self,
        frequency: float,
        c_light: float = 3e8,
        device: str = 'cuda'
    ):
        """
        Initialize RF signal processor.
        
        Args:
            frequency: Operating frequency in Hz
            c_light: Speed of light in m/s
            device: Device for computations
        """
        self.frequency = frequency
        self.wavelength = c_light / frequency
        self.k = 2 * np.pi / self.wavelength  # Wave number
        self.c_light = c_light
        self.device = torch.device(device)
        
        logger.info(f"Initialized RFSignalProcessor at {frequency/1e9:.2f} GHz")
    
    def preprocess_measurements(
        self,
        positions: np.ndarray,
        measurements: np.ndarray,
        measurement_type: str = 'complex_field',
        reference_position: Optional[np.ndarray] = None,
        noise_threshold: float = -60.0  # dB
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Preprocess raw RF measurements.
        
        Args:
            positions: Measurement positions (N, 3)
            measurements: Raw measurements (N,) or (N, 2) for I/Q
            measurement_type: Type of measurement ('complex_field', 'iq', 'magnitude_phase', 'power')
            reference_position: Reference position for phase calibration
            noise_threshold: Noise threshold in dB for filtering
            
        Returns:
            Tuple of (processed_positions, complex_field_values)
        """
        # Convert to tensors
        positions_tensor = torch.tensor(positions, dtype=torch.float32, device=self.device)
        
        # Process measurements based on type
        if measurement_type == 'complex_field':
            if measurements.dtype == np.complex64 or measurements.dtype == np.complex128:
                complex_values = torch.tensor(measurements, dtype=torch.complex64, device=self.device)
            else:
                raise ValueError("Complex field measurements must be complex-valued")
        
        elif measurement_type == 'iq':
            if measurements.shape[1] != 2:
                raise ValueError("I/Q measurements must have shape (N, 2)")
            i_values = measurements[:, 0]
            q_values = measurements[:, 1]
            complex_values = torch.tensor(i_values + 1j * q_values, dtype=torch.complex64, device=self.device)
        
        elif measurement_type == 'magnitude_phase':
            if measurements.shape[1] != 2:
                raise ValueError("Magnitude/phase measurements must have shape (N, 2)")
            magnitude = measurements[:, 0]
            phase = measurements[:, 1]
            complex_values = torch.tensor(
                magnitude * np.exp(1j * phase), 
                dtype=torch.complex64, 
                device=self.device
            )
        
        elif measurement_type == 'power':
            # Convert power to magnitude (assume phase = 0)
            magnitude = np.sqrt(measurements)
            complex_values = torch.tensor(magnitude, dtype=torch.complex64, device=self.device)
        
        else:
            raise ValueError(f"Unsupported measurement type: {measurement_type}")
        
        # Apply noise filtering
        power_db = 20 * torch.log10(torch.abs(complex_values) + 1e-12)
        valid_mask = power_db > noise_threshold
        
        filtered_positions = positions_tensor[valid_mask]
        filtered_values = complex_values[valid_mask]
        
        # Phase calibration if reference position provided
        if reference_position is not None:
            ref_pos = torch.tensor(reference_position, dtype=torch.float32, device=self.device)
            filtered_values = self._calibrate_phase(filtered_positions, filtered_values, ref_pos)
        
        logger.info(f"Preprocessed {len(filtered_values)} measurements from {len(measurements)} raw samples")
        
        return filtered_positions, filtered_values
    
    def _calibrate_phase(
        self,
        positions: torch.Tensor,
        complex_values: torch.Tensor,
        reference_position: torch.Tensor
    ) -> torch.Tensor:
        """Calibrate phase relative to reference position."""
        # Find closest measurement to reference
        distances = torch.norm(positions - reference_position.unsqueeze(0), dim=1)
        ref_idx = torch.argmin(distances)
        
        # Extract reference phase
        ref_phase = torch.angle(complex_values[ref_idx])
        
        # Remove reference phase from all measurements
        calibrated_values = complex_values * torch.exp(-1j * ref_phase)
        
        return calibrated_values
    
    def interpolate_field(
        self,
        positions: torch.Tensor,
        complex_values: torch.Tensor,
        query_positions: torch.Tensor,
        method: str = 'linear'
    ) -> torch.Tensor:
        """
        Interpolate complex field to new positions.
        
        Args:
            positions: Original measurement positions (N, 3)
            complex_values: Complex field values (N,)
            query_positions: Positions to interpolate to (M, 3)
            method: Interpolation method ('linear', 'cubic', 'nearest')
            
        Returns:
            Interpolated complex field values (M,)
        """
        # Convert to numpy for scipy interpolation
        pos_np = positions.cpu().numpy()
        values_np = complex_values.cpu().numpy()
        query_np = query_positions.cpu().numpy()
        
        # Interpolate real and imaginary parts separately
        real_interp = griddata(pos_np, values_np.real, query_np, method=method, fill_value=0.0)
        imag_interp = griddata(pos_np, values_np.imag, query_np, method=method, fill_value=0.0)
        
        # Combine and convert back to tensor
        interpolated = torch.tensor(
            real_interp + 1j * imag_interp,
            dtype=torch.complex64,
            device=self.device
        )
        
        return interpolated
    
    def apply_propagation_correction(
        self,
        positions: torch.Tensor,
        complex_values: torch.Tensor,
        source_position: torch.Tensor,
        correction_type: str = 'free_space'
    ) -> torch.Tensor:
        """
        Apply propagation model corrections.
        
        Args:
            positions: Measurement positions (N, 3)
            complex_values: Complex field values (N,)
            source_position: Source/transmitter position (3,)
            correction_type: Type of correction ('free_space', 'two_ray')
            
        Returns:
            Corrected complex field values (N,)
        """
        distances = torch.norm(positions - source_position.unsqueeze(0), dim=1)
        
        if correction_type == 'free_space':
            # Free space path loss and phase
            path_loss = 1.0 / (4 * np.pi * distances)
            phase_shift = torch.exp(-1j * self.k * distances)
            correction = path_loss * phase_shift
        
        elif correction_type == 'two_ray':
            # Simplified two-ray model (ground reflection)
            # This is a basic implementation - more sophisticated models can be added
            direct_path = distances
            # Assume ground at z=0 and specular reflection
            ground_positions = positions.clone()
            ground_positions[:, 2] = -ground_positions[:, 2]  # Mirror z-coordinate
            reflected_distances = torch.norm(ground_positions - source_position.unsqueeze(0), dim=1)
            
            # Direct and reflected components
            direct_loss = 1.0 / (4 * np.pi * direct_path)
            direct_phase = torch.exp(-1j * self.k * direct_path)
            
            reflected_loss = -1.0 / (4 * np.pi * reflected_distances)  # Phase inversion
            reflected_phase = torch.exp(-1j * self.k * reflected_distances)
            
            correction = direct_loss * direct_phase + reflected_loss * reflected_phase
        
        else:
            raise ValueError(f"Unsupported correction type: {correction_type}")
        
        corrected_values = complex_values / correction
        
        return corrected_values
    
    def filter_measurements(
        self,
        complex_values: torch.Tensor,
        filter_type: str = 'median',
        **filter_kwargs
    ) -> torch.Tensor:
        """
        Apply filtering to complex measurements.
        
        Args:
            complex_values: Complex field values (N,)
            filter_type: Type of filter ('median', 'gaussian', 'butterworth')
            **filter_kwargs: Additional filter parameters
            
        Returns:
            Filtered complex field values (N,)
        """
        values_np = complex_values.cpu().numpy()
        
        if filter_type == 'median':
            kernel_size = filter_kwargs.get('kernel_size', 3)
            # Apply median filter to magnitude, preserve phase
            magnitude = np.abs(values_np)
            phase = np.angle(values_np)
            
            filtered_mag = signal.medfilt(magnitude, kernel_size=kernel_size)
            filtered_values = filtered_mag * np.exp(1j * phase)
        
        elif filter_type == 'gaussian':
            sigma = filter_kwargs.get('sigma', 1.0)
            # Apply Gaussian filter to real and imaginary parts
            real_filtered = signal.gaussian_filter1d(values_np.real, sigma=sigma)
            imag_filtered = signal.gaussian_filter1d(values_np.imag, sigma=sigma)
            filtered_values = real_filtered + 1j * imag_filtered
        
        elif filter_type == 'butterworth':
            cutoff = filter_kwargs.get('cutoff', 0.1)
            order = filter_kwargs.get('order', 4)
            
            # Design Butterworth filter
            b, a = signal.butter(order, cutoff, btype='low')
            
            # Apply to real and imaginary parts
            real_filtered = signal.filtfilt(b, a, values_np.real)
            imag_filtered = signal.filtfilt(b, a, values_np.imag)
            filtered_values = real_filtered + 1j * imag_filtered
        
        else:
            raise ValueError(f"Unsupported filter type: {filter_type}")
        
        return torch.tensor(filtered_values, dtype=torch.complex64, device=self.device)
    
    def compute_field_statistics(
        self,
        complex_values: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute statistical properties of the complex field.
        
        Args:
            complex_values: Complex field values (N,)
            
        Returns:
            Dictionary of field statistics
        """
        magnitude = torch.abs(complex_values)
        phase = torch.angle(complex_values)
        power = magnitude ** 2
        
        stats = {
            'mean_magnitude': torch.mean(magnitude).item(),
            'std_magnitude': torch.std(magnitude).item(),
            'max_magnitude': torch.max(magnitude).item(),
            'min_magnitude': torch.min(magnitude).item(),
            'mean_power_db': 10 * torch.log10(torch.mean(power) + 1e-12).item(),
            'dynamic_range_db': (10 * torch.log10(torch.max(power) + 1e-12) - 
                                10 * torch.log10(torch.min(power) + 1e-12)).item(),
            'phase_std': torch.std(phase).item(),
            'snr_estimate_db': self._estimate_snr(complex_values).item()
        }
        
        return stats
    
    def _estimate_snr(self, complex_values: torch.Tensor) -> torch.Tensor:
        """Estimate signal-to-noise ratio from measurements."""
        # Simple SNR estimation based on signal variance
        magnitude = torch.abs(complex_values)
        signal_power = torch.mean(magnitude ** 2)
        
        # Estimate noise from high-frequency components
        # This is a simplified approach - more sophisticated methods exist
        diff = magnitude[1:] - magnitude[:-1]
        noise_power = torch.var(diff) / 2  # Divide by 2 for differencing effect
        
        snr_linear = signal_power / (noise_power + 1e-12)
        snr_db = 10 * torch.log10(snr_linear)
        
        return snr_db
    
    def export_measurements(
        self,
        positions: torch.Tensor,
        complex_values: torch.Tensor,
        filename: str,
        format: str = 'npz'
    ):
        """
        Export processed measurements to file.
        
        Args:
            positions: Measurement positions (N, 3)
            complex_values: Complex field values (N,)
            filename: Output filename
            format: Export format ('npz', 'csv', 'mat')
        """
        pos_np = positions.cpu().numpy()
        values_np = complex_values.cpu().numpy()
        
        if format == 'npz':
            np.savez(filename, 
                    positions=pos_np,
                    complex_values=values_np,
                    frequency=self.frequency,
                    wavelength=self.wavelength)
        
        elif format == 'csv':
            # Export as CSV with columns: x, y, z, real, imag, magnitude, phase
            data = np.column_stack([
                pos_np,
                values_np.real,
                values_np.imag,
                np.abs(values_np),
                np.angle(values_np)
            ])
            header = 'x,y,z,real,imag,magnitude,phase'
            np.savetxt(filename, data, delimiter=',', header=header, comments='')
        
        elif format == 'mat':
            from scipy.io import savemat
            savemat(filename, {
                'positions': pos_np,
                'complex_values': values_np,
                'frequency': self.frequency,
                'wavelength': self.wavelength
            })
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        logger.info(f"Exported measurements to {filename}")
    
    @classmethod
    def load_measurements(
        cls,
        filename: str,
        device: str = 'cuda'
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Load measurements from file.
        
        Args:
            filename: Input filename
            device: Device for tensors
            
        Returns:
            Tuple of (positions, complex_values, frequency)
        """
        if filename.endswith('.npz'):
            data = np.load(filename)
            positions = torch.tensor(data['positions'], dtype=torch.float32, device=device)
            complex_values = torch.tensor(data['complex_values'], dtype=torch.complex64, device=device)
            frequency = float(data['frequency'])
        
        elif filename.endswith('.csv'):
            data = np.loadtxt(filename, delimiter=',', skiprows=1)
            positions = torch.tensor(data[:, :3], dtype=torch.float32, device=device)
            real_part = data[:, 3]
            imag_part = data[:, 4]
            complex_values = torch.tensor(real_part + 1j * imag_part, dtype=torch.complex64, device=device)
            frequency = None  # Not stored in CSV format
        
        else:
            raise ValueError(f"Unsupported file format: {filename}")
        
        return positions, complex_values, frequency
