"""
Plotting utilities for RF field visualization.

This module provides comprehensive plotting capabilities for complex-valued
RF fields, including 2D/3D visualizations, antenna patterns, and field maps.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Optional, Tuple, Dict, Union, List
import logging
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors

logger = logging.getLogger(__name__)


class RFPlotter:
    """
    Comprehensive plotting utility for RF field visualization.
    
    This class provides various plotting methods for visualizing complex-valued
    RF fields, antenna patterns, and Gaussian Splatting results.
    """
    
    def __init__(self, style: str = 'seaborn', figsize: Tuple[int, int] = (10, 8)):
        """
        Initialize RF plotter.
        
        Args:
            style: Matplotlib style to use
            figsize: Default figure size
        """
        plt.style.use(style if style in plt.style.available else 'default')
        self.figsize = figsize
        self.default_cmap = 'viridis'
        
        logger.info(f"Initialized RFPlotter with style '{style}'")
    
    def plot_field_magnitude(
        self,
        positions: Union[np.ndarray, torch.Tensor],
        complex_field: Union[np.ndarray, torch.Tensor],
        title: str = "Field Magnitude",
        save_path: Optional[str] = None,
        **kwargs
    ) -> plt.Figure:
        """
        Plot field magnitude in 2D or 3D.
        
        Args:
            positions: Field positions (N, 2) or (N, 3)
            complex_field: Complex field values (N,)
            title: Plot title
            save_path: Optional path to save figure
            **kwargs: Additional plotting arguments
            
        Returns:
            Matplotlib figure object
        """
        # Convert to numpy if needed
        if isinstance(positions, torch.Tensor):
            positions = positions.detach().cpu().numpy()
        if isinstance(complex_field, torch.Tensor):
            complex_field = complex_field.detach().cpu().numpy()
        
        magnitude = np.abs(complex_field)
        
        if positions.shape[1] == 2:
            return self._plot_2d_field(positions, magnitude, title, save_path, **kwargs)
        elif positions.shape[1] == 3:
            return self._plot_3d_field(positions, magnitude, title, save_path, **kwargs)
        else:
            raise ValueError("Positions must be 2D or 3D")
    
    def plot_field_phase(
        self,
        positions: Union[np.ndarray, torch.Tensor],
        complex_field: Union[np.ndarray, torch.Tensor],
        title: str = "Field Phase",
        unwrap: bool = True,
        save_path: Optional[str] = None,
        **kwargs
    ) -> plt.Figure:
        """
        Plot field phase in 2D or 3D.
        
        Args:
            positions: Field positions (N, 2) or (N, 3)
            complex_field: Complex field values (N,)
            title: Plot title
            unwrap: Whether to unwrap phase
            save_path: Optional path to save figure
            **kwargs: Additional plotting arguments
            
        Returns:
            Matplotlib figure object
        """
        # Convert to numpy if needed
        if isinstance(positions, torch.Tensor):
            positions = positions.detach().cpu().numpy()
        if isinstance(complex_field, torch.Tensor):
            complex_field = complex_field.detach().cpu().numpy()
        
        phase = np.angle(complex_field)
        if unwrap:
            phase = np.unwrap(phase)
        
        # Use phase-specific colormap
        kwargs.setdefault('cmap', 'hsv')
        
        if positions.shape[1] == 2:
            return self._plot_2d_field(positions, phase, title, save_path, **kwargs)
        elif positions.shape[1] == 3:
            return self._plot_3d_field(positions, phase, title, save_path, **kwargs)
        else:
            raise ValueError("Positions must be 2D or 3D")
    
    def plot_field_power(
        self,
        positions: Union[np.ndarray, torch.Tensor],
        complex_field: Union[np.ndarray, torch.Tensor],
        db_scale: bool = True,
        title: str = "Field Power",
        save_path: Optional[str] = None,
        **kwargs
    ) -> plt.Figure:
        """
        Plot field power (|E|²).
        
        Args:
            positions: Field positions (N, 2) or (N, 3)
            complex_field: Complex field values (N,)
            db_scale: Whether to plot in dB scale
            title: Plot title
            save_path: Optional path to save figure
            **kwargs: Additional plotting arguments
            
        Returns:
            Matplotlib figure object
        """
        # Convert to numpy if needed
        if isinstance(positions, torch.Tensor):
            positions = positions.detach().cpu().numpy()
        if isinstance(complex_field, torch.Tensor):
            complex_field = complex_field.detach().cpu().numpy()
        
        power = np.abs(complex_field) ** 2
        
        if db_scale:
            power = 10 * np.log10(power + 1e-12)
            title += " (dB)"
        
        if positions.shape[1] == 2:
            return self._plot_2d_field(positions, power, title, save_path, **kwargs)
        elif positions.shape[1] == 3:
            return self._plot_3d_field(positions, power, title, save_path, **kwargs)
        else:
            raise ValueError("Positions must be 2D or 3D")
    
    def _plot_2d_field(
        self,
        positions: np.ndarray,
        field_values: np.ndarray,
        title: str,
        save_path: Optional[str] = None,
        **kwargs
    ) -> plt.Figure:
        """Plot 2D field distribution."""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Extract plotting parameters
        cmap = kwargs.get('cmap', self.default_cmap)
        vmin = kwargs.get('vmin', None)
        vmax = kwargs.get('vmax', None)
        
        # Create scatter plot
        scatter = ax.scatter(
            positions[:, 0], positions[:, 1], 
            c=field_values, cmap=cmap, 
            vmin=vmin, vmax=vmax,
            s=kwargs.get('s', 50),
            alpha=kwargs.get('alpha', 0.8)
        )
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(kwargs.get('colorbar_label', 'Field Value'))
        
        # Formatting
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")
        
        return fig
    
    def _plot_3d_field(
        self,
        positions: np.ndarray,
        field_values: np.ndarray,
        title: str,
        save_path: Optional[str] = None,
        **kwargs
    ) -> plt.Figure:
        """Plot 3D field distribution."""
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Extract plotting parameters
        cmap = kwargs.get('cmap', self.default_cmap)
        vmin = kwargs.get('vmin', None)
        vmax = kwargs.get('vmax', None)
        
        # Create 3D scatter plot
        scatter = ax.scatter(
            positions[:, 0], positions[:, 1], positions[:, 2],
            c=field_values, cmap=cmap,
            vmin=vmin, vmax=vmax,
            s=kwargs.get('s', 50),
            alpha=kwargs.get('alpha', 0.6)
        )
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label(kwargs.get('colorbar_label', 'Field Value'))
        
        # Formatting
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(title)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")
        
        return fig
    
    def plot_antenna_pattern(
        self,
        theta: np.ndarray,
        phi: np.ndarray,
        pattern: np.ndarray,
        pattern_type: str = 'magnitude',
        projection: str = 'polar',
        title: str = "Antenna Pattern",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot antenna radiation pattern.
        
        Args:
            theta: Elevation angles (radians)
            phi: Azimuth angles (radians)
            pattern: Pattern values
            pattern_type: Type of pattern ('magnitude', 'power', 'db')
            projection: Plot projection ('polar', 'cartesian', '3d')
            title: Plot title
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure object
        """
        if projection == 'polar':
            return self._plot_polar_pattern(theta, pattern, title, save_path)
        elif projection == 'cartesian':
            return self._plot_cartesian_pattern(theta, phi, pattern, title, save_path)
        elif projection == '3d':
            return self._plot_3d_pattern(theta, phi, pattern, title, save_path)
        else:
            raise ValueError(f"Unsupported projection: {projection}")
    
    def _plot_polar_pattern(
        self,
        theta: np.ndarray,
        pattern: np.ndarray,
        title: str,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot antenna pattern in polar coordinates."""
        fig, ax = plt.subplots(figsize=self.figsize, subplot_kw=dict(projection='polar'))
        
        # Normalize pattern for polar plot
        pattern_norm = pattern / np.max(pattern)
        
        ax.plot(theta, pattern_norm, linewidth=2)
        ax.fill(theta, pattern_norm, alpha=0.3)
        
        ax.set_title(title, pad=20)
        ax.set_ylim(0, 1)
        ax.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")
        
        return fig
    
    def _plot_cartesian_pattern(
        self,
        theta: np.ndarray,
        phi: np.ndarray,
        pattern: np.ndarray,
        title: str,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot antenna pattern in Cartesian coordinates."""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create 2D pattern plot
        if pattern.ndim == 2:
            im = ax.contourf(np.degrees(phi), np.degrees(theta), pattern, levels=20, cmap='viridis')
            plt.colorbar(im, ax=ax, label='Pattern Value')
        else:
            # 1D pattern vs theta
            ax.plot(np.degrees(theta), pattern, linewidth=2)
            ax.set_ylabel('Pattern Value')
        
        ax.set_xlabel('Azimuth (degrees)')
        ax.set_ylabel('Elevation (degrees)' if pattern.ndim == 2 else 'Pattern Value')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")
        
        return fig
    
    def _plot_3d_pattern(
        self,
        theta: np.ndarray,
        phi: np.ndarray,
        pattern: np.ndarray,
        title: str,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot 3D antenna pattern."""
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Convert to Cartesian coordinates
        THETA, PHI = np.meshgrid(theta, phi)
        R = pattern
        
        X = R * np.sin(THETA) * np.cos(PHI)
        Y = R * np.sin(THETA) * np.sin(PHI)
        Z = R * np.cos(THETA)
        
        # Create 3D surface plot
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
        
        # Add colorbar
        plt.colorbar(surf, ax=ax, shrink=0.8, label='Pattern Value')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")
        
        return fig
    
    def plot_field_comparison(
        self,
        positions: Union[np.ndarray, torch.Tensor],
        field1: Union[np.ndarray, torch.Tensor],
        field2: Union[np.ndarray, torch.Tensor],
        labels: Tuple[str, str] = ("Field 1", "Field 2"),
        comparison_type: str = 'magnitude',
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Compare two complex fields side by side.
        
        Args:
            positions: Field positions
            field1: First complex field
            field2: Second complex field
            labels: Labels for the fields
            comparison_type: Type of comparison ('magnitude', 'phase', 'difference')
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure object
        """
        # Convert to numpy if needed
        if isinstance(positions, torch.Tensor):
            positions = positions.detach().cpu().numpy()
        if isinstance(field1, torch.Tensor):
            field1 = field1.detach().cpu().numpy()
        if isinstance(field2, torch.Tensor):
            field2 = field2.detach().cpu().numpy()
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        if comparison_type == 'magnitude':
            values1 = np.abs(field1)
            values2 = np.abs(field2)
            diff_values = np.abs(values1 - values2)
            cmap = 'viridis'
        elif comparison_type == 'phase':
            values1 = np.angle(field1)
            values2 = np.angle(field2)
            diff_values = np.angle(np.exp(1j * (values1 - values2)))
            cmap = 'hsv'
        else:
            raise ValueError(f"Unsupported comparison type: {comparison_type}")
        
        # Plot first field
        if positions.shape[1] == 2:
            im1 = axes[0].scatter(positions[:, 0], positions[:, 1], c=values1, cmap=cmap, s=50)
            im2 = axes[1].scatter(positions[:, 0], positions[:, 1], c=values2, cmap=cmap, s=50)
            im3 = axes[2].scatter(positions[:, 0], positions[:, 1], c=diff_values, cmap='Reds', s=50)
        
        # Add colorbars and labels
        plt.colorbar(im1, ax=axes[0])
        plt.colorbar(im2, ax=axes[1])
        plt.colorbar(im3, ax=axes[2])
        
        axes[0].set_title(labels[0])
        axes[1].set_title(labels[1])
        axes[2].set_title('Difference')
        
        for ax in axes:
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.grid(True, alpha=0.3)
            if positions.shape[1] == 2:
                ax.set_aspect('equal')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved comparison plot to {save_path}")
        
        return fig
    
    def plot_training_history(
        self,
        history: Dict[str, List[float]],
        title: str = "Training History",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot training history from optimization.
        
        Args:
            history: Dictionary containing training metrics
            title: Plot title
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        
        # Plot training loss
        if 'train_loss' in history:
            axes[0].plot(history['train_loss'], label='Training Loss')
            if 'val_loss' in history:
                axes[0].plot(history['val_loss'], label='Validation Loss')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].set_title('Training Loss')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            axes[0].set_yscale('log')
        
        # Plot learning rate
        if 'learning_rate' in history:
            axes[1].plot(history['learning_rate'])
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Learning Rate')
            axes[1].set_title('Learning Rate Schedule')
            axes[1].grid(True, alpha=0.3)
            axes[1].set_yscale('log')
        
        # Plot additional metrics if available
        metric_idx = 2
        for key, values in history.items():
            if key not in ['train_loss', 'val_loss', 'learning_rate'] and metric_idx < 4:
                axes[metric_idx].plot(values)
                axes[metric_idx].set_xlabel('Epoch')
                axes[metric_idx].set_ylabel(key.replace('_', ' ').title())
                axes[metric_idx].set_title(key.replace('_', ' ').title())
                axes[metric_idx].grid(True, alpha=0.3)
                metric_idx += 1
        
        # Hide unused subplots
        for i in range(metric_idx, 4):
            axes[i].set_visible(False)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved training history to {save_path}")
        
        return fig
