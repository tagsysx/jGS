"""
Optimization algorithms for complex-valued Gaussian Splatting.

This module provides optimization routines specifically designed for
complex-valued RF field reconstruction and parameter estimation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import numpy as np
from typing import Optional, Dict, List, Callable, Tuple, Any
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ComplexOptimizer:
    """
    Optimizer for complex-valued Gaussian Splatting parameters.
    
    This class handles the optimization of Gaussian primitive parameters
    to fit complex-valued RF field measurements.
    """
    
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-3,
        optimizer_type: str = 'adam',
        scheduler_type: Optional[str] = 'plateau',
        device: str = 'cuda'
    ):
        """
        Initialize the complex optimizer.
        
        Args:
            model: ComplexGaussianSplatter model to optimize
            learning_rate: Initial learning rate
            optimizer_type: Type of optimizer ('adam', 'sgd', 'rmsprop')
            scheduler_type: Type of learning rate scheduler ('plateau', 'cosine', None)
            device: Device to run optimization on
        """
        self.model = model
        self.device = torch.device(device)
        self.learning_rate = learning_rate
        
        # Initialize optimizer
        if optimizer_type.lower() == 'adam':
            self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_type.lower() == 'sgd':
            self.optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        elif optimizer_type.lower() == 'rmsprop':
            self.optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
        
        # Initialize scheduler
        self.scheduler = None
        if scheduler_type == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
            )
        elif scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=100)
        
        self.loss_history = []
        self.current_epoch = 0
        
        logger.info(f"Initialized ComplexOptimizer with {optimizer_type} optimizer")
    
    def complex_mse_loss(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute complex-valued mean squared error loss.
        
        Args:
            predicted: Predicted complex field values (N,)
            target: Target complex field values (N,)
            weights: Optional weights for each sample (N,)
            
        Returns:
            Complex MSE loss
        """
        diff = predicted - target
        loss = torch.mean(torch.abs(diff) ** 2)
        
        if weights is not None:
            loss = torch.mean(weights * torch.abs(diff) ** 2)
        
        return loss
    
    def magnitude_phase_loss(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        magnitude_weight: float = 1.0,
        phase_weight: float = 1.0,
        weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute separate magnitude and phase losses.
        
        Args:
            predicted: Predicted complex field values (N,)
            target: Target complex field values (N,)
            magnitude_weight: Weight for magnitude loss
            phase_weight: Weight for phase loss
            weights: Optional weights for each sample (N,)
            
        Returns:
            Combined magnitude and phase loss
        """
        # Magnitude loss
        pred_mag = torch.abs(predicted)
        target_mag = torch.abs(target)
        mag_loss = torch.mean((pred_mag - target_mag) ** 2)
        
        # Phase loss (handle phase wrapping)
        pred_phase = torch.angle(predicted)
        target_phase = torch.angle(target)
        phase_diff = torch.angle(torch.exp(1j * (pred_phase - target_phase)))
        phase_loss = torch.mean(phase_diff ** 2)
        
        # Apply weights if provided
        if weights is not None:
            mag_loss = torch.mean(weights * (pred_mag - target_mag) ** 2)
            phase_loss = torch.mean(weights * phase_diff ** 2)
        
        total_loss = magnitude_weight * mag_loss + phase_weight * phase_loss
        
        return total_loss
    
    def regularization_loss(
        self,
        l1_weight: float = 0.0,
        l2_weight: float = 0.0,
        sparsity_weight: float = 0.0
    ) -> torch.Tensor:
        """
        Compute regularization losses.
        
        Args:
            l1_weight: Weight for L1 regularization
            l2_weight: Weight for L2 regularization
            sparsity_weight: Weight for sparsity regularization
            
        Returns:
            Total regularization loss
        """
        reg_loss = torch.tensor(0.0, device=self.device)
        
        for param in self.model.parameters():
            if l1_weight > 0:
                if torch.is_complex(param):
                    reg_loss += l1_weight * torch.sum(torch.abs(param))
                else:
                    reg_loss += l1_weight * torch.sum(torch.abs(param))
            
            if l2_weight > 0:
                if torch.is_complex(param):
                    reg_loss += l2_weight * torch.sum(torch.abs(param) ** 2)
                else:
                    reg_loss += l2_weight * torch.sum(param ** 2)
        
        # Sparsity regularization (encourage fewer active primitives)
        if sparsity_weight > 0:
            complex_values = self.model.complex_values
            sparsity_loss = sparsity_weight * torch.sum(torch.abs(complex_values))
            reg_loss += sparsity_loss
        
        return reg_loss
    
    def fit(
        self,
        query_points: torch.Tensor,
        target_values: torch.Tensor,
        validation_points: Optional[torch.Tensor] = None,
        validation_values: Optional[torch.Tensor] = None,
        num_epochs: int = 1000,
        batch_size: Optional[int] = None,
        loss_function: str = 'complex_mse',
        loss_kwargs: Optional[Dict] = None,
        regularization_kwargs: Optional[Dict] = None,
        frequency: Optional[float] = None,
        verbose: bool = True,
        save_best: bool = True,
        early_stopping_patience: int = 50
    ) -> Dict[str, Any]:
        """
        Fit the model to target field measurements.
        
        Args:
            query_points: Points where field is measured (N, 3)
            target_values: Target complex field values (N,)
            validation_points: Optional validation points (M, 3)
            validation_values: Optional validation values (M,)
            num_epochs: Number of training epochs
            batch_size: Batch size for training (None for full batch)
            loss_function: Loss function to use ('complex_mse', 'magnitude_phase')
            loss_kwargs: Additional arguments for loss function
            regularization_kwargs: Arguments for regularization
            frequency: Optional frequency for rendering
            verbose: Whether to print training progress
            save_best: Whether to save best model state
            early_stopping_patience: Patience for early stopping
            
        Returns:
            Training history dictionary
        """
        if loss_kwargs is None:
            loss_kwargs = {}
        if regularization_kwargs is None:
            regularization_kwargs = {}
        
        n_samples = query_points.shape[0]
        best_loss = float('inf')
        best_state = None
        patience_counter = 0
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
        # Training loop
        pbar = tqdm(range(num_epochs), desc="Training", disable=not verbose)
        
        for epoch in pbar:
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0
            
            # Create batches if batch_size is specified
            if batch_size is not None and batch_size < n_samples:
                indices = torch.randperm(n_samples, device=self.device)
                for i in range(0, n_samples, batch_size):
                    batch_indices = indices[i:i+batch_size]
                    batch_points = query_points[batch_indices]
                    batch_targets = target_values[batch_indices]
                    
                    loss = self._train_step(
                        batch_points, batch_targets, loss_function, 
                        loss_kwargs, regularization_kwargs, frequency
                    )
                    epoch_loss += loss
                    n_batches += 1
            else:
                # Full batch training
                loss = self._train_step(
                    query_points, target_values, loss_function,
                    loss_kwargs, regularization_kwargs, frequency
                )
                epoch_loss = loss
                n_batches = 1
            
            avg_loss = epoch_loss / n_batches
            history['train_loss'].append(avg_loss)
            history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            # Validation
            val_loss = None
            if validation_points is not None and validation_values is not None:
                self.model.eval()
                with torch.no_grad():
                    val_pred = self.model.render(validation_points, frequency)
                    if loss_function == 'complex_mse':
                        val_loss = self.complex_mse_loss(val_pred, validation_values, **loss_kwargs)
                    elif loss_function == 'magnitude_phase':
                        val_loss = self.magnitude_phase_loss(val_pred, validation_values, **loss_kwargs)
                    
                    val_loss = val_loss.item()
                    history['val_loss'].append(val_loss)
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss if val_loss is not None else avg_loss)
                else:
                    self.scheduler.step()
            
            # Early stopping and best model saving
            current_loss = val_loss if val_loss is not None else avg_loss
            if current_loss < best_loss:
                best_loss = current_loss
                patience_counter = 0
                if save_best:
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
            
            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f"{avg_loss:.6f}",
                'Val Loss': f"{val_loss:.6f}" if val_loss is not None else "N/A",
                'LR': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
            
            self.current_epoch = epoch
        
        # Restore best model if saved
        if save_best and best_state is not None:
            self.model.load_state_dict(best_state)
            if verbose:
                print(f"Restored best model with loss: {best_loss:.6f}")
        
        return history
    
    def _train_step(
        self,
        query_points: torch.Tensor,
        target_values: torch.Tensor,
        loss_function: str,
        loss_kwargs: Dict,
        regularization_kwargs: Dict,
        frequency: Optional[float]
    ) -> float:
        """Perform a single training step."""
        self.optimizer.zero_grad()
        
        # Forward pass
        predicted = self.model.render(query_points, frequency)
        
        # Compute loss
        if loss_function == 'complex_mse':
            loss = self.complex_mse_loss(predicted, target_values, **loss_kwargs)
        elif loss_function == 'magnitude_phase':
            loss = self.magnitude_phase_loss(predicted, target_values, **loss_kwargs)
        else:
            raise ValueError(f"Unsupported loss function: {loss_function}")
        
        # Add regularization
        reg_loss = self.regularization_loss(**regularization_kwargs)
        total_loss = loss + reg_loss
        
        # Backward pass
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item()
    
    def evaluate(
        self,
        query_points: torch.Tensor,
        target_values: torch.Tensor,
        frequency: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Evaluate model performance on test data.
        
        Args:
            query_points: Test query points (N, 3)
            target_values: Target complex field values (N,)
            frequency: Optional frequency for rendering
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        
        with torch.no_grad():
            predicted = self.model.render(query_points, frequency)
            
            # Complex MSE
            complex_mse = self.complex_mse_loss(predicted, target_values).item()
            
            # Magnitude and phase errors
            pred_mag = torch.abs(predicted)
            target_mag = torch.abs(target_values)
            mag_mse = torch.mean((pred_mag - target_mag) ** 2).item()
            
            pred_phase = torch.angle(predicted)
            target_phase = torch.angle(target_values)
            phase_diff = torch.angle(torch.exp(1j * (pred_phase - target_phase)))
            phase_mse = torch.mean(phase_diff ** 2).item()
            
            # Correlation coefficient
            pred_flat = predicted.flatten()
            target_flat = target_values.flatten()
            correlation = torch.abs(torch.corrcoef(torch.stack([
                torch.real(pred_flat), torch.real(target_flat)
            ]))[0, 1]).item()
        
        metrics = {
            'complex_mse': complex_mse,
            'magnitude_mse': mag_mse,
            'phase_mse': phase_mse,
            'correlation': correlation,
            'rmse': np.sqrt(complex_mse)
        }
        
        return metrics
