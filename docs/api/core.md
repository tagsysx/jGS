# Core Module API Reference

The `jgs.core` module contains the fundamental algorithms and data structures for complex-valued Gaussian Splatting.

## Classes

### ComplexGaussianSplatter

Main class for complex-valued Gaussian Splatting models.

```python
class ComplexGaussianSplatter(nn.Module)
```

#### Constructor

```python
def __init__(
    self,
    positions: Union[np.ndarray, torch.Tensor],
    complex_values: Union[np.ndarray, torch.Tensor],
    scales: Optional[Union[np.ndarray, torch.Tensor]] = None,
    rotations: Optional[Union[np.ndarray, torch.Tensor]] = None,
    device: str = 'cuda',
    dtype: torch.dtype = torch.complex64
)
```

**Parameters:**
- `positions` (array-like): 3D positions of Gaussian primitives (N, 3)
- `complex_values` (array-like): Complex amplitudes at each position (N,) or (N, C)
- `scales` (array-like, optional): Scaling factors for each Gaussian (N, 3)
- `rotations` (array-like, optional): Rotation quaternions (N, 4)
- `device` (str): Device to run computations on ('cuda' or 'cpu')
- `dtype` (torch.dtype): Data type for complex computations

**Example:**
```python
import jgs
import torch

# Create Gaussian Splatting model
positions = torch.randn(100, 3)
complex_values = torch.randn(100, dtype=torch.complex64)

model = jgs.ComplexGaussianSplatter(
    positions=positions,
    complex_values=complex_values,
    device='cuda'
)
```

#### Methods

##### render()

Render complex RF field at specified query points.

```python
def render(
    self, 
    query_points: Union[np.ndarray, torch.Tensor],
    frequency: Optional[float] = None
) -> torch.Tensor
```

**Parameters:**
- `query_points` (array-like): 3D points to evaluate field at (M, 3)
- `frequency` (float, optional): Frequency for phase calculations (Hz)

**Returns:**
- `torch.Tensor`: Complex field values at query points (M,)

**Example:**
```python
# Render field at specific points
query_points = torch.tensor([[0, 0, 0], [1, 1, 1]], dtype=torch.float32)
field = model.render(query_points, frequency=2.4e9)
print(f"Field values: {field}")
```

##### add_primitive()

Add a new Gaussian primitive to the model.

```python
def add_primitive(
    self,
    position: torch.Tensor,
    complex_value: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    rotation: Optional[torch.Tensor] = None
)
```

**Parameters:**
- `position` (torch.Tensor): 3D position (3,)
- `complex_value` (torch.Tensor): Complex amplitude
- `scale` (torch.Tensor, optional): Scaling factors (3,)
- `rotation` (torch.Tensor, optional): Rotation quaternion (4,)

##### remove_primitive()

Remove a Gaussian primitive by index.

```python
def remove_primitive(self, index: int)
```

##### get_field_magnitude()

Get magnitude of the complex field at query points.

```python
def get_field_magnitude(
    self, 
    query_points: Union[np.ndarray, torch.Tensor]
) -> torch.Tensor
```

##### get_field_phase()

Get phase of the complex field at query points.

```python
def get_field_phase(
    self, 
    query_points: Union[np.ndarray, torch.Tensor]
) -> torch.Tensor
```

##### save_state() / load_state()

Save and load model state.

```python
def save_state(self, filepath: str)
def load_state(self, filepath: str)
```

#### Properties

- `positions`: Current positions of Gaussian primitives
- `complex_values`: Current complex values of Gaussian primitives
- `scales`: Current scales of Gaussian primitives
- `rotations`: Current rotations of Gaussian primitives

---

### ComplexGaussianPrimitive

Individual complex-valued Gaussian primitive.

```python
class ComplexGaussianPrimitive
```

#### Constructor

```python
def __init__(
    self,
    position: torch.Tensor,
    complex_value: torch.Tensor,
    scale: torch.Tensor,
    rotation: torch.Tensor,
    opacity: float = 1.0
)
```

#### Methods

##### evaluate()

Evaluate the Gaussian primitive at query points.

```python
def evaluate(
    self, 
    query_points: torch.Tensor,
    frequency: Optional[float] = None
) -> torch.Tensor
```

##### evaluate_gradient()

Evaluate primitive and its spatial gradient.

```python
def evaluate_gradient(
    self, 
    query_points: torch.Tensor,
    frequency: Optional[float] = None
) -> Tuple[torch.Tensor, torch.Tensor]
```

##### get_bounding_box()

Get axis-aligned bounding box.

```python
def get_bounding_box(
    self, 
    sigma_threshold: float = 3.0
) -> Tuple[torch.Tensor, torch.Tensor]
```

---

### ComplexRenderer

Renderer for complex-valued RF fields.

```python
class ComplexRenderer(nn.Module)
```

#### Constructor

```python
def __init__(
    self,
    device: str = 'cuda',
    dtype: torch.dtype = torch.complex64,
    batch_size: int = 10000
)
```

#### Methods

##### render_field()

Render complex RF field at query points.

```python
def render_field(
    self,
    primitives: List,
    query_points: torch.Tensor,
    frequency: Optional[float] = None,
    use_batching: bool = True
) -> torch.Tensor
```

##### render_magnitude()

Render field magnitude.

```python
def render_magnitude(
    self,
    primitives: List,
    query_points: torch.Tensor,
    frequency: Optional[float] = None
) -> torch.Tensor
```

##### render_phase()

Render field phase.

```python
def render_phase(
    self,
    primitives: List,
    query_points: torch.Tensor,
    frequency: Optional[float] = None
) -> torch.Tensor
```

##### render_power()

Render field power (|E|²).

```python
def render_power(
    self,
    primitives: List,
    query_points: torch.Tensor,
    frequency: Optional[float] = None
) -> torch.Tensor
```

##### render_grid()

Render field on a regular 3D grid.

```python
def render_grid(
    self,
    primitives: List,
    bounds: Tuple[torch.Tensor, torch.Tensor],
    resolution: Union[int, Tuple[int, int, int]],
    frequency: Optional[float] = None
) -> Tuple[torch.Tensor, torch.Tensor]
```

##### render_slice()

Render field on a 2D slice through 3D space.

```python
def render_slice(
    self,
    primitives: List,
    plane_normal: torch.Tensor,
    plane_point: torch.Tensor,
    bounds_2d: Tuple[torch.Tensor, torch.Tensor],
    resolution: Union[int, Tuple[int, int]],
    frequency: Optional[float] = None
) -> Tuple[torch.Tensor, torch.Tensor]
```

---

### ComplexOptimizer

Optimizer for complex-valued Gaussian Splatting parameters.

```python
class ComplexOptimizer
```

#### Constructor

```python
def __init__(
    self,
    model: nn.Module,
    learning_rate: float = 1e-3,
    optimizer_type: str = 'adam',
    scheduler_type: Optional[str] = 'plateau',
    device: str = 'cuda'
)
```

#### Methods

##### fit()

Fit the model to target field measurements.

```python
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
) -> Dict[str, Any]
```

**Parameters:**
- `query_points`: Points where field is measured (N, 3)
- `target_values`: Target complex field values (N,)
- `validation_points`: Optional validation points (M, 3)
- `validation_values`: Optional validation values (M,)
- `num_epochs`: Number of training epochs
- `batch_size`: Batch size for training
- `loss_function`: Loss function ('complex_mse', 'magnitude_phase')
- `frequency`: Optional frequency for rendering
- `verbose`: Whether to print training progress

**Returns:**
- `Dict`: Training history with losses and metrics

##### evaluate()

Evaluate model performance on test data.

```python
def evaluate(
    self,
    query_points: torch.Tensor,
    target_values: torch.Tensor,
    frequency: Optional[float] = None
) -> Dict[str, float]
```

##### Loss Functions

Built-in loss functions:

```python
def complex_mse_loss(
    self,
    predicted: torch.Tensor,
    target: torch.Tensor,
    weights: Optional[torch.Tensor] = None
) -> torch.Tensor

def magnitude_phase_loss(
    self,
    predicted: torch.Tensor,
    target: torch.Tensor,
    magnitude_weight: float = 1.0,
    phase_weight: float = 1.0,
    weights: Optional[torch.Tensor] = None
) -> torch.Tensor

def regularization_loss(
    self,
    l1_weight: float = 0.0,
    l2_weight: float = 0.0,
    sparsity_weight: float = 0.0
) -> torch.Tensor
```

## Usage Examples

### Basic Field Reconstruction

```python
import jgs
import torch
from jgs.core.optimization import ComplexOptimizer

# Generate synthetic measurement data
positions = torch.randn(200, 3)
target_field = torch.randn(200, dtype=torch.complex64)

# Create model
model = jgs.ComplexGaussianSplatter(
    positions=torch.randn(50, 3),
    complex_values=torch.randn(50, dtype=torch.complex64)
)

# Optimize
optimizer = ComplexOptimizer(model)
history = optimizer.fit(positions, target_field, num_epochs=1000)

# Evaluate at new points
test_points = torch.randn(50, 3)
predicted = model.render(test_points)
```

### Grid Rendering

```python
from jgs.core.renderer import ComplexRenderer

renderer = ComplexRenderer()

# Define grid bounds
min_bounds = torch.tensor([-2.0, -2.0, -1.0])
max_bounds = torch.tensor([2.0, 2.0, 1.0])
bounds = (min_bounds, max_bounds)

# Render on grid
grid_points, field_grid = renderer.render_grid(
    primitives=model.primitives,
    bounds=bounds,
    resolution=(64, 64, 32),
    frequency=2.4e9
)

print(f"Grid shape: {field_grid.shape}")
```

---

**See Also:**
- [RF Module API](rf.md)
- [Visualization Module API](visualization.md)
- [Basic Concepts Tutorial](../tutorials/basic_concepts.md)
