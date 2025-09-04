# Contributing to jGS

Thank you for your interest in contributing to jGS (Complex-valued Gaussian Splatting for RF Signal Processing)! This document provides guidelines for contributing to the project.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Setup](#development-setup)
4. [Contributing Guidelines](#contributing-guidelines)
5. [Pull Request Process](#pull-request-process)
6. [Coding Standards](#coding-standards)
7. [Testing](#testing)
8. [Documentation](#documentation)

## Code of Conduct

This project adheres to a code of conduct that we expect all contributors to follow. Please be respectful and constructive in all interactions.

### Our Standards

- Use welcoming and inclusive language
- Be respectful of differing viewpoints and experiences
- Gracefully accept constructive criticism
- Focus on what is best for the community
- Show empathy towards other community members

## Getting Started

### Ways to Contribute

- **Bug Reports**: Report bugs or issues you encounter
- **Feature Requests**: Suggest new features or improvements
- **Code Contributions**: Submit bug fixes, new features, or optimizations
- **Documentation**: Improve documentation, tutorials, or examples
- **Testing**: Add test cases or improve test coverage
- **Examples**: Create new examples or improve existing ones

### Before You Start

1. Check existing [issues](https://github.com/tagsysx/jGS/issues) and [pull requests](https://github.com/tagsysx/jGS/pulls)
2. For major changes, open an issue first to discuss the proposed changes
3. Fork the repository and create a feature branch

## Development Setup

### Prerequisites

- Python 3.8+
- Git
- CUDA-compatible GPU (recommended)

### Setup Instructions

1. **Fork and Clone**
   ```bash
   git clone https://github.com/YOUR_USERNAME/jGS.git
   cd jGS
   ```

2. **Create Development Environment**
   ```bash
   conda create -n jgs-dev python=3.9
   conda activate jgs-dev
   ```

3. **Install Dependencies**
   ```bash
   # Install requirements
   pip install -r requirements.txt
   
   # Install development dependencies
   pip install -e ".[dev]"
   ```

4. **Verify Setup**
   ```bash
   python -c "import jgs; print('Setup successful!')"
   pytest tests/ -v
   ```

### Development Tools

We use the following tools for development:

- **Code Formatting**: `black`
- **Linting**: `flake8`
- **Type Checking**: `mypy`
- **Testing**: `pytest`
- **Documentation**: `sphinx`

Install pre-commit hooks:
```bash
pre-commit install
```

## Contributing Guidelines

### Issue Reporting

When reporting bugs, please include:

- **Environment**: OS, Python version, PyTorch version, CUDA version
- **Reproduction Steps**: Minimal code to reproduce the issue
- **Expected Behavior**: What you expected to happen
- **Actual Behavior**: What actually happened
- **Error Messages**: Full error traceback if applicable

**Bug Report Template:**
```markdown
## Bug Description
Brief description of the bug.

## Environment
- OS: [e.g., Ubuntu 20.04]
- Python: [e.g., 3.9.7]
- PyTorch: [e.g., 1.12.0]
- CUDA: [e.g., 11.6]

## Reproduction Steps
1. Step 1
2. Step 2
3. ...

## Expected Behavior
What should happen.

## Actual Behavior
What actually happens.

## Error Message
```
Full error traceback here
```
```

### Feature Requests

For feature requests, please include:

- **Use Case**: Why is this feature needed?
- **Proposed Solution**: How should it work?
- **Alternatives**: Other approaches considered
- **Implementation Ideas**: Technical details if you have them

### Code Contributions

#### Branch Naming

Use descriptive branch names:
- `feature/add-antenna-array-support`
- `bugfix/fix-cuda-memory-leak`
- `docs/improve-api-documentation`
- `refactor/optimize-rendering-pipeline`

#### Commit Messages

Follow conventional commit format:
```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

Examples:
```
feat(core): add support for frequency-dependent rendering
fix(rf): resolve phase unwrapping issue in signal processor
docs(api): add examples to ComplexGaussianSplatter class
```

## Pull Request Process

### Before Submitting

1. **Code Quality**
   ```bash
   # Format code
   black jgs/ tests/ examples/
   
   # Check linting
   flake8 jgs/ tests/ examples/
   
   # Type checking
   mypy jgs/
   ```

2. **Run Tests**
   ```bash
   pytest tests/ -v --cov=jgs
   ```

3. **Update Documentation**
   - Add docstrings to new functions/classes
   - Update API documentation if needed
   - Add examples for new features

### Pull Request Template

```markdown
## Description
Brief description of changes.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] Added new tests for new functionality
- [ ] Updated existing tests if needed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or clearly documented)

## Related Issues
Closes #[issue_number]
```

### Review Process

1. **Automated Checks**: CI/CD pipeline runs tests and checks
2. **Code Review**: Maintainers review code for quality and correctness
3. **Discussion**: Address feedback and make requested changes
4. **Approval**: Once approved, changes are merged

## Coding Standards

### Python Style

Follow PEP 8 with these specifics:

- **Line Length**: 88 characters (Black default)
- **Imports**: Use absolute imports, group by standard/third-party/local
- **Naming**: 
  - Functions/variables: `snake_case`
  - Classes: `PascalCase`
  - Constants: `UPPER_CASE`

### Docstring Format

Use Google-style docstrings:

```python
def complex_function(
    param1: torch.Tensor,
    param2: Optional[float] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Brief description of the function.
    
    Longer description if needed, explaining the purpose,
    algorithm, or important details.
    
    Args:
        param1: Description of param1, including shape if tensor
        param2: Description of param2, including default behavior
        
    Returns:
        Tuple containing:
            - First element description
            - Second element description
            
    Raises:
        ValueError: When param1 has wrong shape
        RuntimeError: When computation fails
        
    Example:
        >>> result1, result2 = complex_function(torch.randn(10, 3))
        >>> print(result1.shape)
        torch.Size([10])
    """
    # Implementation here
    pass
```

### Type Hints

Use type hints for all public functions:

```python
from typing import Optional, Union, Tuple, List, Dict, Any
import torch
import numpy as np

def process_field(
    positions: Union[np.ndarray, torch.Tensor],
    complex_values: torch.Tensor,
    frequency: Optional[float] = None
) -> Dict[str, torch.Tensor]:
    """Process complex field data."""
    pass
```

### Error Handling

- Use specific exception types
- Provide helpful error messages
- Log errors appropriately
- Don't use bare `except:` clauses

```python
def validate_input(positions: torch.Tensor) -> None:
    """Validate input tensor."""
    if positions.dim() != 2:
        raise ValueError(
            f"Expected 2D tensor, got {positions.dim()}D tensor with shape {positions.shape}"
        )
    
    if positions.shape[1] != 3:
        raise ValueError(
            f"Expected 3D positions (N, 3), got shape {positions.shape}"
        )
```

## Testing

### Test Structure

```
tests/
├── test_core/
│   ├── test_gaussian_splatter.py
│   ├── test_primitives.py
│   └── test_renderer.py
├── test_rf/
│   ├── test_antenna_patterns.py
│   └── test_signal_processor.py
├── test_utils/
│   └── test_complex_math.py
└── conftest.py  # Shared fixtures
```

### Writing Tests

Use pytest with descriptive test names:

```python
import pytest
import torch
import jgs

class TestComplexGaussianSplatter:
    """Test ComplexGaussianSplatter class."""
    
    def test_initialization_with_valid_inputs(self):
        """Test that model initializes correctly with valid inputs."""
        positions = torch.randn(10, 3)
        complex_values = torch.randn(10, dtype=torch.complex64)
        
        model = jgs.ComplexGaussianSplatter(
            positions=positions,
            complex_values=complex_values
        )
        
        assert model.positions.shape == (10, 3)
        assert model.complex_values.shape == (10,)
    
    def test_render_returns_correct_shape(self):
        """Test that render method returns correct output shape."""
        model = jgs.ComplexGaussianSplatter(
            positions=torch.randn(5, 3),
            complex_values=torch.randn(5, dtype=torch.complex64)
        )
        
        query_points = torch.randn(20, 3)
        result = model.render(query_points)
        
        assert result.shape == (20,)
        assert result.dtype == torch.complex64
    
    def test_invalid_input_raises_error(self):
        """Test that invalid inputs raise appropriate errors."""
        with pytest.raises(ValueError, match="Expected 2D tensor"):
            jgs.ComplexGaussianSplatter(
                positions=torch.randn(10),  # Wrong shape
                complex_values=torch.randn(10, dtype=torch.complex64)
            )
```

### Test Coverage

Aim for high test coverage:
- **Unit Tests**: Test individual functions/methods
- **Integration Tests**: Test component interactions
- **End-to-End Tests**: Test complete workflows

Run coverage analysis:
```bash
pytest --cov=jgs --cov-report=html
```

## Documentation

### API Documentation

- All public functions/classes must have docstrings
- Include examples in docstrings when helpful
- Document all parameters and return values
- Mention important exceptions

### Tutorials and Guides

When adding new features:
- Update relevant tutorials
- Add new tutorials for major features
- Include practical examples
- Explain the theory when needed

### Building Documentation

```bash
cd docs/
make html
```

View documentation:
```bash
open _build/html/index.html
```

## Release Process

### Version Numbering

We use semantic versioning (SemVer):
- **Major** (X.0.0): Breaking changes
- **Minor** (0.X.0): New features, backward compatible
- **Patch** (0.0.X): Bug fixes, backward compatible

### Release Checklist

1. Update version in `setup.py` and `jgs/__init__.py`
2. Update `CHANGELOG.md`
3. Run full test suite
4. Update documentation
5. Create release PR
6. Tag release after merge
7. Create GitHub release with notes

## Getting Help

### Communication Channels

- **GitHub Issues**: Bug reports, feature requests
- **GitHub Discussions**: General questions, ideas
- **Email**: Direct contact with maintainers

### Resources

- [Documentation](docs/)
- [Examples](examples/)
- [API Reference](docs/api/)
- [Tutorials](docs/tutorials/)

## Recognition

Contributors will be recognized in:
- `AUTHORS.md` file
- Release notes
- Documentation acknowledgments

Thank you for contributing to jGS! 🚀

---

**Questions?** Open an issue or start a discussion on GitHub.
