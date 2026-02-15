# Agent Guidelines for elo-tests

## Project Overview
ELO rating simulation project using Bayesian optimization to find optimal parameters for user ELO convergence. Uses uv for Python package management.

## Build/Run Commands

```bash
# Install dependencies
uv sync

# Run main simulation
uv run python elo.py

# Run hello script
uv run python hello.py

# Run any Python script
uv run python <script-name>
```

## Testing
**No test suite currently configured.** To add tests:
- Create `tests/` directory
- Use pytest: `uv add --dev pytest`
- Run single test: `uv run pytest tests/test_specific.py::test_function -v`
- Run all tests: `uv run pytest`

## Code Style Guidelines

### Python Version
- **Python 3.12+** (specified in `.python-version` and `pyproject.toml`)

### Imports
- **Avoid wildcard imports** - explicit imports preferred (e.g., `from utils import build_population` instead of `from utils import *`)
- Group imports: stdlib → third-party → local
- Use `Union` from `typing` for type hints (e.g., `Union[int, float]`)

### Naming Conventions
- **Functions/variables**: `snake_case`
- **Constants**: `UPPER_CASE` (e.g., `WINNER`, `LOSER`, `DRAW`)
- **Classes**: `PascalCase` (e.g., `Competitor`)
- **Private**: prefix with underscore (e.g., `_helper_function`)

### Type Hints
- Use type hints for function parameters and return values where practical
- Example: `def update(competitor_list, variance_decrease_pc: float = 0.1) -> list:`

### Formatting
- **Indentation**: 4 spaces
- **Line length**: 100 characters (flexible, but avoid excessive length)
- **Quotes**: Single quotes preferred, but consistency within file matters more

### Error Handling
- Raise `ValueError` for invalid inputs with descriptive messages
- Example: `raise ValueError("At least one competitor is required")`
- Use try/except blocks for operations that may fail (e.g., matplotlib rendering)

### Documentation
- Add docstrings for complex functions explaining:
  - What the function does
  - Args with types
  - Return values
- Keep inline comments minimal; explain "why" not "what"

### Project Structure
```
.
├── elo.py              # Main simulation with Bayesian optimization
├── utils.py            # Core ELO logic, Competitor class, ProbabilityModel
├── config.py           # Configuration management (AssessmentConfig)
├── fit_model.py        # Data fitting utilities (fit_model_to_data)
├── visualizations.py    # Paper-ready plotting functions
├── hello.py            # Simple entry point
├── pyproject.toml      # Dependencies and project config
├── uv.lock            # Locked dependency versions
├── README.md          # Project documentation
└── AGENTS.md          # Agent guidelines (this file)
```

## Dependencies
Managed via `pyproject.toml`:
- `bayesian-optimization>=3.1.0`
- `matplotlib>=3.10.7`
- `numpy>=2.3.4`
- `scipy>=1.14.0`
- `pandas>=2.2.0`

Add new dependencies: `uv add <package>`
Add dev dependencies: `uv add --dev <package>`

## Key Implementation Notes

### ELO System
- Users/competitors have `score` (ELO rating) and `variance` (uncertainty)
- Convergence criteria: variance ≤ 500 AND |guessed - true| ≤ 500
- Variance decays after each competition via `variance_decrease_pc`

### Probability Model
- Uses asymmetric sigmoid: `P(correct) = 1 / (1 + 10^(-steepness * (elo_diff - offset) / 1000))`
- Configurable via `ProbabilityModel` class in utils.py
- Default steepness=1.0 gives ~91% at +1000 ELO difference
- Supports asymmetric discrimination (steepness_upper vs steepness_lower)
- Can be fitted to real data using `fit_model.py`

### Bayesian Optimization
- Parameters optimized: `starting_variance`, `variance_decay`
- Uses `bayes_opt.BayesianOptimization` with UCB acquisition
- Score is negative rounds to convergence (lower is better → maximize negative)

### Configuration
- Use `AssessmentConfig` class in config.py for parameter management
- Configurable: variance threshold, ELO error threshold, probability model params
- Supports save/load to JSON

### Visualization
- Matplotlib with 3D plots and surface interpolations
- Uses `TkAgg` backend for interactive plots
- May fail in headless environments; handle gracefully

## Common Patterns

```python
# Function signature template
def function_name(param: type, optional: type = default) -> return_type:
    """Brief description.
    
    Args:
        param: Description
        optional: Description
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When input is invalid
    """
    pass
```

## Check Before Committing
- [ ] No wildcard imports added
- [ ] Type hints present for new public functions
- [ ] No hardcoded secrets or paths
- [ ] Functions have docstrings if complex logic
- [ ] Code runs: `uv run python elo.py` (test with 'v' and 'g' inputs)

## Notes for AI Agents
- This is a simulation/math project, not a production service
- Focus on correctness of ELO calculations and optimization logic
- Visualization code is experimental; errors should not crash the simulation
- No existing tests; manual verification via running `elo.py` is current validation method
