# Implementation Plan: ELO Assessment Framework

## 1. Fix Probability Model (Priority: Highest)

### Replace linear model with generalized logistic
**Location:** `utils.py:run_competition()`

**New Model Formulation:**
```python
def calculate_win_probability(elo_diff, k_upper=0.0023, k_lower=0.0023, offset=0):
    """
    Generalized logistic with separate discrimination for above/below true ability.
    
    P(correct) = 1 / (1 + exp(-k * (elo_diff - offset)))
    where k = k_upper if elo_diff >= offset else k_lower
    
    Parameters:
    - k_upper: discrimination steepness for questions harder than ability
    - k_lower: discrimination steepness for questions easier than ability  
    - offset: systematic bias offset (default 0 = symmetric)
    
    Default k=0.0023 gives ~0.9 probability at +1000 ELO diff (matches current)
    """
```

**Rationale:** 
- Matches current linear bounds at ±1000 ELO
- Allows asymmetry (k_upper ≠ k_lower) - students typically underperform on harder questions
- Easy to understand and fit
- Fast to compute

### Add ProbabilityModel class
**Location:** `utils.py` (new class)

```python
class ProbabilityModel:
    """Manages the probability calculation with configurable parameters."""
    
    def __init__(self, k_upper=0.0023, k_lower=0.0023, offset=0.0):
        self.k_upper = k_upper
        self.k_lower = k_lower
        self.offset = offset
        self.bounds = {
            'k_upper': (0.0001, 0.01),
            'k_lower': (0.0001, 0.01),
            'offset': (-500, 500)
        }
    
    def get_probability(self, elo_diff):
        # Uses k_upper for harder questions, k_lower for easier
        k = self.k_upper if elo_diff >= self.offset else self.k_lower
        # Adjust diff by offset
        adjusted_diff = elo_diff - self.offset
        prob = 1 / (1 + np.exp(-k * adjusted_diff))
        return np.clip(prob, 0.001, 0.999)  # Avoid 0/1 for numerical stability
```

## 2. Add Validation Metrics (Priority: High)

### New utility functions
**Location:** `utils.py` or new `metrics.py`

```python
def calculate_convergence_metrics(simulation_results):
    """
    Calculate metrics across multiple simulation runs.
    
    Args:
        simulation_results: List of (actual_elo, guessed_elo, rounds, converged) tuples
        
    Returns:
        dict with metrics:
        - mean_rounds_to_convergence
        - std_rounds
        - convergence_rate (percentage that converged within 100 rounds)
        - mean_absolute_error (|guessed - true| for converged cases)
        - median_rounds
    """

def compare_models(model_a, model_b, population, n_trials=1000):
    """
    Compare two probability models head-to-head on same population.
    Returns comparison metrics.
    """
```

### Enhanced simulation tracking
**Location:** `elo.py` modifications

Track per-simulation data:
- Initial ELO guess vs final guess
- Convergence path (round-by-round score/variance)
- Convergence success/failure
- Time to convergence

## 3. Real Data Fitting Utility (Priority: High)

### New file: `fit_model.py`

```python
"""
Fit probability model coefficients to real assessment data.

Expected CSV format:
    student_id,student_elo,question_elo,correct
    1,1200,1300,0
    1,1200,1100,1
    ...
"""

def load_data(filepath):
    """Load and validate assessment data from CSV."""
    
def calculate_elo_diff(data_df):
    """Calculate elo_diff = question_elo - student_elo."""
    
def bin_data_by_elo_diff(data_df, bin_width=50):
    """
    Bin data by elo_diff for visualization and verification.
    Returns binned data with mean success rate per bin.
    """

def fit_model_to_data(data_df):
    """
    Fit ProbabilityModel parameters to maximize log-likelihood.
    
    Uses scipy.optimize.minimize to find optimal k_upper, k_lower, offset
    that maximize the log-likelihood of observed outcomes.
    
    Returns fitted ProbabilityModel instance.
    """

def visualize_fit(data_df, fitted_model, output_path=None):
    """
    Create plot comparing empirical data to fitted curve.
    Shows binned empirical rates vs fitted sigmoid.
    """
```

### Fitting approach
Use **maximum likelihood estimation**:
- Objective: maximize Σ [correct * log(P) + (1-correct) * log(1-P)]
- Variables: k_upper, k_lower, offset
- Constraints: bounds as defined in ProbabilityModel
- Optimization: scipy.optimize.minimize with L-BFGS-B

## 4. Configurable Parameters (Priority: Medium)

### Configuration system
**Location:** New `config.py`

```python
"""Configuration management for ELO assessment framework."""

class AssessmentConfig:
    """Centralized configuration for simulation parameters."""
    
    def __init__(self):
        # ELO system parameters
        self.elo_min = 0
        self.elo_max = 10000
        self.elo_level_width = 1000
        
        # Convergence criteria
        self.variance_threshold = 500
        self.elo_error_threshold = 500
        self.max_rounds = 100
        
        # Probability model parameters (default theoretical values)
        self.prob_k_upper = 0.0023
        self.prob_k_lower = 0.0023
        self.prob_offset = 0.0
        
    def load_from_dict(self, config_dict):
        """Override defaults from dict."""
        
    def save_to_file(self, filepath):
        """Save config to JSON."""
        
    @classmethod
    def load_from_file(cls, filepath):
        """Load config from JSON."""
```

### Integration points
- `elo.py`: Accept config object or use global default
- `utils.py`: All functions accept optional config parameter
- `black_box()` in elo.py: Pass config through

## 5. Paper-Ready Visualizations (Priority: Medium)

### New file: `visualizations.py`

```python
"""Paper-ready visualization generation."""

def plot_convergence_distribution(results_by_model, output_path):
    """
    Plot histogram of rounds to convergence for different models.
    Side-by-side comparison showing superiority of improved model.
    """

def plot_probability_curves_comparison(models_dict, output_path):
    """
    Plot probability curves for multiple models on same axes.
    Show linear (current), symmetric sigmoid, asymmetric sigmoid.
    """

def plot_empirical_vs_fitted(raw_data, fitted_model, output_path):
    """
    For paper section showing data-driven model calibration.
    Shows empirical data points with fitted curve overlaid.
    """

def plot_convergence_trajectories(sample_trajectories, output_path):
    """
    Show 10-20 example convergence paths.
    Illustrates how quickly system converges across different true ELOs.
    """

def generate_all_paper_figures(config, simulation_results, output_dir='figures/'):
    """Batch generate all figures for paper."""
```

## 6. Update Simulation Framework (Priority: High)

### Modify `elo.py`
1. Replace current `black_box()` call signature to use ProbabilityModel
2. Store model as parameter passed through optimization
3. Track detailed results for metrics calculation
4. Compare linear vs sigmoid models side-by-side

### Modify `utils.py`
1. Replace `run_competition()` probability calculation
2. Update `simulate_convergence()` to use config
3. Add metrics tracking to simulation functions

## 7. Documentation & Paper Integration (Priority: Medium)

### README updates
- Add section on probability model
- Document how to fit to real data
- Include example outputs

### AGENTS.md updates
- Document new code patterns (ProbabilityModel, Config)
- Update import guidelines

## Implementation Order

1. **Create ProbabilityModel class** with sigmoid (utils.py)
2. **Replace probability calculation** in run_competition()
3. **Test** with existing simulations to verify similar results
4. **Add metrics utilities** (new file or utils.py)
5. **Create fit_model.py** for data-driven calibration
6. **Add config.py** for parameter management
7. **Create visualizations.py** for paper figures
8. **Update elo.py** to use new system end-to-end
9. **Update documentation**
10. **Generate paper figures** and validate metrics

## Expected Outcomes

- **Theoretical demonstration:** Default sigmoid model shows convergence in <5 rounds
- **Data-driven option:** fit_model.py allows calibration to real assessment data
- **Publication-ready:** Comprehensive metrics and visualizations
- **Reproducible:** Config system ensures results are repeatable
- **Extensible:** Easy to add new probability models or metrics
