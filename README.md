# ELO Assessment Framework

A simulation framework for rapid skill diagnosis using an adaptive Elo-inspired assessment framework for personalized learning.

## Overview

This project implements an adaptive assessment framework that dynamically adjusts question difficulty based on learner performance. It assigns Elo ratings to both questions and students, enabling personalized assessment pathways that converge on a student's true skill level in fewer than 5 questions on average.

## Installation

```bash
uv sync
```

## Quick Start

```bash
# Run main simulation
uv run python elo.py

# Run hello script
uv run python hello.py
```

## Probability Model

The framework uses an **asymmetric logistic (sigmoid) function** to model student performance:

```
P(correct) = 1 / (1 + 10^(-steepness * (elo_diff - offset) / 1000))
```

Where:
- `elo_diff` = Question Elo - Student True Elo
- `steepness_upper` = Discrimination for questions harder than ability (default: 1.0)
- `steepness_lower` = Discrimination for questions easier than ability (default: 1.0)  
- `offset` = Systematic bias (default: 0)

With default parameters (steepness=1.0):
- +1000 ELO difference: ~91% probability of correct
- 0 ELO difference: 50% probability
- -1000 ELO difference: ~9% probability

### Asymmetric Model

Real-world data often shows asymmetry - students may underperform on harder questions. You can configure this:

```python
from utils import ProbabilityModel
from config import AssessmentConfig

# Asymmetric model (students struggle more with hard questions)
model = ProbabilityModel(
    steepness_upper=0.8,  # Less discrimination on hard questions
    steepness_lower=1.2,  # Better discrimination on easy questions
    offset=0
)
```

## Configuration

Edit `config.py` or create a custom configuration:

```python
from config import AssessmentConfig

config = AssessmentConfig({
    'variance_threshold': 500,
    'elo_error_threshold': 500,
    'prob_steepness_upper': 1.0,
    'prob_steepness_lower': 1.0,
    'prob_offset': 0.0,
    'entries_per_1000': 12,
})

# Save config
config.save_to_file('my_config.json')
```

## Fitting to Real Data

The framework includes tools to calibrate the probability model to real assessment data:

```python
from fit_model import load_data, fit_model_to_data, evaluate_model_fit, save_fitted_model

# Load your assessment data (CSV format)
df = load_data('assessment_data.csv')

# Fit the model to your data
model = fit_model_to_data(df)

# Evaluate fit quality
metrics = evaluate_model_fit(df, model)
print(f"Accuracy: {metrics['accuracy']:.3f}")
print(f"AIC: {metrics['aic']:.3f}")

# Save for later use
save_fitted_model(model, metrics, 'fitted_model.json')
```

### Expected CSV Format

```csv
student_id,student_elo,question_elo,correct
1,1200,1300,0
1,1200,1100,1
2,1500,1600,1
...
```

## Visualization

Generate paper-ready figures:

```python
from visualizations import (
    plot_probability_curves_comparison,
    plot_convergence_distribution,
    plot_empirical_vs_fitted,
    plot_convergence_trajectories
)

# Compare different probability models
models = {
    'Default': ProbabilityModel(),
    'Asymmetric': ProbabilityModel(steepness_upper=0.8, steepness_lower=1.2)
}
plot_probability_curves_comparison(models, output_path='figures/probability_curves.png')
```

## Project Structure

```
.
├── elo.py              # Main simulation with Bayesian optimization
├── utils.py            # Core ELO logic, Competitor class, probability model
├── config.py           # Configuration management
├── fit_model.py        # Data fitting utilities
├── visualizations.py   # Paper-ready plotting functions
├── pyproject.toml      # Dependencies
└── README.md          # This file
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `variance_threshold` | 500 | Convergence criteria for variance |
| `elo_error_threshold` | 500 | Convergence criteria for ELO error |
| `max_rounds` | 100 | Maximum rounds before giving up |
| `entries_per_1000` | 12 | Number of competitor entries per 1000 ELO |
| `prob_steepness_upper` | 1.0 | Discrimination for hard questions |
| `prob_steepness_lower` | 1.0 | Discrimination for easy questions |
| `prob_offset` | 0.0 | Systematic bias |

## Paper

This framework was developed for the paper:

**"Rapid Skill Diagnosis: An Adaptive Elo-Inspired Assessment Framework for Personalized Learning"**

Abstract: This study addresses the need for a dynamic, efficient assessment framework capable of quickly converging on a learner's competence. Through simulations using Python, we demonstrate that this approach accurately converges on a student's true skill level within an average of fewer than five questions.
