"""
Fit probability model coefficients to real assessment data.

This module provides tools to calibrate the probability model to empirical
data, allowing the ELO assessment framework to match real-world performance
patterns.

Expected CSV format:
    student_id,student_elo,question_elo,correct
    1,1200,1300,0
    1,1200,1100,1
    ...

The 'correct' column should be 1 for correct answers, 0 for incorrect.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Tuple, Optional
import json
from utils import ProbabilityModel


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load and validate assessment data from CSV.
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        DataFrame with columns: student_id, student_elo, question_elo, correct
        
    Raises:
        ValueError: If required columns are missing
    """
    df = pd.read_csv(filepath)
    
    required_cols = ['student_id', 'student_elo', 'question_elo', 'correct']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Validate data types
    if not pd.api.types.is_numeric_dtype(df['correct']):
        raise ValueError("'correct' column must be numeric (0 or 1)")
    
    if not set(df['correct'].unique()).issubset({0, 1}):
        raise ValueError("'correct' column must contain only 0 or 1")
    
    return df


def calculate_elo_diff(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate elo_diff = question_elo - student_elo for each response.
    
    Positive elo_diff means the question was harder than the student's ability.
    """
    df = df.copy()
    df['elo_diff'] = df['question_elo'] - df['student_elo']
    return df


def bin_data_by_elo_diff(df: pd.DataFrame, bin_width: int = 50) -> pd.DataFrame:
    """
    Bin data by elo_diff for visualization and verification.
    
    Args:
        df: DataFrame with elo_diff column
        bin_width: Width of each bin in ELO units
        
    Returns:
        DataFrame with binned statistics: bin_center, mean_success_rate, count
    """
    df = df.copy()
    
    # Create bins
    min_diff = df['elo_diff'].min()
    max_diff = df['elo_diff'].max()
    bins = np.arange(min_diff - bin_width/2, max_diff + bin_width, bin_width)
    
    df['bin'] = pd.cut(df['elo_diff'], bins=bins)
    
    # Calculate statistics per bin
    binned = df.groupby('bin').agg({
        'correct': ['mean', 'count'],
        'elo_diff': 'mean'
    }).reset_index()
    
    binned.columns = ['bin', 'mean_success_rate', 'count', 'bin_center']
    
    # Filter out bins with very few observations
    binned = binned[binned['count'] >= 5]
    
    return binned


def calculate_log_likelihood(params, elo_diffs, outcomes) -> float:
    """
    Calculate negative log-likelihood for given model parameters.
    
    Args:
        params: [steepness_upper, steepness_lower, offset]
        elo_diffs: Array of ELO differences (question - student)
        outcomes: Array of 0/1 outcomes
        
    Returns:
        Negative log-likelihood (to be minimized)
    """
    steepness_upper = params[0]
    steepness_lower = params[1]
    offset = params[2]
    
    # Calculate probabilities
    adjusted_diffs = elo_diffs - offset
    
    # Use different steepness for positive vs negative diffs
    probs = np.where(
        adjusted_diffs >= 0,
        1.0 / (1.0 + 10.0 ** (-steepness_upper * adjusted_diffs / 1000.0)),
        1.0 / (1.0 + 10.0 ** (-steepness_lower * adjusted_diffs / 1000.0))
    )
    
    # Clamp for numerical stability
    probs = np.clip(probs, 0.001, 0.999)
    
    # Calculate log-likelihood
    log_likelihood = np.sum(
        outcomes * np.log(probs) + (1 - outcomes) * np.log(1 - probs)
    )
    
    return -log_likelihood  # Return negative for minimization


def fit_model_to_data(df: pd.DataFrame, 
                      initial_guess: Optional[list] = None) -> ProbabilityModel:
    """
    Fit ProbabilityModel parameters to maximize log-likelihood of data.
    
    Uses scipy.optimize.minimize with L-BFGS-B algorithm to find optimal
    parameters that best explain the observed outcomes.
    
    Args:
        df: DataFrame with elo_diff and correct columns
        initial_guess: Optional [steepness_upper, steepness_lower, offset]
                      Defaults to [1.0, 1.0, 0.0] (symmetric model)
                      
    Returns:
        Fitted ProbabilityModel instance
        
    Raises:
        ValueError: If optimization fails
    """
    if 'elo_diff' not in df.columns:
        df = calculate_elo_diff(df)
    
    elo_diffs = df['elo_diff'].values
    outcomes = df['correct'].values
    
    if initial_guess is None:
        initial_guess = [1.0, 1.0, 0.0]
    
    # Set bounds for parameters
    bounds = [(0.1, 5.0), (0.1, 5.0), (-500, 500)]
    
    # Optimize
    result = minimize(
        calculate_log_likelihood,
        initial_guess,
        args=(elo_diffs, outcomes),
        method='L-BFGS-B',
        bounds=bounds
    )
    
    if not result.success:
        raise ValueError(f"Optimization failed: {result.message}")
    
    steepness_upper, steepness_lower, offset = result.x
    
    model = ProbabilityModel(
        steepness_upper=steepness_upper,
        steepness_lower=steepness_lower,
        offset=offset
    )
    
    return model


def evaluate_model_fit(df: pd.DataFrame, model: ProbabilityModel) -> dict:
    """
    Evaluate how well the model fits the data.
    
    Returns metrics:
    - log_likelihood: Final log-likelihood
    - aic: Akaike Information Criterion
    - bic: Bayesian Information Criterion
    - accuracy: Classification accuracy
    - mean_abs_error: Mean absolute error between predicted and actual
    """
    if 'elo_diff' not in df.columns:
        df = calculate_elo_diff(df)
    
    elo_diffs = df['elo_diff'].values
    outcomes = df['correct'].values
    
    # Calculate predicted probabilities
    predicted_probs = np.array([model.get_probability(d) for d in elo_diffs])
    predicted_binary = (predicted_probs >= 0.5).astype(int)
    
    # Calculate metrics
    n = len(outcomes)
    k = 3  # Number of parameters
    
    log_likelihood = -calculate_log_likelihood(
        [model.steepness_upper, model.steepness_lower, model.offset],
        elo_diffs, outcomes
    )
    
    aic = 2 * k - 2 * log_likelihood
    bic = np.log(n) * k - 2 * log_likelihood
    accuracy = np.mean(predicted_binary == outcomes)
    mean_abs_error = np.mean(np.abs(predicted_probs - outcomes))
    
    return {
        'log_likelihood': log_likelihood,
        'aic': aic,
        'bic': bic,
        'accuracy': accuracy,
        'mean_absolute_error': mean_abs_error,
        'n_observations': n
    }


def save_fitted_model(model: ProbabilityModel, metrics: dict, filepath: str):
    """
    Save fitted model and metrics to JSON file.
    
    Args:
        model: Fitted ProbabilityModel
        metrics: Dictionary of fit metrics
        filepath: Output JSON file path
    """
    data = {
        'model_parameters': model.to_dict(),
        'fit_metrics': metrics
    }
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_fitted_model(filepath: str) -> Tuple[ProbabilityModel, dict]:
    """
    Load fitted model and metrics from JSON file.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Tuple of (ProbabilityModel, metrics_dict)
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    model = ProbabilityModel.from_dict(data['model_parameters'])
    metrics = data['fit_metrics']
    
    return model, metrics


if __name__ == '__main__':
    # Example usage demonstration
    print("Probability Model Fitting Tool")
    print("=" * 50)
    print()
    print("This module provides tools to fit the ELO probability model to real data.")
    print()
    print("Example workflow:")
    print("1. Load data: df = load_data('assessment_data.csv')")
    print("2. Fit model: model = fit_model_to_data(df)")
    print("3. Evaluate: metrics = evaluate_model_fit(df, model)")
    print("4. Save: save_fitted_model(model, metrics, 'fitted_model.json')")
    print()
    print("Default theoretical model parameters:")
    default_model = ProbabilityModel()
    print(f"  steepness_upper: {default_model.steepness_upper}")
    print(f"  steepness_lower: {default_model.steepness_lower}")
    print(f"  offset: {default_model.offset}")
    print()
    print("Example probabilities:")
    for diff in [-1000, -500, 0, 500, 1000]:
        prob = default_model.get_probability(diff)
        print(f"  ELO diff {diff:+4d}: P(correct) = {prob:.3f}")
