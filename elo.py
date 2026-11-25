from utils import *
# https://bayesian-optimization.github.io/BayesianOptimization/2.0.0/
from bayes_opt import BayesianOptimization
from bayes_opt import acquisition
import time
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from config import AssessmentConfig

# Add at the top of the file, after imports
parameter_history = {}  # Global dict to store parameter combinations and their results

# Default configuration - uses new sigmoid probability model
config = AssessmentConfig()

def discretize_params(params):
    """Convert parameters to discrete values"""
    return {
        'starting_variance': round(params['starting_variance'] / 10) * 10,
        'variance_decay': round(params['variance_decay'], 4),
    }

def black_box(user_actual_placement, entries_per_1000, wants_guessed_starting_elos,
              starting_variance, variance_decay:int, prob_model=None):
    """
    Run simulation for a single user with given parameters.
    
    Args:
        user_actual_placement: True ELO of the user
        entries_per_1000: Number of competitor entries per 1000 ELO range
        wants_guessed_starting_elos: Whether to use guessed starting ELO
        starting_variance: Initial variance
        variance_decay: Variance decay factor
        prob_model: ProbabilityModel instance (uses default if None)
    
    Returns:
        Negative rounds to convergence (lower is better for optimization)
    """
    if prob_model is None:
        prob_model = config.get_probability_model()
    
    rounds = 0
    true_elo = user_actual_placement
    user = initialialize_user(true_elo, starting_variance, wants_guessed_starting_elos)
    # Ensure we have enough competitors for the maximum number of rounds (100)
    available_population = build_population(max(entries_per_1000, 10))  # At least 100 competitors
    last_match_won = None
    
    # Track when we first enter the correct band
    first_correct_entry = None
    
    while (user.variance > 500 or abs(user.score - true_elo) > 500) and rounds < 100:
        rounds += 1
        
        # Calculate target based on last match
        target_elo = user.score + (user.variance if last_match_won else -user.variance)
        competitor_elo = find_closest_competitor(available_population, target_elo)
        
        # Run competition and update user with probability model
        _, user, last_match_won = run_competition(
            user, true_elo, competitor_elo, variance_decay, prob_model
        )
        
        if first_correct_entry and abs(user.score - true_elo) > 500:
            first_correct_entry = None
        elif first_correct_entry is None and abs(user.score - true_elo) <= 500:
            first_correct_entry = rounds
    
    # If we converged (variance <= 500) and stayed in band, return first entry point
    # Otherwise return the total rounds taken (negative as before)
    if user.variance <= 500 and first_correct_entry is not None:
        # we reward early entry
        return -first_correct_entry
    return -rounds

if __name__ == '__main__':
    pref = input('Type "v" if you want example visualizations, type "g" if you want guessed starting elos')
    # check contains v or g, might be both
    wants_example_visualizations = 'v' in pref
    wants_guessed_starting_elos = 'g' in pref
    
    # Print probability model being used
    prob_model = config.get_probability_model()
    print(f"Using probability model:")
    print(f"  steepness_upper: {prob_model.steepness_upper}")
    print(f"  steepness_lower: {prob_model.steepness_lower}")
    print(f"  offset: {prob_model.offset}")
    
    # init params
    params_gbm ={
        'starting_variance':(100, 4000),
        'variance_decay':(0.001, 0.13),
    }
    entries_per_1000 = 12
    acq = acquisition.UpperConfidenceBound(kappa=1.3)
    optimizer = BayesianOptimization(
        f=None,
        acquisition_function=acq,
        pbounds=params_gbm,
        verbose=3,
        random_state=None,
        allow_duplicate_points=True  # Allow testing duplicate parameter combinations
    )

    # Add global best score tracking
    global_best_score = -100
    
    # Run Bayesian Optimization
    start = time.time()
    user_starting_placements = build_population(entries_per_1000=2)
    for i in range(700):
        next_point = optimizer.suggest()
        
        # Discretize parameters
        discrete_point = discretize_params(next_point)
        
        # Calculate new score
        scores = []
        for user_actual_placement in user_starting_placements:
            scores.append(black_box(user_actual_placement, entries_per_1000, 
                                    wants_guessed_starting_elos, **discrete_point,
                                    prob_model=prob_model))
        avg_score = sum(scores)/len(scores)
        # Register with optimizer using discrete parameters
        optimizer.register(params=discrete_point, target=avg_score)
        
        # Output current state
        output = f"{avg_score}: " + ", ".join([f"{label}:{value:.4f}" for label,value in discrete_point.items()])
        print(output)

        if i % 50 == 0 and wants_example_visualizations:
            simulate_convergence(discrete_point, 1250, wants_guessed_starting_elos, prob_model=prob_model)
        
        # Check if we have a new best score and generate plots
        if avg_score > global_best_score:
            global_best_score = avg_score
            if wants_example_visualizations:
                plot_optimization_surfaces(optimizer, global_best_score)
                plot_combined_visualization(optimizer)

    print(f"Time taken: {time.time() - start}")
    print(optimizer.max)
    print('rerunning best params to verify')
    best_params = optimizer.max['params']
    scores = []
    user_starting_placements = build_population(entries_per_1000=8)
    for user_actual_placement in user_starting_placements:
        scores.append(black_box(user_actual_placement, entries_per_1000, wants_guessed_starting_elos, 
                                **best_params, prob_model=prob_model))
    avg_score = sum(scores) / len(scores)
    print(f"Verified score: {avg_score}")

    # After finding optimal parameters, run simulation
    best_params = optimizer.max['params']
    if wants_example_visualizations:
        plot_optimization_surfaces(optimizer, global_best_score)
        plot_combined_visualization(optimizer)

    true_elo = 1250
    while True:
        simulate_convergence(best_params, true_elo, wants_guessed_starting_elos, prob_model=prob_model)
        new_true_elo = input("Enter a new true ELO (or enter to exit): ")
        if new_true_elo:
            true_elo = int(new_true_elo)
        else:
            break