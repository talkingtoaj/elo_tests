from utils import *
# https://bayesian-optimization.github.io/BayesianOptimization/2.0.0/
from bayes_opt import BayesianOptimization
from bayes_opt import acquisition
import time
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def black_box(user_actual_placement, entries_per_1000, starting_variance, variance_decay:int):
    rounds = 0
    true_elo = user_actual_placement
    user = initialialize_user(true_elo, starting_variance)
    available_population = build_population(entries_per_1000)
    last_match_won = None
    
    # Track when we first enter the correct band
    first_correct_entry = None
    
    while (user.variance > 500 or abs(user.score - true_elo) > 500) and rounds < 100:
        rounds += 1
        
        # Calculate target based on last match
        target_elo = user.score + (user.variance if last_match_won else -user.variance)
        competitor_elo = find_closest_competitor(available_population, target_elo)
        
        # Run competition and update user
        _, user, last_match_won = run_competition(user, true_elo, competitor_elo, variance_decay)
        
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
    wants_example_visualizations = input("Do you want example visualizations? (y/n): ").lower() == "y"
    # init params
    params_gbm ={
        'starting_variance':(100, 4000),
        'variance_decay':(0.001, 0.13),
    }
    entries_per_1000 = 12
    acq = acquisition.UpperConfidenceBound(kappa=1.5)
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
    user_starting_placements = build_population(entries_per_1000=4)
    for i in range(500):
        next_point = optimizer.suggest()
        # Round parameters before evaluation
        next_point = {
            'starting_variance': round(next_point['starting_variance'] / 10) * 10,
            'variance_decay': round(next_point['variance_decay'], 4),
        }        
        scores = []
        for user_actual_placement in user_starting_placements:
            scores.append(black_box(user_actual_placement, entries_per_1000, **next_point))
        avg_score = sum(scores) / len(scores)
        optimizer.register(params=next_point, target=avg_score)
        output = f"{avg_score}: " + ", ".join([f"{label}:{value:.1f}" for label,value in next_point.items()])
        print(output)
        if i % 50 == 0 and wants_example_visualizations:
            simulate_convergence(next_point, 1250)  # Or any true_elo value
        
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
    for user_actual_placement in user_starting_placements:
        scores.append(black_box(user_actual_placement, entries_per_1000, **best_params))
    avg_score = sum(scores) / len(scores)
    print(f"Verified score: {avg_score}")

    # After finding optimal parameters, run simulation
    best_params = optimizer.max['params']
    plot_optimization_surfaces(optimizer, global_best_score)
    plot_combined_visualization(optimizer)

    true_elo = 1250
    while True:
        simulate_convergence(best_params, true_elo)  # Or any true_elo value
        new_true_elo = input("Enter a new true ELO (or enter to exit): ")
        if new_true_elo:
            true_elo = int(new_true_elo)
        else:
            break