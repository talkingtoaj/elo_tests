from utils import *
# https://bayesian-optimization.github.io/BayesianOptimization/2.0.0/
from bayes_opt import BayesianOptimization
from bayes_opt import acquisition
import time
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def black_box(user_actual_placement, alternating, entries_per_1000, max_variance, variance_decay, competitor_elo_delta, variance_sensitivity):
    rounds = 0
    true_elo = user_actual_placement
    user = Competitor(5000, max_variance, LOSER)
    population = build_population(entries_per_1000)
    last_match_won = None
    
    # Track when we first enter the correct band
    first_correct_entry = None
    
    while (user.variance > 500 or abs(user.score - true_elo) > 500) and rounds < 100:
        rounds += 1
        
        # Check if we're in the correct band
        in_correct_band = abs(user.score - true_elo) <= 500
        
        # Record first entry into correct band
        if in_correct_band and first_correct_entry is None:
            first_correct_entry = rounds
        # Reset if we leave the band
        elif not in_correct_band:
            first_correct_entry = None
            
        if alternating:
            if rounds % 2 == 0:
                lower_elo, upper_elo = user.score, user.score + competitor_elo_delta
            else:
                lower_elo, upper_elo = user.score - competitor_elo_delta, user.score
        else:
            lower_elo, upper_elo = get_competitor_elo_bounds(
                user.score,
                competitor_elo_delta,
                last_match_won,
                user.variance,
                variance_sensitivity
            )

        competitor, user = run_competition(population, user, true_elo, upper_elo, lower_elo, variance_decay)
        last_match_won = competitor.position == LOSER
    
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
        'max_variance':(1000, 4000),
        'variance_decay':(0.01, 0.13),
        'competitor_elo_delta':(1000, 9000),
        'variance_sensitivity':(0.1, 5.0),
    }
    alternating = False # also try True
    entries_per_1000 = 4
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
            'max_variance': round(next_point['max_variance'] / 10) * 10,
            'competitor_elo_delta': round(next_point['competitor_elo_delta'] / 10) * 10,
            'variance_decay': round(next_point['variance_decay'], 4),
            'variance_sensitivity': round(next_point['variance_sensitivity'], 1)
        }        
        scores = []
        for user_actual_placement in user_starting_placements:
            scores.append(black_box(user_actual_placement, alternating, entries_per_1000, **next_point))
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