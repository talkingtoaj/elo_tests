from typing import Union
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import random
from scipy.interpolate import CubicSpline
from scipy.interpolate import griddata

WINNER = 1
LOSER = 0
DRAW = 0.5

def build_population(entries_per_1000):
    elos = []
    for i in range(0, 10000, 1000):
        for j in range(entries_per_1000):
            elos.append(random.randint(i, i+1000))
    elos.sort()
    return elos

def initialialize_user(true_elo, starting_variance):
    # We simulate knowing the user's grammar score which is going to be roughly related to their reading comprehension score
    grammar_score = random.gauss(true_elo, 3000)
    # Clamp the value between 0 and 10000 to stay within reasonable ELO bounds
    initial_guess = max(0, min(10000, grammar_score))
    return Competitor(initial_guess, starting_variance, LOSER)
    
def simulate_convergence(optimal_params, true_elo, entries_per_1000=4, pause=0.1):
    """Simulate and visualize the convergence of ELO ratings using the optimal parameters"""
    # Initialize tracking arrays
    rounds, actual_scores, guessed_scores, competitor_scores, variances = [], [], [], [], []

    user = initialialize_user(true_elo, optimal_params['starting_variance'])
    population = build_population(entries_per_1000)
    
    plt.figure(figsize=(12, 6))
    plt.ion()
    
    # Track last match result
    last_match_won = None
    
    for round_num in range(40):
        # Store current state
        rounds.append(round_num)
        actual_scores.append(true_elo)
        guessed_scores.append(user.score)
        variances.append(user.variance)
        
        # Calculate target ELO based on last match result and variance (consistent with black_box)
        target_elo = user.score + (user.variance if last_match_won else -user.variance) if last_match_won is not None else user.score
        competitor_elo = find_closest_competitor(population, target_elo)
            
        # Run competition
        competitor, user, last_match_won = run_competition(
            user, 
            true_elo, 
            competitor_elo, 
            optimal_params['variance_decay']
        )
        
        competitor_scores.append(competitor.score)
        
        # Update visualization
        plt.clf()
        plt.plot(rounds, actual_scores, 'g-', label='True ELO')
        plt.plot(rounds, guessed_scores, 'b-', label='Guessed ELO')
        plt.plot(rounds, competitor_scores, 'r.', label='Competitor ELO')
        
        # Create continuous variance bands
        scores = np.array(guessed_scores)
        vars_array = np.array(variances)
        lower_band = scores - vars_array
        upper_band = scores + vars_array
        
        # Track transitions for continuous fills
        close_mask = np.abs(scores - true_elo) <= 500
        transition_points = np.where(close_mask[:-1] != close_mask[1:])[0]
        
        # Fill far (blue) regions
        far_mask = ~close_mask
        if np.any(far_mask):
            plt.fill_between(rounds, lower_band, upper_band, 
                           where=far_mask, alpha=0.2, color='blue')
            
        # Fill close (green) regions
        if np.any(close_mask):
            plt.fill_between(rounds, lower_band, upper_band, 
                           where=close_mask, alpha=0.2, color='green')
            
        # Fill transition points to ensure continuity
        for tp in transition_points:
            plt.fill_between([rounds[tp], rounds[tp+1]], 
                           [lower_band[tp], lower_band[tp+1]],
                           [upper_band[tp], upper_band[tp+1]],
                           alpha=0.2, color='blue' if close_mask[tp] else 'green')
        
        plt.xlabel('Round')
        plt.ylabel('ELO Score')
        plt.title(f'ELO Rating Convergence\n'
                 f"var:{optimal_params['starting_variance']:.0f}, "
                 f"decay:{optimal_params['variance_decay']:.2f}, "
        )
        plt.legend()
        plt.grid(True)
        plt.pause(pause)
        
        if user.variance <= 500 and abs(user.score - true_elo) <= 500:
            break
    
    plt.ioff()
    plt.show()
    return round_num

def run_competition(user, actual_user_elo, competitor_elo, variance_decay):
    """Run a single competition between user and competitor.
    
    Args:
        population: List of available ELO scores
        user: Competitor object representing the user
        actual_user_elo: True ELO of the user
        competitor_elo: ELO of the competitor
        variance_decay: Variance decay factor
        
    Returns:
        tuple: (competitor object, user object, whether user won)
    """
    competitor = Competitor(competitor_elo, 100, LOSER)
    
    # Calculate win probability based on ELO difference
    elo_diff = competitor.score - actual_user_elo
    win_probability = 0.5 + (elo_diff / 1000)
    win_probability = max(0, min(1, win_probability))
    
    # Determine winner based on probability
    user_won = random.random() >= win_probability
    if user_won:
        competitor.position = LOSER
        user.position = WINNER
    else:
        competitor.position = WINNER
        user.position = LOSER
        
    update([user, competitor], variance_decay)
    return competitor, user, user_won


def update(competitor_list, variance_decrease_pc:float=0.1):
    """
        <competitor_list>: a list of competitor objects. 
          These can be any model, but require the following attributes to be pre-set:
          score: ELO score
          variance
          position: must be from WINNER, LOSER, DRAW

        returns competitor_list with updated score and variance values
    """

    num_competitors = len(competitor_list)
    if len(competitor_list) < 1:
        raise ValueError("At least one competitor is required")
    if len(competitor_list) == 1:
        promote_single(winner=competitor_list[0], variance_decrease_pc=variance_decrease_pc)
    if variance_decrease_pc > 1.0 or variance_decrease_pc < 0.0:
        raise ValueError("variance_decrease_pc must be between 0 and 1.0")

    R = {}  # define dictionary
    K = {}
    sum = 0

    for idx, competitor in enumerate(competitor_list):
        # R(1) = 10**(1elo/400)
        # R(2) = 10**(2elo/400)
        R[idx] = 10**(competitor.score/400)
        sum = sum + R[idx]       
        K[idx] = competitor.variance

    # score = 1 for win, 0 for loss, middle players get 0.5/(number of middle players)
    if num_competitors > 2:
        neutral_score = 0.5/(num_competitors-2)
    else:
        neutral_score = DRAW

    for idx, competitor in enumerate(competitor_list):
        result = competitor.position
        if competitor.position == DRAW:
            result = neutral_score
        competitor.score = competitor.score + K[idx] * (result -(R[idx] / sum))
        competitor.variance = competitor.variance * (1-variance_decrease_pc) # decrease K-variance after a contest
    return competitor_list

def promote_single (winner, variance_decrease_pc:float=0.1, competitor_elo=2000):
    """ Given a single competitor selected as suitable, with no losers, ELO and variances are updated against fictional rival"""    
    loser = Competitor(elo_score=2000, variance=0, position=LOSER)
    competitors = [winner, loser]
    return update(competitors)[0]


class Competitor:
    """ If the element you want to elo_compete does not have score and variance attributes you can set, you can use this wrapper class """
    # Unfortunately this currently does not allow us to calculate multiple competitions (e.g. this competitor coming 2nd out of 3 competitors. 
    # That will require two competitions until a wrapper is built)
    def __init__(self, elo_score:Union[int,float], variance:Union[int,float], position, obj=None):
        if position not in [WINNER, LOSER, DRAW]:
            raise ValueError (f"Invalid value for position {position}")
        self.score = elo_score
        self.variance = variance
        self.position = position
        self.obj = obj

def plot_optimization_surfaces(optimizer, global_best_score):
    """
    Create four separate plots showing polynomial fit for each parameter
    """
    plt.figure(figsize=(20, 5))
    plt.tight_layout()
    
    param_names = list(optimizer.space.keys)
    param_bounds = optimizer.space.bounds
    
    for i, (param_name, bounds) in enumerate(zip(param_names, param_bounds)):
        plt.subplot(1, 4, i+1)
        
        # Get observed data for this parameter
        observed_params = optimizer.space.params[:, i]
        observed_targets = optimizer.space.target
        
        # Sort points for plotting
        sort_idx = np.argsort(observed_params)
        x_sorted = observed_params[sort_idx]
        y_sorted = observed_targets[sort_idx]
        
        # Determine polynomial degree (adaptive to number of points)
        n_points = len(observed_params)
        poly_degree = min(6, max(1, n_points // 10))
        
        # Fit polynomial
        coeffs = np.polyfit(x_sorted, y_sorted, poly_degree)
        
        # Create smooth points for plotting
        x_smooth = np.linspace(bounds[0], bounds[1], 1000)
        y_smooth = np.polyval(coeffs, x_smooth)
        
        # Calculate confidence bands using residuals
        y_fit = np.polyval(coeffs, x_sorted)
        residuals = y_sorted - y_fit
        std_dev = np.std(residuals)
        
        # Plot the results
        plt.plot(x_smooth, y_smooth, '--', color='k', 
                label=f'Polynomial (degree {poly_degree})')
        plt.fill_between(x_smooth, 
                        y_smooth - 1.96 * std_dev,
                        y_smooth + 1.96 * std_dev,
                        alpha=0.2, color='c',
                        label='95% confidence interval')
        
        plt.scatter(observed_params, observed_targets,
                   color='r', marker='D', label='Observations')
        
        plt.xlabel(param_name)
        plt.ylabel('Target')
        plt.title(f'Polynomial Fit for {param_name}')
        plt.legend()
    
    plt.tight_layout()
    plt.show()

def plot_combined_visualization(optimizer):
    """
    Creates a visualization combining the parameters
    """
    plt.switch_backend('TkAgg')
    
    # Extract data once
    points = np.array([[res["params"]["starting_variance"], 
                       res["params"]["variance_decay"]]
                      for res in optimizer.res])
    scores = np.array([res["target"] for res in optimizer.res])
    
    # Create normalization object first
    norm = plt.Normalize(scores.min(), scores.max())
    
    # 3D scatter plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(points[:, 0], points[:, 1], scores,
                        c=scores, cmap='viridis',
                        s=100)
    
    ax.set_xlabel('starting_variance')
    ax.set_ylabel('variance_decay')
    ax.set_zlabel('score')
    
    plt.colorbar(scatter, label='Score')
    plt.title('Parameter Space Visualization')
    plt.tight_layout()
    plt.show()
    
    # Parallel coordinates plot with reduced width
    fig2, ax2 = plt.subplots(figsize=(8, 6))  # Reduced width from 12 to 8
    
    # Normalize parameters to [0,1]
    normalized_points = (points - points.min(axis=0)) / (points.max(axis=0) - points.min(axis=0))
    
    # Create parallel coordinates
    param_count = points.shape[1]
    for i in range(len(points)):
        ax2.plot(range(param_count), normalized_points[i], 
                c=plt.cm.viridis(norm(scores[i])),
                alpha=0.5)
    
    ax2.set_xticks(range(param_count))
    ax2.set_xticklabels(['starting_variance', 'variance_decay'])
    
    # Add colorbar with smaller size
    sm = plt.cm.ScalarMappable(norm=norm, cmap='viridis')
    sm.set_array([])
    plt.colorbar(sm, ax=ax2, label='Score', fraction=0.046)  # Added fraction parameter to make colorbar thinner
    
    plt.title('Parallel Coordinates Visualization')
    plt.tight_layout()
    plt.show()

def find_closest_competitor(population, target_elo):
    """Find the competitor closest to target_elo and remove from population."""
    if not population:
        raise ValueError("Population is empty")
    
    closest = min(population, key=lambda x: abs(x - target_elo))
    population.remove(closest)
    return closest

def plot_3d_surface(optimizer):
    """
    Create 3D surface plot of optimization results
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract data points
    points = np.array([[res["params"]["starting_variance"], 
                       res["params"]["variance_decay"]] 
                      for res in optimizer.res])
    scores = np.array([res["target"] for res in optimizer.res])
    
    # Create grid for surface (reduced resolution for memory efficiency)
    grid_points = 50  # Reduced from 100 for better performance
    xi = np.linspace(points[:, 0].min(), points[:, 0].max(), grid_points)
    yi = np.linspace(points[:, 1].min(), points[:, 1].max(), grid_points)
    xi, yi = np.meshgrid(xi, yi)
    
    try:
        # Interpolate with error handling
        zi = griddata((points[:, 0], points[:, 1]), scores, (xi, yi), 
                     method='cubic', fill_value=np.nan)
        
        # Plot surface (only where interpolation succeeded)
        mask = ~np.isnan(zi)
        surf = ax.plot_surface(xi[mask], yi[mask], zi[mask], 
                             cmap='viridis', alpha=0.6)
        
        # Plot actual points
        scatter = ax.scatter(points[:, 0], points[:, 1], scores, 
                           c=scores, cmap='viridis', 
                           s=50, alpha=1)
        
        ax.set_xlabel('Starting Variance')
        ax.set_ylabel('Variance Decay')
        ax.set_zlabel('Score')
        
        plt.colorbar(surf, label='Score')
        plt.title('Optimization Surface')
        
    except Exception as e:
        print(f"Error creating surface plot: {e}")
        # Fall back to scatter plot only
        scatter = ax.scatter(points[:, 0], points[:, 1], scores, 
                           c=scores, cmap='viridis', 
                           s=50, alpha=1)
        plt.colorbar(scatter, label='Score')
        plt.title('Optimization Points (Surface interpolation failed)')
    
    plt.tight_layout()
    plt.show()
