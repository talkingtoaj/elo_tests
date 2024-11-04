from typing import Union
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import random
from scipy.interpolate import CubicSpline

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

    
def simulate_convergence(optimal_params, true_elo, alternating=False, entries_per_1000=4, pause=0.1):
    """Simulate and visualize the convergence of ELO ratings using the optimal parameters"""
    # Initialize tracking arrays
    rounds, actual_scores, guessed_scores, competitor_scores, variances = [], [], [], [], []
    user = Competitor(5000, optimal_params['max_variance'], LOSER)
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
        
        # Determine competitor ELO bounds using shared logic
        if alternating:
            if round_num % 2 == 0:
                lower_elo, upper_elo = user.score, user.score + optimal_params['competitor_elo_delta']
            else:
                lower_elo, upper_elo = user.score - optimal_params['competitor_elo_delta'], user.score
        else:
            lower_elo, upper_elo = get_competitor_elo_bounds(
                user.score, 
                optimal_params['competitor_elo_delta'],
                last_match_won,
                user.variance
            )
            
        # Run competition
        competitor, user = run_competition(population, user, true_elo, upper_elo, lower_elo, 
                                        optimal_params['variance_decay'])
        # Update last match result
        last_match_won = competitor.position == LOSER
        
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
                 f"var:{optimal_params['max_variance']:.0f}, "
                 f"decay:{optimal_params['variance_decay']:.2f}, "
                 f"delta:{optimal_params['competitor_elo_delta']:.0f}")
        plt.legend()
        plt.grid(True)
        plt.pause(pause)
        
        if user.variance <= 500 and abs(user.score - true_elo) <= 500:
            break
    
    plt.ioff()
    plt.show()
    return round_num

def run_competition(population, user, actual_user_elo, upper_elo, lower_elo, variance_decay):
    competitor = Competitor(random.choice(population), 100, LOSER)
    
    # Calculate win probability based on ELO difference
    elo_diff = competitor.score - actual_user_elo
    # Linear scale: 0 diff = 50% chance, 500 diff = 100% chance (or 0% if negative)
    win_probability = 0.5 + (elo_diff / 1000)  # This creates a scale from 0 to 1 centered at 0.5
    win_probability = max(0, min(1, win_probability))  # Clamp between 0 and 1
    
    # Determine winner based on probability
    if random.random() < win_probability:
        competitor.position = WINNER
        user.position = LOSER
    else:
        competitor.position = LOSER
        user.position = WINNER
        
    update([user, competitor], variance_decay)
    return competitor, user


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
    Creates a single visualization combining all four parameters
    """
    plt.switch_backend('TkAgg')
    
    # Approach 1: 3D scatter plot with color (using first 3 parameters)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract optimization history
    points = np.array([[res["params"]["max_variance"], 
                       res["params"]["variance_decay"],
                       res["params"]["competitor_elo_delta"],
                       res["params"]["variance_sensitivity"]] 
                      for res in optimizer.res])
    scores = np.array([res["target"] for res in optimizer.res])
    
    # Normalize scores for coloring
    norm = plt.Normalize(scores.min(), scores.max())
    colors = plt.cm.viridis(norm(scores))
    
    # Plot first 3 parameters in 3D
    scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                        c=scores, cmap='viridis',
                        s=100)  # Size of points
    
    ax.set_xlabel('max_variance')
    ax.set_ylabel('variance_decay')
    ax.set_zlabel('competitor_elo_delta')
    
    plt.colorbar(scatter, label='Score')
    plt.title('Combined Parameter Space Visualization (First 3 Parameters)')
    plt.tight_layout()
    plt.show()
    
    # Approach 2: Parallel coordinates plot (all 4 parameters)
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    
    # Normalize all parameters to [0,1] for visualization
    normalized_points = (points - points.min(axis=0)) / (points.max(axis=0) - points.min(axis=0))
    
    # Plot parallel coordinates
    for i in range(len(points)):
        ax2.plot(range(4), normalized_points[i], 
                c=plt.cm.viridis(norm(scores[i])),
                alpha=0.5)
    
    ax2.set_xticks(range(4))
    ax2.set_xticklabels(['max_variance', 'variance_decay', 
                        'competitor_elo_delta', 'variance_sensitivity'])
    
    # Add colorbar to the parallel coordinates plot
    sm = plt.cm.ScalarMappable(norm=norm, cmap='viridis')
    sm.set_array([])
    plt.colorbar(sm, ax=ax2, label='Score')
    
    plt.title('Parallel Coordinates Visualization')
    plt.tight_layout()
    plt.show()

def get_competitor_elo_bounds(user_score, competitor_elo_delta, last_match_won=None, user_variance=0, variance_sensitivity=1.0):
    """
    Args:
        user_score: Current ELO score of the user
        competitor_elo_delta: Maximum ELO difference to consider
        last_match_won: None for first match, otherwise True if user won last match
        user_variance: Current variance of user's ELO estimate
        variance_sensitivity: Controls how quickly the sigmoid transitions
    """
    if last_match_won is None:
        return (
            user_score - competitor_elo_delta,
            user_score + competitor_elo_delta
        )
    
    # Sigmoid function centered at variance=2500 (midpoint)
    opposite_probability = 1 / (1 + np.exp(-variance_sensitivity * (user_variance - 2500)/1000))
    
    if random.random() < opposite_probability:
        last_match_won = not last_match_won
    
    if last_match_won:
        return (user_score, user_score + competitor_elo_delta)
    else:
        return (user_score - competitor_elo_delta, user_score)
