from typing import Union
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

WINNER = 1
LOSER = 0
DRAW = 0.5


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
    Create three separate plots showing the Gaussian Process model for each parameter
    """
    plt.figure(figsize=(15, 5))
    
    # Get the parameter names and bounds
    param_names = list(optimizer.space.keys)
    param_bounds = optimizer.space.bounds
    
    # Create subplots for each parameter
    for i, (param_name, bounds) in enumerate(zip(param_names, param_bounds)):
        plt.subplot(1, 3, i+1)
        
        # Create parameter grid for this dimension
        param_grid = np.linspace(bounds[0], bounds[1], 1000).reshape(-1, 1)
        
        # Create full input grid by using mean values for other parameters
        X_grid = np.zeros((1000, len(param_names)))
        mean_values = np.mean(optimizer.space.params, axis=0)
        for j in range(len(param_names)):
            if j == i:
                X_grid[:, j] = param_grid.ravel()
            else:
                X_grid[:, j] = mean_values[j]
        
        # Get predictions from GP
        mu, sigma = optimizer._gp.predict(X_grid, return_std=True)
        
        # Plot the GP
        plt.plot(param_grid, mu, '--', color='k', label='Prediction')
        plt.fill_between(param_grid.ravel(), 
                        mu - 1.96 * sigma, 
                        mu + 1.96 * sigma, 
                        alpha=0.2, color='c', 
                        label='95% confidence interval')
        
        # Plot observations
        observed_params = optimizer.space.params[:, i]
        observed_targets = optimizer.space.target
        plt.scatter(observed_params, observed_targets, 
                   color='r', marker='D', label='Observations')
        
        plt.xlabel(param_name)
        plt.ylabel('Target')
        plt.title(f'GP Model for {param_name}')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('optimization_surfaces.png')
    plt.close()

def plot_combined_visualization(optimizer):
    """
    Create a visualization showing the acquisition function for each parameter
    """
    plt.figure(figsize=(15, 5))
    
    # Get the parameter names and bounds
    param_names = list(optimizer.space.keys)
    param_bounds = optimizer.space.bounds
    
    # Create subplots for each parameter
    for i, (param_name, bounds) in enumerate(zip(param_names, param_bounds)):
        plt.subplot(1, 3, i+1)
        
        # Create parameter grid for this dimension
        param_grid = np.linspace(bounds[0], bounds[1], 1000).reshape(-1, 1)
        
        # Create full input grid by using mean values for other parameters
        X_grid = np.zeros((1000, len(param_names)))
        mean_values = np.mean(optimizer.space.params, axis=0)
        for j in range(len(param_names)):
            if j == i:
                X_grid[:, j] = param_grid.ravel()
            else:
                X_grid[:, j] = mean_values[j]
        
        # Get acquisition function values
        acq_values = -optimizer.acquisition_function._get_acq(optimizer._gp)(X_grid)
        
        # Plot acquisition function
        plt.plot(param_grid, acq_values, color='purple', label='Acquisition Function')
        
        # Plot next best guess
        next_best_idx = np.argmax(acq_values)
        plt.plot(param_grid[next_best_idx], acq_values[next_best_idx], '*', 
                markersize=15, label='Next Best Guess', 
                markerfacecolor='gold', markeredgecolor='k', markeredgewidth=1)
        
        plt.xlabel(param_name)
        plt.ylabel('Acquisition Value')
        plt.title(f'Acquisition Function for {param_name}')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('acquisition_functions.png')
    plt.close()

def plot_combined_visualization(optimizer):
    """
    Creates a single visualization combining all three parameters
    """
    plt.switch_backend('TkAgg')
    
    # Approach 1: 3D scatter plot with color
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract optimization history
    points = np.array([[res["params"]["max_variance"], 
                       res["params"]["variance_decay"],
                       res["params"]["competitor_elo_delta"]] 
                      for res in optimizer.res])
    scores = np.array([res["target"] for res in optimizer.res])
    
    # Normalize scores for coloring
    norm = plt.Normalize(scores.min(), scores.max())
    colors = plt.cm.viridis(norm(scores))
    
    scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                        c=scores, cmap='viridis',
                        s=100)  # Size of points
    
    ax.set_xlabel('max_variance')
    ax.set_ylabel('variance_decay')
    ax.set_zlabel('competitor_elo_delta')
    
    plt.colorbar(scatter, label='Score')
    plt.title('Combined Parameter Space Visualization')
    plt.tight_layout()
    plt.show()
    
    # Approach 2: Parallel coordinates plot
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    
    # Normalize all parameters to [0,1] for visualization
    normalized_points = (points - points.min(axis=0)) / (points.max(axis=0) - points.min(axis=0))
    
    # Plot parallel coordinates
    for i in range(len(points)):
        ax2.plot(range(3), normalized_points[i], 
                c=plt.cm.viridis(norm(scores[i])),
                alpha=0.5)
    
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(['max_variance', 'variance_decay', 'competitor_elo_delta'])
    
    # Add colorbar to the parallel coordinates plot
    sm = plt.cm.ScalarMappable(norm=norm, cmap='viridis')
    sm.set_array([])
    plt.colorbar(sm, ax=ax2, label='Score')
    
    plt.title('Parallel Coordinates Visualization')
    plt.tight_layout()
    plt.show()