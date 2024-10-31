from utils import *
# https://bayesian-optimization.github.io/BayesianOptimization/2.0.0/
from bayes_opt import BayesianOptimization
from bayes_opt import acquisition
import time
import random

def run_competition(population, user, actual_user_elo, upper_elo, lower_elo, variance_decay):
    # returns a random number from the range
    competitor = Competitor(random.choice(population), 100, LOSER)
    if competitor.score > actual_user_elo:
        competitor.position = WINNER
        user.position = LOSER
    else:
        competitor.position = LOSER
        user.position = WINNER
    update([user, competitor], variance_decay)
    return competitor, user

def build_population(entries_per_1000):
    elos = []
    for i in range(0, 10000, 1000):
        for j in range(entries_per_1000):
            elos.append(random.randint(i, i+1000))
    elos.sort()
    return elos

def black_box(user_actual_placement, alternating, entries_per_1000, max_variance, variance_decay, competitor_elo_delta):
    # we run ELO contests until the user's variance is less than 500 or the user's score is within 500 of the true elo, then we return the number of rounds
    rounds = 0
    last_higher = False
    true_elo = user_actual_placement
    guessed_elo = 5000
    user = Competitor(guessed_elo, max_variance, LOSER)
    # create a list representing the population of elos; between 0 and 10000, we create 4 elos randomly for each 1000 range
    population = build_population(entries_per_1000)

    while (user.variance > 500 or abs(user.score - true_elo) > 500) and rounds < 100:
        rounds += 1
        if alternating:
            if last_higher:
                upper_elo = user.score
                lower_elo = user.score - competitor_elo_delta
            else:
                lower_elo = user.score
                upper_elo = user.score + competitor_elo_delta
            last_higher = not last_higher
        else:
            lower_elo = user.score - competitor_elo_delta
            upper_elo = user.score + competitor_elo_delta
        competitor, user = run_competition(population, user, true_elo, upper_elo, lower_elo, variance_decay)
    # as we are trying to maximize black_box results, but we want to minimize rounds, we return the negative
    if rounds >= 100:
        return -1000
    return -rounds 
    
if __name__ == '__main__':
    # init params
    params_gbm ={
        'max_variance':(500, 4000),
        'variance_decay':(0.02, 0.5),
        'competitor_elo_delta':(1000, 3000),
    }
    alternating = True # also try False
    entries_per_1000 = 4
    acq = acquisition.UpperConfidenceBound(kappa=2.5)
    optimizer = BayesianOptimization(
        f=None,
        acquisition_function=acq,
        pbounds=params_gbm,
        verbose=3,
        random_state=1,
    )

    # Run Bayesian Optimization
    start = time.time()
    user_starting_placements = build_population(entries_per_1000=4)
    for i in range(500):
        next_point = optimizer.suggest()
        # we will run the black box 4 times for each point, with 4 different user actual placements, then take the average
        scores = []
        for user_actual_placement in user_starting_placements:
            scores.append(black_box(user_actual_placement, alternating, entries_per_1000, **next_point))
        avg_score = sum(scores) / len(scores)
        # print(scores)
        optimizer.register(params=next_point, target=avg_score)
        output = f"{avg_score}: " + ", ".join([f"{label}:{value:.1f}" for label,value in next_point.items()])
        print(output)

    print(f"Time taken: {time.time() - start}")
    print(optimizer.max)