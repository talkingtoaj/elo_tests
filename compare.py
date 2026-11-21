"""Compare the current Elo approach with Bayesian adaptive versions.

Every method sees the same students, the same opponent pool size, the same
starting guess, and the same simulated match results (utils.run_competition's
model). Run:  python compare.py
"""
import random
import sys
import numpy as np
from utils import (build_population, initialialize_user, run_competition,
                   find_closest_competitor)
from bayes import Belief, PICKERS

MAX_ROUNDS = 12
ENTRIES_PER_1000 = 12
TOL = 500
STOP_CONF = 0.90


def sim_win(true_elo, opponent):
    """Same result model as utils.run_competition."""
    p_opp = max(0, min(1, 0.5 + (opponent - true_elo) / 1000))
    return random.random() >= p_opp


def run_elo(true_elo, params):
    """Current approach (elo.black_box loop). Returns estimate after each round."""
    user = initialialize_user(true_elo, params["starting_variance"])
    pool = build_population(ENTRIES_PER_1000)
    last_won = None
    est, stop_round = [], None
    for r in range(1, MAX_ROUNDS + 1):
        target = user.score + (user.variance if last_won else -user.variance)
        opp = find_closest_competitor(pool, target)
        _, user, last_won = run_competition(user, true_elo, opp, params["variance_decay"])
        est.append(user.score)
        if stop_round is None and user.variance <= TOL:
            stop_round = r
    return est, stop_round


def run_bayes(true_elo, picker):
    belief = Belief(initialialize_user(true_elo, 1).score)
    pool = build_population(ENTRIES_PER_1000)
    est, stop_round = [], None
    for r in range(1, MAX_ROUNDS + 1):
        opp = picker(belief, pool)
        pool.remove(opp)
        belief.update(opp, sim_win(true_elo, opp))
        est.append(belief.mean)
        if stop_round is None and belief.confidence(TOL) >= STOP_CONF:
            stop_round = r
    return est, stop_round


def summarize(name, runs, truths):
    est = np.array([e for e, _ in runs])           # (trials, rounds)
    ok = np.abs(est - np.array(truths)[:, None]) <= TOL
    acc_by_round = ok.mean(axis=0)
    stops = [s for _, s in runs]
    stopped = [i for i, s in enumerate(stops) if s is not None]
    stop_rounds = [stops[i] for i in stopped]
    acc_at_stop = np.mean([ok[i, stops[i] - 1] for i in stopped]) if stopped else float("nan")
    print(f"\n{name}")
    print("  accuracy (within +-500) after round n:")
    print("   n:   " + " ".join(f"{n:5d}" for n in range(1, MAX_ROUNDS + 1)))
    print("   acc: " + " ".join(f"{a*100:5.1f}" for a in acc_by_round))
    mean_stop = np.mean(stop_rounds) if stop_rounds else float("nan")
    print(f"  stops by round {MAX_ROUNDS}: {len(stopped)/len(runs)*100:.0f}% of students | "
          f"mean stop round {mean_stop:.1f} | accuracy when it stops {acc_at_stop*100:.1f}%")
    return acc_by_round


def tune_elo(truths):
    """Fair baseline: quick random search for the best Elo params on its own metric."""
    best, best_score = None, -1
    rng = random.Random(1)
    for _ in range(150):
        p = {"starting_variance": rng.randrange(500, 4001, 10),
             "variance_decay": round(rng.uniform(0.01, 0.13), 4)}
        runs = [run_elo(t, p) for t in truths[:300]]
        est = np.array([e for e, _ in runs])
        ok = np.abs(est - np.array(truths[:300])[:, None]) <= TOL
        score = ok[:, 5].mean()  # accuracy after 6 rounds, the README's target
        if score > best_score:
            best, best_score = p, score
    return best


if __name__ == "__main__":
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    random.seed(7)
    np.random.seed(7)
    truths = [random.randint(0, 10000) for _ in range(n_trials)]

    print(f"{n_trials} simulated students, true score uniform 0..10000")
    print("Tuning Elo params (best of 150 random tries) ...")
    elo_params = tune_elo(truths)
    print(f"Elo params used: {elo_params}")

    summarize("Elo (current)", [run_elo(t, elo_params) for t in truths], truths)
    for name, picker in PICKERS.items():
        summarize(f"Bayesian, opponent = {name}",
                  [run_bayes(t, picker) for t in truths], truths)
