"""What if real students do not follow the model the Bayesian version assumes?

The Bayesian code assumes a straight-line win curve (bayes.win_prob). Here the
simulated students follow other curves. Elo is re-tuned for each curve, so it
gets a fair shot. Run:  python robustness.py [students]
"""
import random
import sys
import numpy as np
import bayes
from bayes import Belief, pick_median, pick_eig
from compare import MAX_ROUNDS, ENTRIES_PER_1000, TOL, STOP_CONF
from utils import (build_population, initialialize_user, find_closest_competitor,
                   update, Competitor, WINNER, LOSER)

# P(student beats opponent), as a function of (student - opponent)
MODELS = {
    "linear (assumed)": lambda d: min(1, max(0, 0.5 + d / 1000)),
    "smooth, steep (logistic 250)": lambda d: 1 / (1 + np.exp(-d / 250)),
    "smooth, flat (logistic 700)": lambda d: 1 / (1 + np.exp(-d / 700)),
    "linear + 15% random flips": lambda d: 0.15 * 0.5 + 0.85 * min(1, max(0, 0.5 + d / 1000)),
}


def run_elo(true_elo, params, p_win):
    user = initialialize_user(true_elo, params["starting_variance"])
    pool = build_population(ENTRIES_PER_1000)
    last_won, est, stop = None, [], None
    for r in range(1, MAX_ROUNDS + 1):
        target = user.score + (user.variance if last_won else -user.variance)
        opp = find_closest_competitor(pool, target)
        last_won = random.random() < p_win(true_elo - opp)
        rival = Competitor(opp, 100, LOSER)
        user.position, rival.position = (WINNER, LOSER) if last_won else (LOSER, WINNER)
        update([user, rival], params["variance_decay"])
        est.append(user.score)
        if stop is None and user.variance <= TOL:
            stop = r
    return est, stop


def run_bayes(true_elo, picker, p_win):
    belief = Belief(initialialize_user(true_elo, 1).score)
    pool = build_population(ENTRIES_PER_1000)
    est, stop = [], None
    for r in range(1, MAX_ROUNDS + 1):
        opp = picker(belief, pool)
        pool.remove(opp)
        belief.update(opp, random.random() < p_win(true_elo - opp))
        est.append(belief.mean)
        if stop is None and belief.confidence(TOL) >= STOP_CONF:
            stop = r
    return est, stop


def tune_elo(truths, p_win):
    best, best_acc, rng = None, -1, random.Random(1)
    for _ in range(100):
        p = {"starting_variance": rng.randrange(500, 4001, 10),
             "variance_decay": round(rng.uniform(0.01, 0.13), 4)}
        runs = [run_elo(t, p, p_win)[0] for t in truths[:250]]
        acc = (np.abs(np.array(runs) - np.array(truths[:250])[:, None]) <= TOL)[:, 5].mean()
        if acc > best_acc:
            best, best_acc = p, acc
    return best


def row(name, runs, truths):
    est = np.array([e for e, _ in runs])
    ok = (np.abs(est - np.array(truths)[:, None]) <= TOL).mean(axis=0) * 100
    stops = [(i, s) for i, (_, s) in enumerate(runs) if s is not None]
    mean_stop = np.mean([s for _, s in stops]) if stops else float("nan")
    acc_stop = np.mean([np.abs(est[i, s - 1] - truths[i]) <= TOL for i, s in stops]) * 100 if stops else float("nan")
    print(f"  {name:<34} {ok[2]:5.0f} {ok[3]:5.0f} {ok[5]:5.0f} {ok[7]:5.0f} | "
          f"{mean_stop:5.1f} {acc_stop:6.0f}%")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 800
    random.seed(11)
    np.random.seed(11)
    truths = [random.randint(0, 10000) for _ in range(n)]
    print(f"{n} students. Columns: % within +-500 after 3, 4, 6, 8 questions | "
          f"mean stop round, accuracy at stop\n")
    for model_name, p_win in MODELS.items():
        print(model_name)
        params = tune_elo(truths, p_win)
        row("Elo (re-tuned)", [run_elo(t, params, p_win) for t in truths], truths)
        bayes.EPS = 0.02
        row("Bayes, assumes linear", [run_bayes(t, pick_median, p_win) for t in truths], truths)
        bayes.EPS = 0.15
        row("Bayes, hedged (15% slack)", [run_bayes(t, pick_median, p_win) for t in truths], truths)
        bayes.EPS = 0.02
        print()
