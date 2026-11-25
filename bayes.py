"""Bayesian adaptive assessment.

Instead of a step size ("variance") that shrinks on a timer, we keep a full
belief curve over the student's score and update it with Bayes' rule after
every match. The next opponent is chosen to teach us the most.

This file has no plotting or tuning code, so it imports fast.
"""
import random
import numpy as np

GRID = np.arange(0, 10001, 50, dtype=float)  # candidate scores
EPS = 0.02  # floor on win/loss odds, so one fluke result cannot kill the belief


def win_prob(user_scores, opponent):
    """P(user beats opponent). Same model as utils.run_competition."""
    p = 0.5 + (user_scores - opponent) / 1000
    return np.clip(p, EPS, 1 - EPS)


class Belief:
    def __init__(self, guess, guess_sd=3000.0):
        # Prior matches initialialize_user: guess is the true score + N(0, 3000) noise
        post = np.exp(-0.5 * ((GRID - guess) / guess_sd) ** 2)
        self.post = post / post.sum()

    def update(self, opponent, user_won):
        like = win_prob(GRID, opponent)
        if not user_won:
            like = 1 - like
        post = self.post * like
        self.post = post / post.sum()

    @property
    def mean(self):
        return float((GRID * self.post).sum())

    @property
    def median(self):
        return float(GRID[np.searchsorted(np.cumsum(self.post), 0.5)])

    def confidence(self, width=500):
        """Belief that the true score is within +-width of our estimate."""
        m = self.mean
        return float(self.post[np.abs(GRID - m) <= width].sum())

    def sample(self):
        return float(np.random.choice(GRID, p=self.post))


def pick_median(belief, pool):
    """Opponent nearest the belief median (about 1 bit per match)."""
    return min(pool, key=lambda c: abs(c - belief.median))


def pick_thompson(belief, pool):
    """Thompson sampling: draw a score from the belief, match the closest opponent."""
    target = belief.sample()
    return min(pool, key=lambda c: abs(c - target))


def pick_eig(belief, pool):
    """Opponent with the biggest expected drop in uncertainty (entropy)."""
    opps = np.array(pool, dtype=float)
    like = win_prob(GRID[None, :], opps[:, None])  # (opps, grid) P(win)
    p_win = like @ belief.post
    post_w = like * belief.post
    post_l = (1 - like) * belief.post

    def entropy(unnorm, total):
        p = unnorm / total[:, None]
        return -(p * np.log(p + 1e-12)).sum(axis=1)

    exp_h = p_win * entropy(post_w, p_win) + (1 - p_win) * entropy(post_l, 1 - p_win)
    return pool[int(np.argmin(exp_h))]


PICKERS = {"median": pick_median, "thompson": pick_thompson, "eig": pick_eig}
