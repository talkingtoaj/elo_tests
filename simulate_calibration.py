"""Fake-data study: what can a small course learn from its own calibration data?

For each hidden "truth" (a win curve plus real question difficulties, some of
them misplaced), we:
  1. run a course of N students through 20 calibration questions each,
  2. set up the library four ways:
       defaults         no fitting, built-in curve, questions sit at band centre
       curve only       fit scale and slip
       curve+difficulty fit scale, slip and each question's difficulty
       oracle           the true curve and true difficulties (best possible)
  3. test each setup on fresh students with the adaptive test,
  4. score: questions asked, exact-band accuracy (all students, and only
     those not near a band edge), how often the reported range holds the
     true band (the test claims 90%), and the average range width.

Run:  python simulate_calibration.py [reps] [test_students]
"""
import sys
import time

import numpy as np

from bandtest import Course, Curve

BANDS = 10
PER_BAND = 10
GUESS = 0.02

TRUTHS = {
    "matches defaults (scale .5, slip .05)": Curve(0.5, 0.05, GUESS),
    "flatter and noisier (scale .9, slip .12)": Curve(0.9, 0.12, GUESS),
    "steeper and cleaner (scale .3, slip .02)": Curve(0.3, 0.02, GUESS),
}
SIZES = [30, 100, 300]
SETTINGS = ["defaults", "curve only", "curve+difficulty", "oracle"]


def p_right(curve, level, difficulty):
    z = (level - difficulty) / curve.scale
    return curve.guess + (1 - curve.guess - curve.slip) / (1 + np.exp(-z))


def make_world(rng, per_band=PER_BAND):
    """True question difficulties: band centre plus noise, 8% misplaced by 2 bands."""
    ids, bands, true_d = [], [], {}
    for b in range(1, BANDS + 1):
        for k in range(per_band):
            qid = f"b{b}-{k}"
            d = b - 0.5 + rng.normal(0, 0.3)
            if rng.random() < 0.08:
                d += rng.choice([-2.0, 2.0])
            ids.append(qid)
            bands.append(b)
            true_d[qid] = float(np.clip(d, 0, BANDS))
    return ids, bands, true_d


def calibrate(n_students, truth, ids, bands, true_d, rng, seed):
    course = Course(BANDS, seed=seed)
    for q, b in zip(ids, bands):
        course.add_question(q, b)
    for i in range(n_students):
        level = rng.uniform(0, BANDS)
        s = course.start_student(f"cal{i}", mode="calibrate", n_calibration=20)
        while (q := s.next_question()) is not None:
            s.record(q, bool(rng.random() < p_right(truth, level, true_d[q])))
    return course


def configure(base_json, setting, truth, true_d, seed):
    course = Course.from_json(base_json, seed=seed)
    if setting == "curve only":
        course.fit(fit_curve=True, fit_difficulty=False)
    elif setting == "curve+difficulty":
        course.fit(fit_curve=True, fit_difficulty=True)
    elif setting == "oracle":
        course.set_curve(truth)
        for q, d in true_d.items():
            course.set_difficulty(q, d)
    return course


def evaluate(course, truth, true_d, n_test, rng):
    """Adaptive test on fresh students. Uses a copy so the course log is untouched."""
    stats = {"exact": 0, "clear": 0, "covers": 0, "width": 0, "asked": 0}
    n_clear = 0
    for i in range(n_test):
        level = rng.uniform(0, BANDS)
        s = course.start_student(f"test{i}", mode="adaptive")
        while (q := s.next_question()) is not None:
            s.record(q, bool(rng.random() < p_right(truth, level, true_d[q])))
        est = s.estimate()
        true_band = int(level) + 1
        lo, hi = est.band_range
        stats["exact"] += est.band == true_band
        stats["covers"] += lo <= true_band <= hi
        stats["width"] += hi - lo + 1
        stats["asked"] += est.n_answered
        if min(level % 1, 1 - level % 1) >= 0.3:   # not near a band edge
            n_clear += 1
            stats["clear"] += est.band == true_band
    out = {k: v / n_test for k, v in stats.items()}
    out["clear"] = stats["clear"] / max(n_clear, 1)
    return out


def flag_quality(course, true_d, ids, bands):
    flagged = {f[0] for f in course.flagged_questions()}
    truly_bad = {q for q, b in zip(ids, bands) if abs(true_d[q] - (b - 0.5)) >= 1.5}
    hit = len(flagged & truly_bad)
    return hit, len(flagged), len(truly_bad)


if __name__ == "__main__":
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    n_test = int(sys.argv[2]) if len(sys.argv) > 2 else 150
    t0 = time.time()
    for t_name, truth in TRUTHS.items():
        print(f"\n=== Truth: {t_name} ===")
        print(f"{'students':>8}  {'setup':<17} {'asked':>6} {'exact%':>6} {'clear%':>6} "
              f"{'range%':>6} {'width':>5}  fitted curve")
        for n in SIZES:
            acc = {s: [] for s in SETTINGS}
            fitted, flags = [], [0, 0, 0]
            for rep in range(reps):
                rng = np.random.default_rng(1000 * rep + n)
                ids, bands, true_d = make_world(rng)
                base = calibrate(n, truth, ids, bands, true_d, rng, seed=rep)
                base_json = base.to_json()
                for setting in SETTINGS:
                    course = configure(base_json, setting, truth, true_d, seed=rep)
                    acc[setting].append(evaluate(course, truth, true_d, n_test,
                                                 np.random.default_rng(777 + rep)))
                    if setting == "curve+difficulty":
                        fitted.append((course.curve.scale, course.curve.slip))
                        for k, v in enumerate(flag_quality(course, true_d, ids, bands)):
                            flags[k] += v
            for setting in SETTINGS:
                m = {k: np.mean([r[k] for r in acc[setting]]) for k in acc[setting][0]}
                extra = ""
                if setting == "curve+difficulty":
                    sc = np.mean([f[0] for f in fitted])
                    sl = np.mean([f[1] for f in fitted])
                    extra = f"scale {sc:.2f}, slip {sl:.2f}; flags {flags[0]}/{flags[1]} right, {flags[0]}/{flags[2]} found"
                print(f"{n:>8}  {setting:<17} {m['asked']:6.1f} {m['exact']*100:6.1f} {m['clear']*100:6.1f} "
                      f"{m['covers']*100:6.1f} {m['width']:5.2f}  {extra}")
    print(f"\nDone in {time.time() - t0:.0f}s ({reps} courses per row, {n_test} test students each)")
