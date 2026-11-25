"""Does a smaller question bank help a small course fit question difficulty?

Same study as simulate_calibration.py, but the bank size changes. Students
still answer 20 calibration questions each, so a smaller bank means each
question gets more answers (30 students x 20 answers spread over the bank).

Run:  python simulate_bank_size.py [reps] [test_students]
"""
import sys
import time

import numpy as np

from simulate_calibration import (TRUTHS, BANDS, make_world, calibrate, configure,
                                  evaluate, flag_quality)

SETTINGS = ["defaults", "curve only", "curve+difficulty", "oracle"]
PER_BAND = [10, 5, 3, 2]       # bank of 100, 50, 30, 20 questions
SIZES = [30, 100]

if __name__ == "__main__":
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    n_test = int(sys.argv[2]) if len(sys.argv) > 2 else 150
    t0 = time.time()
    for t_name, truth in TRUTHS.items():
        print(f"\n=== Truth: {t_name} ===")
        print(f"{'students':>8} {'bank':>5} {'ans/q':>5}  {'setup':<17} {'asked':>6} {'exact%':>6} "
              f"{'clear%':>6} {'range%':>6} {'width':>5}  flags")
        for n in SIZES:
            for pb in PER_BAND:
                bank = pb * BANDS
                acc = {s: [] for s in SETTINGS}
                flags = [0, 0, 0]
                for rep in range(reps):
                    rng = np.random.default_rng(1000 * rep + n + pb)
                    ids, bands, true_d = make_world(rng, per_band=pb)
                    base_json = calibrate(n, truth, ids, bands, true_d, rng, seed=rep).to_json()
                    for setting in SETTINGS:
                        course = configure(base_json, setting, truth, true_d, seed=rep)
                        acc[setting].append(evaluate(course, truth, true_d, n_test,
                                                     np.random.default_rng(777 + rep)))
                        if setting == "curve+difficulty":
                            for k, v in enumerate(flag_quality(course, true_d, ids, bands)):
                                flags[k] += v
                for setting in SETTINGS:
                    m = {k: np.mean([r[k] for r in acc[setting]]) for k in acc[setting][0]}
                    extra = ""
                    if setting == "curve+difficulty":
                        extra = f"{flags[0]}/{flags[1]} right, {flags[0]}/{flags[2]} found"
                    print(f"{n:>8} {bank:>5} {n*20/bank:5.0f}  {setting:<17} {m['asked']:6.1f} "
                          f"{m['exact']*100:6.1f} {m['clear']*100:6.1f} {m['covers']*100:6.1f} "
                          f"{m['width']:5.2f}  {extra}")
    print(f"\nDone in {time.time() - t0:.0f}s ({reps} courses per row, {n_test} test students each)")
