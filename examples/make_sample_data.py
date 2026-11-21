"""Make the sample answer file used by the cookbook and the browser demos.

Writes examples/sample_answers.csv and docs/sample-data.js (the same data, so the
browser demo can load it when opened straight from disk).

The fake course: 10 bands, 6 questions per band, 200 students who each answered
20 questions. The real curve is flatter than the library default, and 3 questions
sit in the wrong band on purpose, so there is something to find.
Run:  python examples/make_sample_data.py
"""
import csv
import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
BANDS, PER_BAND, STUDENTS, EACH = 10, 6, 200, 20
SCALE, SLIP, GUESS = 0.8, 0.10, 0.02
MISPLACED = {"q07": +2.0, "q33": -2.0, "q55": -2.0}   # question -> real shift in bands

rng = np.random.default_rng(2024)
questions = []   # (id, listed band, real difficulty)
for b in range(1, BANDS + 1):
    for k in range(PER_BAND):
        qid = f"q{(b - 1) * PER_BAND + k + 1:02d}"
        questions.append((qid, b, b - 0.5 + rng.normal(0, 0.25) + MISPLACED.get(qid, 0.0)))

rows = []
for i in range(STUDENTS):
    level = rng.uniform(0, BANDS)
    # a calibration run: two questions from every band
    picks = []
    for b in range(1, BANDS + 1):
        pool = [q for q in questions if q[1] == b]
        picks += [pool[j] for j in rng.choice(len(pool), 2, replace=False)]
    rng.shuffle(picks)
    for qid, band, d in picks[:EACH]:
        p = GUESS + (1 - GUESS - SLIP) / (1 + np.exp(-(level - d) / SCALE))
        rows.append((f"s{i + 1:03d}", qid, band, int(rng.random() < p)))

csv_path = ROOT / "examples" / "sample_answers.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["student", "question", "band", "correct"])
    w.writerows(rows)

text = csv_path.read_text()
(ROOT / "docs" / "sample-data.js").write_text("window.SAMPLE_CSV = " + json.dumps(text) + ";\n")
print(f"wrote {len(rows)} answers from {STUDENTS} students; misplaced on purpose: {sorted(MISPLACED)}")
