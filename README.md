# bandtest

**Find which band a student is in with as few questions as possible, and tell the teacher how sure you are.**

`bandtest` is a small Python library (numpy only) that your course software can plug into. Your software keeps its own questions, grading and screens. `bandtest` decides which question to ask next, when to stop, and what to report.

- **Adaptive.** Every question is picked to teach the most about this student. Strong students are not bored with easy questions and weak students are not crushed by hard ones.
- **Honest.** The result is a band, a range and a confidence. A student on the edge of two bands gets a range such as "band 4 to 5", not a coin flip dressed up as a fact.
- **Learns your course.** After a short calibration run it learns how clean your answers are, and (with enough students) which questions sit in the wrong band.
- **Easy to plug in.** Use it from Python, over a tiny web service, or from a spreadsheet. The only thing you store is one JSON file.

## See it in two minutes

Open these files in a web browser. They work offline, with nothing to install: download or clone the repo and double-click them. (On GitHub they show as code. Turn on GitHub Pages for the `docs` folder to get a link you can share.)

| Demo | What you can do |
|---|---|
| [`docs/demo-test.html`](docs/demo-test.html) | **Watch a test.** Run a simulated student, or answer the questions yourself, and watch the guess about their level narrow after every answer. |
| [`docs/demo-calibration.html`](docs/demo-calibration.html) | **Calibrate a made-up course.** Pick the class size, question bank and answer quality. See what the library learns, which questions it flags, and whether the test improves. |
| [`docs/try-your-data.html`](docs/try-your-data.html) | **Try your own data.** Drop in a spreadsheet of past answers and get a plain summary, flagged questions and a band for every student. Nothing leaves your browser. |

## Plug it into your course software

The **[Cookbook](COOKBOOK.md)** is written for people who are not programmers. It has four recipes:

1. **Browser.** Check your data with no install.
2. **One command.** A spreadsheet in, three result files out.
3. **Web service.** One command starts it. Your platform talks to it with plain web requests.
4. **Python.** Ten lines.

## Quick start

```python
from bandtest import Course

course = Course(bands=10)
course.add_question("q17", band=4)           # your questions, each with a band (1 = easiest)

s = course.start_student("s42")              # mode="calibrate" for the calibration run
while (q := s.next_question()) is not None:  # None means the test is finished
    s.record(q, correct=your_platform_grades(q))
print(s.estimate())                          # band, band_range, confidence, done

course.fit()                                 # learn the curve (and each question, with 150+ students)
print(course.flagged_questions())            # questions whose answers fit another band
saved = course.to_json()                     # the only state you store
course = Course.from_json(saved)             # students half way through simply resume
```

Needs Python 3.10+ and `pip install numpy`. Run `python -m pytest tests` to check everything.

## How the test works

1. A student's level is a point on a line from 0 to the number of bands. Band 3 covers 2 to 3.
2. The library keeps a **belief curve** over that line, starting wide. After each answer it updates the curve with Bayes' rule. A right answer to a hard question moves the belief up a lot. A right answer to an easy one barely moves it.
3. The next question is the one expected to **shrink the uncertainty about the band** the most. In practice that is a question near where the student seems to be.
4. It stops when one band reaches 90% confidence, or the student is clearly pinned on the edge of two bands, or after 25 questions.
5. **Calibration** is a separate mode where each student gets about 20 questions spread evenly over every band. `fit()` then learns how sharp the right/wrong curve is, how often a strong student slips, and (with enough students) each question's real difficulty. A question is only moved a little by a few answers, so one odd result cannot wreck it.

## What we measured

All numbers come from simulated students (10 bands, 100 questions, 20 calibration questions each), so treat them as a guide to how the method behaves, not as a promise for your course. Every table is reproducible with the scripts named below. Raw output is in [`results/`](results/).

**What a small course can learn.** ([`simulate_calibration.py`](simulate_calibration.py), 6 courses per row)

| Calibration students | Learn the curve (2 numbers) | Learn each question's difficulty | Flag misplaced questions |
|---|---|---|---|
| 30 | Yes, safe | No: exact band got 2 to 9 points worse than learning the curve alone | No: about half of the few flags were right, and it found only 5 to 26% of bad questions |
| 100 | Yes | Mixed: about 7 fewer questions, but 2 to 8 points less exact-band accuracy | Half to two thirds of flags right, finds 30 to 75% of bad questions (varied a lot between runs) |
| 300 | Yes | Yes for clean or typical answers: about 7 fewer questions for 0 to 4 points of accuracy. No gain when answers are noisy. | Good: 69 to 86% of flags right, finds 82 to 98% |

`course.fit()` follows this by default: it learns each question's difficulty only when there are at least 150 students.

- **Learning the curve keeps the test honest.** When real answers were noisier than the default assumed, the test claimed its range held the true band 90% of the time but it was only right 83 to 85% of the time. After learning the curve from just 30 students it was right 93 to 95% of the time.
- **Answer quality sets the test length.** With a perfectly known curve the test needed about 9 questions for clean answers, about 18 for typical answers, and about 24 for very noisy ones (the limit is 25).
- **Exact band is hard at the edges.** Even with perfect settings, about 79% of all students land in the exact band, and about 90% of students who are not near an edge. The reported range held the true band 95 to 97% of the time.

**Does a smaller question bank help a 30-student course?** ([`simulate_bank_size.py`](simulate_bank_size.py)) Short answer: it helps flagging but hurts the test, so keep the bank large.

| Bank size | Answers per question | Best possible exact band | Flags that were right |
|---|---|---|---|
| 100 questions (10 per band) | 6 | 78% | 42% |
| 50 (5 per band) | 12 | 74% | 67% |
| 30 (3 per band) | 20 | 65% | 75% |
| 20 (2 per band) | 30 | 58% | 75% |

(30 students, answers matching the default curve. The other two curves show the same pattern.) A smaller bank gives each question more answers and fewer chances for a false alarm, so flags become more trustworthy. But the test has fewer questions to choose from, so even perfect settings get worse, and tests with sharp answers needed more than twice as many questions. **Use at least 10 questions per band (the largest we tested), and learn the curve only, unless your course has more than about 150 students.**

## Honest limits

- **Simulated students only so far.** The first thing to do with a real course is run Recipe 1 on your own past answers and see whether the learned curve looks sensible.
- **Right or wrong answers only.** Partial credit is not supported. Grading is assumed to be correct. How it is graded is the host platform's job.
- **One skill per course.** A course is treated as one line from easy to hard.
- **Questions in a band are assumed to be about equally hard.** The library learns how far each is from its band, but only with enough students.
- **The example web service has no password.** It listens on your own computer only. Put your own login in front of it before exposing it.

## What is in this repo

| Path | What it is |
|---|---|
| `bandtest/` | The library |
| `docs/` | The three interactive demos, plus a browser copy of the library (`bandtest.js`) |
| `COOKBOOK.md` | Step by step recipes for people who are not programmers |
| `examples/calibrate_from_csv.py` | Spreadsheet in, results out |
| `examples/server.py` | Small web service so any platform can use the library |
| `examples/sample_answers.csv` | 4,000 made-up answers to try things on |
| `simulate_calibration.py`, `simulate_bank_size.py` | The studies above |
| `tests/` | Automated tests |
| `elo.py`, `bayes.py`, `compare.py`, `robustness.py` | The earlier Elo experiments (below) |

## Where this started: the Elo experiments

This repo began as a sandbox for finding a student's level with Elo ratings, tuning the starting variance and variance decay with Bayesian optimization.

- **Setup.** 10 levels, each 1000 Elo points wide. A user counts as converged when their guess is within 500 of the truth and the variance is below 500.
- **Result.** Tuned Elo got there in about 6 rounds, with 85% of users within 500 points.
- **Bayesian version.** `bayes.py` replaces the step-size schedule with a belief curve. On the same simulated students (`compare.py`) it was 96% accurate after 6 questions, and reached 87% after 4, where Elo needed 6 questions for 85%.
- **Mismatched models.** `robustness.py` tests what happens when real students do not follow the assumed curve. The Bayesian version stayed ahead in every case, by a smaller margin when answers were very noisy. That is why `bandtest` learns the curve from calibration data.

`bandtest` is the result of that work: the same belief-curve idea, made exact-band, with calibration, and packaged to plug into other platforms.
