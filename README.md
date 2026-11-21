A sandbox for running ELO contests where a user converges on their true value.

Scenario:

We have 10 levels, these are represented by 
- level 1 == ELOs 1..1000,
- level 2 == ELOs 1001..2000,
- ...
- level 10 == ELOs 9001..10000

We want a user of unknown starting ELO to converge on their true ELO. We consider a user converged if
1. Their guessed ELO is within +- 500 of their true ELO
2. Their variance is less than 500

By running a simulation with varying starting parameters, we can determine which are the best at convergence.

Starting parameters include:
- starting variance,
- variance decay,

Since we might have an idea of the true ELO, we can include a guess of the true ELO.

Current results show convergence can be reliably achieved with around 6 rounds.

## bandtest: the library

`bandtest/` is a plain Python library (numpy only) that other platforms plug into. The host owns the questions, the grading and the screens. The library tracks student level and question difficulty, picks the next question, and says when to stop.

```python
from bandtest import Course

course = Course(bands=10)
course.add_question("q17", band=4)           # host registers each question

s = course.start_student("s42")              # adaptive; mode="calibrate" for the 20-question run
while (q := s.next_question()) is not None:
    s.record(q, correct=host_grades(q))      # host reports right or wrong
est = s.estimate()                           # band, band_range, confidence, done

course.fit()                                 # learn the curve and each question's difficulty
course.flagged_questions()                   # questions whose answers fit another band
saved = course.to_json()                     # the only state the host stores
course = Course.from_json(saved)             # students in progress resume from the log
```

- A student's level is a number from 0 to `bands` (band b covers b-1 to b). The library keeps a belief curve over it and updates with Bayes' rule after each answer.
- The next question is the one that most reduces uncertainty about the band.
- The test stops when one band reaches 90% confidence, when the student is clearly on the edge of two bands, or at the question cap.
- `fit()` learns how sharp the right/wrong curve is, how often strong students slip, and each question's real difficulty (pulled back toward its band so a few answers can't move it far).

Tests: `python -m pytest tests`. Fake-data study of what a small course can learn: `python simulate_calibration.py`.
