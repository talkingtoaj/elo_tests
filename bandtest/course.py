"""Adaptive band placement as a plain Python library (numpy only).

The host platform owns the questions, the grading and the screens. This
library only does the maths:

    course = Course(bands=10)
    course.add_question("q17", band=4)

    s = course.start_student("s42")
    while (q := s.next_question()) is not None:
        s.record(q, correct=host_grades(q))
    print(s.estimate())

    course.fit()                    # learn the curve and question difficulty
    course.flagged_questions()      # questions whose answers fit another band
    text = course.to_json()         # the only state the host has to store

How it works: a student's level is a number from 0 to `bands` (band b covers
b-1 up to b). We keep a belief curve over that number and update it with
Bayes' rule after every answer. The chance of a right answer is

    guess + (1 - guess - slip) * sigmoid((level - difficulty) / scale)

A question starts with its band's centre as difficulty. `fit()` learns the
curve (scale, slip) and moves each question's difficulty if the answers say
so, with a pull back toward its band so a few answers cannot move it far.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field

import numpy as np

POINTS_PER_BAND = 10  # grid points per band for the belief curve
_EPS = 1e-6
AUTO_DIFFICULTY_STUDENTS = 150  # rule of thumb from simulate_calibration.py


@dataclass
class Curve:
    """How likely a right answer is, as a function of level minus difficulty."""
    scale: float = 0.5   # smaller means a sharper step from wrong to right
    slip: float = 0.05   # chance a student who knows it still gets it wrong
    guess: float = 0.02  # chance of a right answer by luck (near 0 for typed answers)


@dataclass
class Estimate:
    band: int                      # most likely band, 1..bands
    band_range: tuple              # smallest run of bands holding `range_confidence`
    confidence: float              # chance the student is in `band`
    range_confidence: float        # chance the student is inside `band_range`
    n_answered: int
    done: bool
    band_probs: list = field(default_factory=list)


@dataclass
class FitReport:
    curve: Curve
    n_responses: int
    n_students: int
    log_likelihood: float
    fitted_difficulty: bool = False


def _curve_matrix(curve, difficulty, grid):
    """P(right) for each question difficulty (rows) at each level (columns)."""
    z = (grid[None, :] - np.asarray(difficulty)[:, None]) / curve.scale
    p = curve.guess + (1 - curve.guess - curve.slip) / (1 + np.exp(-z))
    return np.clip(p, _EPS, 1 - _EPS)


class Course:
    def __init__(self, bands: int, curve: Curve | None = None, seed: int | None = None):
        if bands < 2:
            raise ValueError("a course needs at least 2 bands")
        self.bands = bands
        self.curve = curve or Curve()
        self.rng = np.random.default_rng(seed)
        self._grid = (np.arange(bands * POINTS_PER_BAND) + 0.5) / POINTS_PER_BAND
        self._ids: list[str] = []
        self._index: dict[str, int] = {}
        self._band: list[int] = []
        self._offset: list[float] = []
        self._log: list[tuple] = []                  # (student_id, question index, correct)
        self._by_student: dict[str, list] = {}       # student_id -> [(question index, correct)]
        self._pc_cache = None
        self._version = 0                            # bumps when curve or difficulty change

    # ---- questions -------------------------------------------------------
    def add_question(self, question_id: str, band: int) -> None:
        if not 1 <= band <= self.bands:
            raise ValueError(f"band must be 1..{self.bands}")
        if question_id in self._index:
            raise ValueError(f"question {question_id!r} already exists")
        self._index[question_id] = len(self._ids)
        self._ids.append(question_id)
        self._band.append(band)
        self._offset.append(0.0)
        self._pc_cache = None

    def difficulty(self, question_id: str) -> float:
        """Current difficulty in band units (band centre plus learned offset)."""
        i = self._index[question_id]
        return self._band[i] - 0.5 + self._offset[i]

    def set_curve(self, curve: Curve) -> None:
        self.curve = curve
        self._touch()

    def set_difficulty(self, question_id: str, difficulty: float) -> None:
        i = self._index[question_id]
        self._offset[i] = difficulty - (self._band[i] - 0.5)
        self._touch()

    def _touch(self):
        self._pc_cache = None
        self._version += 1

    def _difficulties(self):
        return np.array(self._band, dtype=float) - 0.5 + np.array(self._offset)

    def _pc(self):
        if self._pc_cache is None or self._pc_cache.shape[0] != len(self._ids):
            self._pc_cache = _curve_matrix(self.curve, self._difficulties(), self._grid)
        return self._pc_cache

    # ---- students --------------------------------------------------------
    def start_student(self, student_id: str, prior_band: int | None = None,
                      mode: str = "adaptive", n_calibration: int = 20,
                      confidence: float = 0.9, max_questions: int = 25,
                      range_mass: float = 0.9, allow_borderline: bool = True) -> "Session":
        """Begin, or resume, a student. Earlier answers in the log are replayed.

        mode "calibrate": stratified questions (every band covered evenly), no
        early stop. Use it to gather data for fit().
        mode "adaptive": always ask the most informative question; stop when
        one band reaches `confidence`, or (if `allow_borderline`) when two
        neighbouring bands together reach it and the level is pinned down to
        within about 0.3 of a band (a borderline student), or after
        `max_questions`.
        """
        if mode not in ("adaptive", "calibrate"):
            raise ValueError("mode must be 'adaptive' or 'calibrate'")
        return Session(self, student_id, prior_band, mode, n_calibration,
                       confidence, max_questions, range_mass, allow_borderline)

    def record_answer(self, student_id: str, question_id: str, correct: bool) -> None:
        """Add one past answer to the log (for importing history, no test needed)."""
        if question_id not in self._index:
            raise ValueError(f"unknown question {question_id!r}; add_question() it first")
        q = self._index[question_id]
        if any(q == qq for qq, _ in self._by_student.get(student_id, [])):
            raise ValueError(f"{student_id!r} already answered {question_id!r}")
        self._record(student_id, q, correct)

    def _record(self, student_id, qidx, correct):
        self._log.append((student_id, qidx, bool(correct)))
        self._by_student.setdefault(student_id, []).append((qidx, bool(correct)))

    # ---- fitting ---------------------------------------------------------
    def _data(self):
        students = {s: i for i, s in enumerate(self._by_student)}
        s_idx = np.array([students[s] for s, _, _ in self._log], dtype=int)
        q_idx = np.array([q for _, q, _ in self._log], dtype=int)
        y = np.array([c for _, _, c in self._log], dtype=bool)
        return s_idx, q_idx, y, len(students)

    @staticmethod
    def _student_loglik(pc, s_idx, q_idx, y, n_students):
        logp = np.where(y[:, None], np.log(pc[q_idx]), np.log1p(-pc[q_idx]))
        out = np.zeros((n_students, pc.shape[1]))
        np.add.at(out, s_idx, logp)
        return out

    @staticmethod
    def _logsumexp(a, axis):
        m = a.max(axis=axis, keepdims=True)
        return (m + np.log(np.exp(a - m).sum(axis=axis, keepdims=True))).squeeze(axis)

    def fit(self, fit_curve: bool = True, fit_difficulty: bool | None = None,
            iterations: int = 4, tau: float = 0.5) -> FitReport:
        """Learn from the answers logged so far.

        fit_curve: learn scale and slip (safe at any course size).
        fit_difficulty: learn each question's offset from its band centre.
        None (the default) means "only if there are at least 150 students",
        because the fake-data study found that fewer students make the test
        worse, not better. tau sets how strongly a question is pulled back
        toward its band (smaller is stronger). Population level is taken as
        even across bands.
        """
        if not self._log:
            raise ValueError("no answers recorded yet")
        s_idx, q_idx, y, n_students = self._data()
        if fit_difficulty is None:
            fit_difficulty = n_students >= AUTO_DIFFICULTY_STUDENTS
        G = len(self._grid)
        log_prior = np.full(G, -np.log(G))
        offsets = np.array(self._offset)
        bands = np.array(self._band)
        scales = [0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 1.0, 1.3]
        slips = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2]
        curve = self.curve

        def marginal(cv, off):
            pc = _curve_matrix(cv, bands - 0.5 + off, self._grid)
            ll = self._student_loglik(pc, s_idx, q_idx, y, n_students)
            return float(self._logsumexp(ll + log_prior, axis=1).sum()), pc, ll

        rounds = iterations if fit_difficulty else 1
        for _ in range(rounds):
            if fit_curve:
                best = None
                for sc in scales:
                    for sl in slips:
                        cand = Curve(sc, sl, curve.guess)
                        score = marginal(cand, offsets)[0]
                        if best is None or score > best[0]:
                            best = (score, cand)
                curve = best[1]
            if fit_difficulty:
                _, pc, ll = marginal(curve, offsets)
                post = np.exp(ll + log_prior - self._logsumexp(ll + log_prior, axis=1)[:, None])
                w1 = np.zeros((len(self._ids), G))
                w0 = np.zeros((len(self._ids), G))
                np.add.at(w1, q_idx[y], post[s_idx[y]])
                np.add.at(w0, q_idx[~y], post[s_idx[~y]])
                cand_off = np.arange(-3.0, 3.01, 0.1)
                log_pen = -np.log1p((cand_off / tau) ** 2)
                tables = {}
                for b in range(1, self.bands + 1):
                    p = _curve_matrix(curve, b - 0.5 + cand_off, self._grid)
                    tables[b] = (np.log(p), np.log1p(-p))
                answered = np.flatnonzero((w1.sum(1) + w0.sum(1)) > 0)
                for q in answered:
                    lp, lq = tables[bands[q]]
                    score = lp @ w1[q] + lq @ w0[q] + log_pen
                    offsets[q] = cand_off[int(np.argmax(score))]
        total, _, _ = marginal(curve, offsets)
        self.curve = curve
        self._offset = [float(o) for o in offsets]
        self._touch()
        return FitReport(curve, len(self._log), n_students, total, bool(fit_difficulty))

    def flagged_questions(self, min_answers: int = 8, min_shift: float = 1.0):
        """Questions whose answers look like they belong to another band.

        Returns (question_id, nominal_band, apparent_band, n_answers), biggest
        shift first. Only questions with at least `min_answers` are judged.
        """
        counts = np.zeros(len(self._ids), dtype=int)
        for _, q, _ in self._log:
            counts[q] += 1
        out = []
        for i, qid in enumerate(self._ids):
            if counts[i] >= min_answers and abs(self._offset[i]) >= min_shift:
                apparent = int(np.clip(np.floor(self._band[i] - 0.5 + self._offset[i]) + 1, 1, self.bands))
                if apparent != self._band[i]:
                    out.append((qid, self._band[i], apparent, int(counts[i])))
        out.sort(key=lambda t: -abs(t[2] - t[1]))
        return out

    # ---- saving ----------------------------------------------------------
    def to_json(self) -> str:
        return json.dumps({
            "bands": self.bands,
            "curve": asdict(self.curve),
            "questions": [[q, b, o] for q, b, o in zip(self._ids, self._band, self._offset)],
            "log": [[s, q, int(c)] for s, q, c in self._log],
        })

    @classmethod
    def from_json(cls, text: str, seed: int | None = None) -> "Course":
        d = json.loads(text)
        course = cls(d["bands"], Curve(**d["curve"]), seed=seed)
        for qid, band, offset in d["questions"]:
            course.add_question(qid, band)
            course._offset[-1] = offset
        for s, q, c in d["log"]:
            course._record(s, q, bool(c))
        return course


class Session:
    """One student's test. Built from the course log, so it can be resumed."""

    def __init__(self, course, student_id, prior_band, mode, n_calibration,
                 confidence, max_questions, range_mass, allow_borderline=True):
        self.course = course
        self.student_id = student_id
        self.mode = mode
        self.n_calibration = n_calibration
        self.confidence = confidence
        self.max_questions = max_questions
        self.range_mass = range_mass
        self.allow_borderline = allow_borderline
        self.borderline_sd = 0.3  # level uncertainty (in bands) that counts as "pinned down"
        grid = course._grid
        if prior_band is None:
            self._log_prior = np.zeros_like(grid)
        else:
            # soft prior: about 1.5 bands wide around the guess
            self._log_prior = -0.5 * ((grid - (prior_band - 0.5)) / 1.5) ** 2
        self._answers = list(course._by_student.get(student_id, []))
        self._seen = {q for q, _ in self._answers}
        self._logpost = None
        self._version = -1
        self._order: list[int] = []  # calibrate mode: band order, refilled as used

    # -- belief
    def _ensure(self):
        c = self.course
        if self._logpost is None or self._version != c._version:
            pc = c._pc()
            lp = self._log_prior.copy()
            for q, ok in self._answers:
                lp += np.log(pc[q]) if ok else np.log1p(-pc[q])
            self._logpost = lp
            self._version = c._version

    def _post(self):
        self._ensure()
        p = np.exp(self._logpost - self._logpost.max())
        return p / p.sum()

    def record(self, question_id: str, correct: bool) -> None:
        c = self.course
        if question_id not in c._index:
            raise ValueError(f"unknown question {question_id!r}; add_question() it first")
        q = c._index[question_id]
        if q in self._seen:
            raise ValueError(f"{question_id!r} was already answered by this student")
        self._ensure()
        pc = c._pc()[q]
        self._logpost = self._logpost + (np.log(pc) if correct else np.log1p(-pc))
        self._answers.append((q, bool(correct)))
        self._seen.add(q)
        c._record(self.student_id, q, correct)

    # -- result
    def estimate(self) -> Estimate:
        c = self.course
        B = c.bands
        probs = self._post().reshape(B, POINTS_PER_BAND).sum(1)
        top = int(np.argmax(probs))
        lo = hi = top
        mass = float(probs[top])
        while mass < self.range_mass and (lo > 0 or hi < B - 1):
            left = probs[lo - 1] if lo > 0 else -1
            right = probs[hi + 1] if hi < B - 1 else -1
            if left >= right:
                lo -= 1
                mass += float(left)
            else:
                hi += 1
                mass += float(right)
        n = len(self._answers)
        out_of_questions = len(self._seen) >= len(c._ids)
        if self.mode == "calibrate":
            done = n >= self.n_calibration or out_of_questions
        else:
            pair = float((probs[:-1] + probs[1:]).max())
            post = self._post()
            level_sd = float(np.sqrt((post * c._grid ** 2).sum() - (post * c._grid).sum() ** 2))
            # borderline = we know the level well, and it straddles a band edge
            borderline = self.allow_borderline and pair >= self.confidence and level_sd <= self.borderline_sd
            sure = probs[top] >= self.confidence or borderline
            done = sure or n >= self.max_questions or out_of_questions
        return Estimate(top + 1, (lo + 1, hi + 1), float(probs[top]), min(mass, 1.0),
                        n, bool(done), [float(p) for p in probs])

    # -- next question
    def next_question(self):
        """Id of the question to ask next, or None when the test is done."""
        if self.estimate().done:
            return None
        c = self.course
        unseen = np.array([i for i in range(len(c._ids)) if i not in self._seen])
        if self.mode == "calibrate":
            return c._ids[self._pick_stratified(unseen)]
        return c._ids[self._pick_informative(unseen)]

    def _pick_stratified(self, unseen):
        c = self.course
        bands = np.array(c._band)[unseen]
        while True:
            if not self._order:
                self._order = list(c.rng.permutation(np.arange(1, c.bands + 1)))
            want = self._order.pop()
            pool = unseen[bands == want]
            if len(pool):
                return int(c.rng.choice(pool))
            if not any(len(unseen[bands == b]) for b in range(1, c.bands + 1)):
                return int(c.rng.choice(unseen))
            # that band is used up; try the next one in the order

    def _pick_informative(self, unseen):
        """Question with the biggest expected drop in band uncertainty."""
        c = self.course
        B = c.bands
        post = self._post()
        pcs = c._pc()[unseen]
        p_right = pcs @ post
        right = (pcs * post).reshape(len(unseen), B, POINTS_PER_BAND).sum(2)
        wrong = ((1 - pcs) * post).reshape(len(unseen), B, POINTS_PER_BAND).sum(2)

        def entropy(mass, total):
            p = mass / total[:, None]
            return -(p * np.log(p + 1e-12)).sum(1)

        expected = p_right * entropy(right, p_right) + (1 - p_right) * entropy(wrong, 1 - p_right)
        expected = expected + c.rng.random(len(unseen)) * 1e-9  # random tie-break
        return int(unseen[int(np.argmin(expected))])
