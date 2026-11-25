import numpy as np
import pytest

from bandtest import Course, Curve

TRUE = Curve(scale=0.5, slip=0.05, guess=0.02)


def make_course(bands=10, per_band=10, seed=0):
    course = Course(bands, seed=seed)
    for b in range(1, bands + 1):
        for k in range(per_band):
            course.add_question(f"b{b}-{k}", b)
    return course


def answer(course, qid, level, rng, curve=TRUE):
    d = course.difficulty(qid)
    p = curve.guess + (1 - curve.guess - curve.slip) / (1 + np.exp(-(level - d) / curve.scale))
    return bool(rng.random() < p)


def run_student(course, sid, level, rng, **kw):
    s = course.start_student(sid, **kw)
    while (q := s.next_question()) is not None:
        s.record(q, answer(course, q, level, rng))
    return s.estimate()


def test_adaptive_finds_the_band_for_clear_cases():
    rng = np.random.default_rng(1)
    course = make_course()
    hits = 0
    for i, level in enumerate([0.5, 2.5, 4.5, 6.5, 8.5, 9.5] * 5):
        est = run_student(course, f"s{i}", level, rng)
        hits += est.band == int(level) + 1
    assert hits >= 26  # at least 26 of 30 land in the exact band


def test_calibrate_mode_asks_n_questions_covering_all_bands():
    rng = np.random.default_rng(2)
    course = make_course()
    s = course.start_student("a", mode="calibrate", n_calibration=20)
    bands = []
    while (q := s.next_question()) is not None:
        bands.append(int(q.split("-")[0][1:]))
        s.record(q, answer(course, q, 5.0, rng))
    assert len(bands) == 20
    assert sorted(set(bands)) == list(range(1, 11))
    assert all(bands.count(b) == 2 for b in range(1, 11))


def test_json_round_trip_and_resume():
    rng = np.random.default_rng(3)
    course = make_course(bands=5, per_band=6)
    s = course.start_student("x")
    for _ in range(3):
        q = s.next_question()
        s.record(q, answer(course, q, 2.0, rng))
    before = s.estimate()
    restored = Course.from_json(course.to_json())
    resumed = restored.start_student("x")
    after = resumed.estimate()
    assert after.n_answered == 3
    assert after.band == before.band
    assert np.allclose(after.band_probs, before.band_probs)


def test_same_question_cannot_be_answered_twice():
    course = make_course(bands=3, per_band=3)
    s = course.start_student("x")
    q = s.next_question()
    s.record(q, True)
    with pytest.raises(ValueError):
        s.record(q, True)


def test_fit_learns_a_flatter_curve_and_flags_a_misplaced_question():
    rng = np.random.default_rng(4)
    truth = Curve(scale=1.0, slip=0.1, guess=0.02)
    course = make_course(bands=6, per_band=8, seed=5)
    # q "b2-0" is nominally band 2 but is really as hard as a band 5 question
    true_d = {q: course.difficulty(q) for q in course._ids}
    true_d["b2-0"] = 4.5
    for i in range(300):
        level = rng.uniform(0, 6)
        s = course.start_student(f"s{i}", mode="calibrate", n_calibration=20)
        while (q := s.next_question()) is not None:
            p = truth.guess + (1 - truth.guess - truth.slip) / (1 + np.exp(-(level - true_d[q]) / truth.scale))
            s.record(q, bool(rng.random() < p))
    report = course.fit()
    assert report.curve.scale >= 0.8
    assert report.curve.slip >= 0.05
    assert any(f[0] == "b2-0" for f in course.flagged_questions())


def test_fit_needs_data():
    with pytest.raises(ValueError):
        make_course(bands=3, per_band=3).fit()
