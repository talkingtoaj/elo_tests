"""Check your question bank and your students from a spreadsheet of past answers.

    python examples/calibrate_from_csv.py answers.csv

answers.csv needs four columns (any order, any capitals):

    student, question, band, correct

`correct` can be 1/0, yes/no, true/false or right/wrong. One row per answer.

It writes three files next to a summary on screen:
    question_difficulty.csv   what the answers say about each question
    flagged_questions.csv     questions that look like they sit in the wrong band
    student_bands.csv         each student's band, with a range and a confidence
"""
import argparse
import csv
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from bandtest import Course  # noqa: E402

YES = {"1", "true", "yes", "y", "right", "correct", "t"}
NO = {"0", "false", "no", "n", "wrong", "incorrect", "f"}


def read_answers(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise SystemExit("The file is empty.")
        names = {n.strip().lower(): n for n in reader.fieldnames}
        missing = [c for c in ("student", "question", "band", "correct") if c not in names]
        if missing:
            raise SystemExit(f"Missing column(s): {', '.join(missing)}. "
                             f"Found: {', '.join(reader.fieldnames)}")
        rows = []
        for n, r in enumerate(reader, start=2):
            word = r[names["correct"]].strip().lower()
            if word in YES:
                ok = True
            elif word in NO:
                ok = False
            else:
                raise SystemExit(f"Row {n}: don't understand correct={word!r}. Use 1/0 or yes/no.")
            try:
                band = int(float(r[names["band"]]))
            except ValueError:
                raise SystemExit(f"Row {n}: band must be a whole number, got {r[names['band']]!r}.")
            rows.append((r[names["student"]].strip(), r[names["question"]].strip(), band, ok))
    return rows


def build_course(rows, bands=None):
    qband = {}
    for _, q, b, _ in rows:
        if q in qband and qband[q] != b:
            raise SystemExit(f"Question {q!r} is listed in band {qband[q]} and band {b}. "
                             "Each question needs one band.")
        qband[q] = b
    course = Course(bands or max(qband.values()), seed=0)
    for q, b in qband.items():
        course.add_question(q, b)
    for s, q, _, ok in rows:
        course.record_answer(s, q, ok)
    return course


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("answers", help="CSV file of past answers")
    ap.add_argument("--bands", type=int, help="number of bands (default: highest band in the file)")
    ap.add_argument("--out", default=".", help="folder for the result files (default: here)")
    args = ap.parse_args(argv)

    rows = read_answers(args.answers)
    course = build_course(rows, args.bands)
    students = {s for s, *_ in rows}
    report = course.fit()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    counts = {}
    for _, q, _, _ in rows:
        counts[q] = counts.get(q, 0) + 1
    flags = course.flagged_questions()
    flagged_ids = {f[0] for f in flags}

    write_csv(out / "question_difficulty.csv",
              ["question", "listed_band", "difficulty_in_bands", "answers", "flagged"],
              [[q, course._band[i], round(course.difficulty(q), 2), counts.get(q, 0),
                "yes" if q in flagged_ids else "no"] for i, q in enumerate(course._ids)])
    write_csv(out / "flagged_questions.csv",
              ["question", "listed_band", "looks_like_band", "answers"], flags)
    student_rows = []
    for s in sorted(students):
        est = course.start_student(s).estimate()
        student_rows.append([s, est.band, est.band_range[0], est.band_range[1],
                             round(est.confidence, 2), est.n_answered])
    write_csv(out / "student_bands.csv",
              ["student", "band", "range_low", "range_high", "confidence", "answers"], student_rows)

    c = report.curve
    print(f"Read {len(rows)} answers from {len(students)} students on {len(course._ids)} questions.")
    print(f"Answer curve: scale {c.scale}, slip {c.slip}  (library default: scale 0.5, slip 0.05)")
    if c.scale > 0.6:
        print("  Your answers are noisier than the default assumes, so tests will need more questions.")
    elif c.scale < 0.4:
        print("  Your answers are sharper than the default assumes, so tests can be shorter.")
    else:
        print("  Close to the default.")
    if report.fitted_difficulty:
        print(f"Fitted each question's difficulty ({len(students)} students is enough).")
        print(f"Flagged {len(flags)} question(s) that look like they sit in the wrong band.")
        for q, listed, looks, n in flags[:10]:
            print(f"  {q}: listed band {listed}, answers look like band {looks} ({n} answers)")
    else:
        print(f"Did not fit question difficulty: {len(students)} students is too few "
              "(about 150 are needed). Curve only.")
    print(f"Files written to {out.resolve()}")


if __name__ == "__main__":
    main()
