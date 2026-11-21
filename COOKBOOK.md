# Cookbook: plug bandtest into your course software

This is for teachers, course coordinators and the person who looks after your course website. You do not need to understand the maths. Pick the recipe that fits you and follow the steps.

## The idea in one minute

Your course software already knows the questions, grades the answers and shows the screens. `bandtest` does only the thinking:

- Which question should this student get next?
- How sure are we about their band?
- Which band are they in?

They talk like this:

```
Your course software                         bandtest
--------------------                         --------
"Ann is starting a test"           ------->
                                   <-------  "Ask her question q17"
shows q17, grades it
"Ann got q17 right"                ------->
                                   <-------  "Band 5, maybe 4 to 6. Keep going."
...a few more rounds...
                                   <-------  "Done. Band 5, 91% sure."
```

`bandtest` never sees the question text or the student's real answer. It only needs a question ID, the band you gave that question, and whether the answer was right or wrong.

## Before you start

Collect these four things.

1. **A list of questions, each with a band.** Band 1 is the easiest. Give every question an ID like `q17` and a band number. Aim for **at least 10 questions per band**. A small bank makes the test worse (see the [bank size results](README.md#what-we-measured)). Try to keep questions in the same band at roughly the same effort.
2. **Right or wrong grading.** Your software does this. Typed answers, multiple choice, anything, as long as the result is right or wrong.
3. **Anonymous student IDs.** Use `s0042`, not names. The library only needs a label.
4. **A group of students for calibration.** Each answers about 20 questions covering all bands. At least 30 students. About 150 or more is much better, because then the library can also check each question.

## Pick your recipe

| I want to... | Recipe | What you need |
|---|---|---|
| See how it works, or check my own data, with nothing to install | [1. Browser](#recipe-1-check-your-data-in-your-browser) | A web browser |
| Check my question bank and get a band for each student from a spreadsheet | [2. One command](#recipe-2-one-command-on-your-computer) | Python, and 10 minutes |
| Let my course platform ask for questions and send back answers | [3. Web service](#recipe-3-let-your-platform-talk-to-it) | Someone who can run a program on a server |
| Use it inside my own Python code | [4. Python](#recipe-4-use-it-from-python) | A developer |

Not sure? Start with Recipe 1.

---

## Recipe 1: Check your data in your browser

Nothing is installed and nothing is uploaded. The page runs on your computer.

1. **Export your past answers as a CSV file** from your course software or spreadsheet. It needs four columns, one row per answer:

   | student | question | band | correct |
   |---|---|---|---|
   | s001 | q19 | 4 | 1 |
   | s001 | q06 | 1 | 1 |
   | s001 | q49 | 9 | 0 |

   `correct` can be `1/0`, `yes/no`, `true/false` or `right/wrong`. The column names can be in any order and any capitals.
   *In Excel, Numbers or Google Sheets: File, then Save As or Download, then CSV.*
2. **Open `docs/try-your-data.html`** by double-clicking it. It opens in your browser.
3. **Choose your file** (or paste the rows), then press **Analyse**.
4. **Read the results.** You get a plain summary, a list of questions that look misplaced, and a band for every student. Three buttons download the same results as CSV files.

No data yet? Press **Use the sample file** to see what the results look like.

If something is wrong with your file, the page tells you which row and what to fix.

---

## Recipe 2: One command on your computer

Same results as Recipe 1, but as files you can keep, and it handles large files.

**One-time setup** (ask your IT person if you are unsure):

```
pip install numpy
```

Download this repository and open a terminal in its folder.

**Run it:**

```
python examples/calibrate_from_csv.py my_answers.csv --out results
```

Add `--bands 8` if your course has more bands than appear in the file.

**What you get** in the `results` folder:

| File | What it tells you |
|---|---|
| `student_bands.csv` | For each student: `band`, `range_low` and `range_high`, `confidence` (0 to 1), and how many `answers` it used. A student near the edge of two bands gets a range, for example 4 to 5. |
| `flagged_questions.csv` | Questions whose answers look like they belong in a different band: the `listed_band` and the `looks_like_band`. Only appears when you have about 150 students. |
| `question_difficulty.csv` | Every question with the difficulty the answers suggest, and whether it is flagged. |

**What to do with a flagged question:** read it first. Often the answer key is wrong, or the wording is confusing. If it is fine, move it to the band the data suggests. The shift shown is cautious, so the real gap may be bigger.

---

## Recipe 3: Let your platform talk to it

This is the one for running real tests. You start a small program. Your platform then asks it questions over the web, using plain requests. Most platforms can do this under names like *webhook*, *HTTP request*, *external tool* or *API call*. Tools like Zapier and Make can too.

### Start it

```
python examples/server.py --bands 10 --state course.json --port 8080
```

- `--bands` is the number of bands in your course.
- `--state` is the file where everything is saved. Stop and restart it any time. Nothing is lost.

> **Keep it private.** It listens on your own computer only and has no password. If another computer needs to reach it, put your own login in front of it first. Do not put it on the open internet as it is.

### The conversation

Below is each request and a real reply. `curl` is a command that sends a web request. Your platform sends the same thing in its own way.

**1. Tell it your questions** (once, and again when you add questions):

```
curl -X POST localhost:8080/questions -d '{"questions":[{"id":"q1","band":1},{"id":"q2","band":1},{"id":"q3","band":2}]}'
```
```
{"questions": 3}
```

**2. Ask what to give a student next:**

```
curl -X POST localhost:8080/students/ann/next -d '{}'
```
```
{"question": "q6", "estimate": {"band": 1, "band_range": [1, 5], "confidence": 0.2, "n_answered": 0, "done": false}}
```

Show `q6` to Ann. (Before any answers, the estimate is just a placeholder. Ignore it while `n_answered` is 0.)

**3. Tell it how she did:**

```
curl -X POST localhost:8080/students/ann/answer -d '{"question":"q6","correct":true}'
```
```
{"estimate": {"band": 5, "band_range": [3, 5], "confidence": 0.38, "n_answered": 1, "done": false}}
```

**4. Repeat steps 2 and 3** until step 2 says `"question": null` or `"done": true`. Then show the result:

```
curl localhost:8080/students/ann
```

| Field | Meaning |
|---|---|
| `band` | The most likely band |
| `band_range` | The bands the student could be in, for example `[4, 5]` for a borderline student |
| `confidence` | Chance the student is in `band` (0 to 1) |
| `done` | `true` when the test is finished |

### Calibration first, then live tests

The library needs real answers before it can learn how your course behaves.

1. **Calibration phase.** For your first group of students, add `"mode": "calibrate"` to step 2:
   `curl -X POST localhost:8080/students/ann/next -d '{"mode":"calibrate"}'`
   Each student gets 20 questions covering every band, then `question` comes back `null`.
2. **Learn from them:** `curl -X POST localhost:8080/fit`. The reply says what it learned.
3. **Live phase.** From now on use step 2 as written, with no mode. Tests are shorter and more honest.
4. **Check for bad questions any time:** `curl localhost:8080/flags`.

Run `/fit` again after about 100 more students, when you add questions, or when the course changes.

### If something goes wrong

The reply is a short message with a code 400, for example `{"error": "unknown question 'nope'; add_question() it first"}`. Fix the request and send it again. Nothing is changed by a request that fails.

---

## Recipe 4: Use it from Python

```python
from bandtest import Course

course = Course(bands=10)
course.add_question("q17", band=4)          # repeat for every question

s = course.start_student("s42")             # mode="calibrate" for the calibration run
while (q := s.next_question()) is not None:
    s.record(q, correct=ask_and_grade(q))   # your code shows q and grades it
print(s.estimate())                         # band, band_range, confidence

course.fit()                                # learn from everyone so far
print(course.flagged_questions())           # questions that look misplaced
open("course.json", "w").write(course.to_json())   # save; Course.from_json() restores it
```

Already have past answers? Load them with `course.record_answer(student, question, correct)` and then call `course.fit()`.

A student who stops half way can come back later: call `course.start_student("s42")` again and it carries on from their saved answers.

---

## Planning your calibration

- **How many students?** 30 is the minimum. At 30 the library learns how clean your answers are. At about 150 it also checks every question.
- **Which students?** A mix of levels. If only your best students take part, the library cannot tell where the easy and hard questions really are.
- **How many questions each?** About 20, covering every band. Calibration mode does this for you.
- **First tries only.** Count each student's first try on a question, not retakes.
- **Recalibrate** after about 100 new students, when you add or change questions, or when the course content changes.

## Reading the results

| What you see | What it usually means | What to do |
|---|---|---|
| Many students get a two-band range | They sit on the edge of two bands. That is normal and honest. | Report the range. Do not force a single band. |
| Tests keep running to 25 questions | Answers are noisy, or there are too few questions per band. | Add questions. Check for badly worded ones. |
| Confidence is low at the end | Same as above. | Same as above. |
| Nothing is flagged | Either your bank is good, or you have fewer than 150 students. | Gather more calibration students. |
| A question is flagged | Its answers look like another band. | Check the wording and answer key first, then move it. |

## Words used here

- **Band**: a level of the course, 1 is easiest.
- **Calibration**: a run where each student answers about 20 questions across all bands, so the library can learn how your course behaves.
- **Curve**: how likely a right answer is when a student is above or below a question's level. Cleaner answers make a steeper curve and a shorter test.
- **Flag**: a question whose answers look like it sits in a different band than you listed.
- **Confidence**: the chance the student is in the reported band.
- **Range**: the smallest group of neighbouring bands that is 90% likely to hold the student.

## Want to see it first?

Open the demos in `docs/`: [watch a test](docs/demo-test.html), [calibrate a made-up course](docs/demo-calibration.html), or [try your own data](docs/try-your-data.html).
