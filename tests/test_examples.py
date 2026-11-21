import csv
import json
import pathlib
import sys
import threading
import urllib.request

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "examples"))

import calibrate_from_csv  # noqa: E402
import server  # noqa: E402


def test_csv_tool_finds_the_planted_questions(tmp_path):
    calibrate_from_csv.main([str(ROOT / "examples" / "sample_answers.csv"), "--out", str(tmp_path)])
    flagged = {r["question"] for r in csv.DictReader(open(tmp_path / "flagged_questions.csv"))}
    assert {"q07", "q33"} <= flagged
    students = list(csv.DictReader(open(tmp_path / "student_bands.csv")))
    assert len(students) == 200
    assert all(1 <= int(r["band"]) <= 10 for r in students)


def test_csv_tool_gives_plain_errors(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("student,question,band\ns1,q1,2\n")
    with pytest.raises(SystemExit, match="Missing column"):
        calibrate_from_csv.read_answers(bad)
    bad.write_text("student,question,band,correct\ns1,q1,2,maybe\n")
    with pytest.raises(SystemExit, match="don't understand"):
        calibrate_from_csv.read_answers(bad)
    bad.write_text("student,question,band,correct\ns1,q1,2,1\ns2,q1,3,0\n")
    with pytest.raises(SystemExit, match="one band"):
        calibrate_from_csv.build_course(calibrate_from_csv.read_answers(bad))


@pytest.fixture()
def live(tmp_path):
    state = tmp_path / "course.json"
    srv = server.make_server(server.Service(4, state), port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", state
    srv.shutdown()


def call(base, method, path, body=None):
    req = urllib.request.Request(base + path, method=method,
                                 data=json.dumps(body).encode() if body is not None else None)
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_server_runs_a_whole_test_and_survives_a_restart(live):
    base, state = live
    qs = [{"id": f"b{b}-{k}", "band": b} for b in range(1, 5) for k in range(5)]
    assert call(base, "POST", "/questions", {"questions": qs}) == (200, {"questions": 20})
    asked = 0
    while True:
        _, r = call(base, "POST", "/students/ann/next", {})
        if r["question"] is None:
            break
        band = int(r["question"][1])
        call(base, "POST", "/students/ann/answer", {"question": r["question"], "correct": band <= 2})
        asked += 1
        assert asked < 25
    _, r = call(base, "GET", "/students/ann")
    assert r["estimate"]["done"] and r["estimate"]["band"] in (2, 3)
    # restart: a new server reading the same state file still knows ann
    srv2 = server.make_server(server.Service(4, state), port=0)
    threading.Thread(target=srv2.serve_forever, daemon=True).start()
    base2 = f"http://127.0.0.1:{srv2.server_address[1]}"
    _, again = call(base2, "GET", "/students/ann")
    assert again["estimate"]["n_answered"] == asked
    srv2.shutdown()


def test_server_errors_are_json(live):
    base, _ = live
    assert call(base, "GET", "/nowhere")[0] == 404
    code, body = call(base, "POST", "/students/x/answer", {"question": "nope", "correct": True})
    assert code == 400 and "error" in body
