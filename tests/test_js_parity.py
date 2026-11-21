"""The browser copy (docs/bandtest.js) must give the same answers as the Python library."""
import json
import pathlib
import shutil
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "examples"))

import calibrate_from_csv  # noqa: E402


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_browser_copy_matches_python():
    out = subprocess.run(["node", str(ROOT / "tests" / "js_parity.js")], capture_output=True, text=True, check=True)
    js = json.loads(out.stdout)
    course = calibrate_from_csv.build_course(calibrate_from_csv.read_answers(ROOT / "examples" / "sample_answers.csv"))
    report = course.fit()
    est = course.start_student("s001").estimate()
    assert js["curve"] == [report.curve.scale, report.curve.slip]
    assert [tuple(f) for f in js["flagged"]] == course.flagged_questions()
    assert js["s001"][0] == est.band
    assert tuple(js["s001"][1]) == est.band_range
    assert js["s001"][2] == pytest.approx(est.confidence, abs=0.01)
