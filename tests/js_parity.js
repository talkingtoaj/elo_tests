// Prints the browser library's results on the sample file, for tests/test_js_parity.py.
const fs = require("fs");
const path = require("path");
require(path.join(__dirname, "..", "docs", "bandtest.js"));
const { Course } = globalThis.BandTest;
const lines = fs.readFileSync(path.join(__dirname, "..", "examples", "sample_answers.csv"), "utf8").trim().split("\n").slice(1);
const qband = {};
const rows = lines.map(l => { const [s, q, b, c] = l.split(","); qband[q] = +b; return [s, q, +c === 1]; });
const course = new Course(10, { seed: 0 });
Object.keys(qband).forEach(q => course.addQuestion(q, qband[q]));
rows.forEach(([s, q, c]) => course.recordAnswer(s, q, c));
const report = course.fit();
const est = course.startStudent("s001").estimate();
console.log(JSON.stringify({
  curve: [report.curve.scale, report.curve.slip],
  flagged: course.flaggedQuestions().map(f => [f.question, f.listedBand, f.looksLikeBand, f.answers]),
  s001: [est.band, est.bandRange, Math.round(est.confidence * 100) / 100],
}));
