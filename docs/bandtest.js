/* bandtest.js: a browser copy of the Python library in ../bandtest/course.py.
 * Same maths, no dependencies. The demos in this folder use it so you can
 * play with the real algorithm without installing anything.
 */
(function (root) {
  "use strict";

  var POINTS = 10;            // belief grid points per band
  var EPS = 1e-6;
  var AUTO_DIFFICULTY_STUDENTS = 150;

  // ---- small helpers ---------------------------------------------------
  function makeRng(seed) {    // mulberry32: a seeded random number generator
    var a = seed >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) >>> 0;
      var t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function gauss(rand) {
    var u = 1 - rand(), v = rand();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  // chance of a right answer for a student at `level` on a question of `difficulty`
  function pRight(curve, level, difficulty) {
    var z = (level - difficulty) / curve.scale;
    var p = curve.guess + (1 - curve.guess - curve.slip) / (1 + Math.exp(-z));
    return Math.min(1 - EPS, Math.max(EPS, p));
  }

  function logSumExp(arr) {
    var m = -Infinity, i;
    for (i = 0; i < arr.length; i++) if (arr[i] > m) m = arr[i];
    var s = 0;
    for (i = 0; i < arr.length; i++) s += Math.exp(arr[i] - m);
    return m + Math.log(s);
  }

  // ---- Course ----------------------------------------------------------
  function Course(bands, opts) {
    opts = opts || {};
    if (bands < 2) throw new Error("a course needs at least 2 bands");
    this.bands = bands;
    this.curve = opts.curve ? { scale: opts.curve.scale, slip: opts.curve.slip, guess: opts.curve.guess } :
      { scale: 0.5, slip: 0.05, guess: 0.02 };
    this.rand = makeRng(opts.seed == null ? 1 : opts.seed);
    var G = bands * POINTS;
    this.grid = new Float64Array(G);
    for (var g = 0; g < G; g++) this.grid[g] = (g + 0.5) / POINTS;
    this.ids = []; this.index = {}; this.band = []; this.offset = [];
    this.log = [];            // [student, question index, correct]
    this.byStudent = {};      // student -> [[question index, correct], ...]
    this.version = 0;
    this._pc = null;
  }

  Course.prototype.addQuestion = function (id, band) {
    if (band < 1 || band > this.bands) throw new Error("band must be 1.." + this.bands);
    if (Object.prototype.hasOwnProperty.call(this.index, id)) throw new Error("question " + id + " already exists");
    this.index[id] = this.ids.length;
    this.ids.push(id); this.band.push(band); this.offset.push(0);
    this._pc = null;
  };

  Course.prototype.difficulty = function (id) {
    var i = this.index[id];
    return this.band[i] - 0.5 + this.offset[i];
  };

  Course.prototype.setCurve = function (c) {
    this.curve = { scale: c.scale, slip: c.slip, guess: c.guess == null ? this.curve.guess : c.guess };
    this._touch();
  };

  Course.prototype.setDifficulty = function (id, d) {
    var i = this.index[id];
    this.offset[i] = d - (this.band[i] - 0.5);
    this._touch();
  };

  Course.prototype._touch = function () { this._pc = null; this.version++; };

  Course.prototype._difficulties = function () {
    var out = [];
    for (var i = 0; i < this.ids.length; i++) out.push(this.band[i] - 0.5 + this.offset[i]);
    return out;
  };

  // rows: questions, columns: grid levels
  Course.prototype._matrix = function (curve, difficulties) {
    var G = this.grid.length, rows = [];
    for (var q = 0; q < difficulties.length; q++) {
      var row = new Float64Array(G);
      for (var g = 0; g < G; g++) row[g] = pRight(curve, this.grid[g], difficulties[q]);
      rows.push(row);
    }
    return rows;
  };

  Course.prototype.pc = function () {
    if (!this._pc || this._pc.length !== this.ids.length) this._pc = this._matrix(this.curve, this._difficulties());
    return this._pc;
  };

  Course.prototype.recordAnswer = function (student, id, correct) {
    if (!Object.prototype.hasOwnProperty.call(this.index, id)) throw new Error("unknown question " + id);
    var q = this.index[id], list = this.byStudent[student] || (this.byStudent[student] = []);
    for (var i = 0; i < list.length; i++) if (list[i][0] === q) throw new Error(student + " already answered " + id);
    this._record(student, q, !!correct);
  };

  Course.prototype._record = function (student, q, correct) {
    this.log.push([student, q, correct]);
    (this.byStudent[student] || (this.byStudent[student] = [])).push([q, correct]);
  };

  Course.prototype.startStudent = function (student, opts) {
    return new Session(this, student, opts || {});
  };

  // ---- fitting ---------------------------------------------------------
  Course.prototype.fit = function (opts) {
    opts = opts || {};
    var self = this, n = this.log.length;
    if (!n) throw new Error("no answers recorded yet");
    var fitCurve = opts.fitCurve !== false;
    var iterations = opts.iterations || 4, tau = opts.tau || 0.5;

    var sIndex = {}, ns = 0, sIdx = new Int32Array(n), qIdx = new Int32Array(n), y = new Uint8Array(n), r;
    for (r = 0; r < n; r++) {
      var rec = this.log[r];
      if (!(rec[0] in sIndex)) sIndex[rec[0]] = ns++;
      sIdx[r] = sIndex[rec[0]]; qIdx[r] = rec[1]; y[r] = rec[2] ? 1 : 0;
    }
    var fitDifficulty = opts.fitDifficulty == null ? ns >= AUTO_DIFFICULTY_STUDENTS : !!opts.fitDifficulty;

    var G = this.grid.length, Q = this.ids.length;
    var offsets = this.offset.slice(), bands = this.band;
    var curve = { scale: this.curve.scale, slip: this.curve.slip, guess: this.curve.guess };
    var logG = Math.log(G);

    function studentLoglik(cv, off) {
      var diffs = [], q;
      for (q = 0; q < Q; q++) diffs.push(bands[q] - 0.5 + off[q]);
      var pc = self._matrix(cv, diffs), lp = [], lq = [];
      for (q = 0; q < Q; q++) {
        var a = new Float64Array(G), b = new Float64Array(G);
        for (var g = 0; g < G; g++) { a[g] = Math.log(pc[q][g]); b[g] = Math.log(1 - pc[q][g]); }
        lp.push(a); lq.push(b);
      }
      var LL = [];
      for (var s = 0; s < ns; s++) LL.push(new Float64Array(G));
      for (r = 0; r < n; r++) {
        var row = LL[sIdx[r]], src = y[r] ? lp[qIdx[r]] : lq[qIdx[r]];
        for (var g2 = 0; g2 < G; g2++) row[g2] += src[g2];
      }
      var total = 0, tmp = new Float64Array(G);
      for (var s2 = 0; s2 < ns; s2++) {
        for (var g3 = 0; g3 < G; g3++) tmp[g3] = LL[s2][g3] - logG;
        total += logSumExp(tmp);
      }
      return { score: total, LL: LL };
    }

    var scales = [0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 1.0, 1.3];
    var slips = [0, 0.02, 0.05, 0.1, 0.15, 0.2];
    var rounds = fitDifficulty ? iterations : 1;

    for (var it = 0; it < rounds; it++) {
      if (fitCurve) {
        var best = null;
        for (var i = 0; i < scales.length; i++) {
          for (var j = 0; j < slips.length; j++) {
            var cand = { scale: scales[i], slip: slips[j], guess: curve.guess };
            var sc = studentLoglik(cand, offsets).score;
            if (!best || sc > best.score) best = { score: sc, curve: cand };
          }
        }
        curve = best.curve;
      }
      if (fitDifficulty) {
        var LL = studentLoglik(curve, offsets).LL;
        var w1 = [], w0 = [], q2, g4;
        for (q2 = 0; q2 < Q; q2++) { w1.push(new Float64Array(G)); w0.push(new Float64Array(G)); }
        var post = [];
        for (var s3 = 0; s3 < ns; s3++) {
          var lse = logSumExp(LL[s3]), p = new Float64Array(G);
          for (g4 = 0; g4 < G; g4++) p[g4] = Math.exp(LL[s3][g4] - lse);
          post.push(p);
        }
        var count = new Int32Array(Q);
        for (r = 0; r < n; r++) {
          var target = y[r] ? w1[qIdx[r]] : w0[qIdx[r]], pp = post[sIdx[r]];
          for (g4 = 0; g4 < G; g4++) target[g4] += pp[g4];
          count[qIdx[r]]++;
        }
        var cands = [], pen = [], k;
        for (k = 0; k <= 60; k++) { cands.push(-3 + k * 0.1); pen.push(-Math.log(1 + Math.pow((-3 + k * 0.1) / tau, 2))); }
        var tables = {};
        for (var b = 1; b <= this.bands; b++) {
          var lpT = [], lqT = [];
          for (k = 0; k < cands.length; k++) {
            var a1 = new Float64Array(G), a0 = new Float64Array(G);
            for (g4 = 0; g4 < G; g4++) {
              var pr = pRight(curve, this.grid[g4], b - 0.5 + cands[k]);
              a1[g4] = Math.log(pr); a0[g4] = Math.log(1 - pr);
            }
            lpT.push(a1); lqT.push(a0);
          }
          tables[b] = [lpT, lqT];
        }
        for (q2 = 0; q2 < Q; q2++) {
          if (!count[q2]) continue;
          var tb = tables[bands[q2]], bestK = 0, bestS = -Infinity;
          for (k = 0; k < cands.length; k++) {
            var sum = pen[k];
            for (g4 = 0; g4 < G; g4++) sum += tb[0][k][g4] * w1[q2][g4] + tb[1][k][g4] * w0[q2][g4];
            if (sum > bestS) { bestS = sum; bestK = k; }
          }
          offsets[q2] = cands[bestK];
        }
      }
    }
    var total = studentLoglik(curve, offsets).score;
    this.curve = curve;
    this.offset = offsets;
    this._touch();
    return { curve: curve, nResponses: n, nStudents: ns, logLikelihood: total, fittedDifficulty: fitDifficulty };
  };

  Course.prototype.flaggedQuestions = function (minAnswers, minShift) {
    minAnswers = minAnswers == null ? 8 : minAnswers;
    minShift = minShift == null ? 1 : minShift;
    var counts = new Int32Array(this.ids.length), i;
    for (i = 0; i < this.log.length; i++) counts[this.log[i][1]]++;
    var out = [];
    for (i = 0; i < this.ids.length; i++) {
      if (counts[i] >= minAnswers && Math.abs(this.offset[i]) >= minShift) {
        var apparent = Math.min(this.bands, Math.max(1, Math.floor(this.band[i] - 0.5 + this.offset[i]) + 1));
        if (apparent !== this.band[i]) out.push({ question: this.ids[i], listedBand: this.band[i], looksLikeBand: apparent, answers: counts[i] });
      }
    }
    out.sort(function (a, b) { return Math.abs(b.looksLikeBand - b.listedBand) - Math.abs(a.looksLikeBand - a.listedBand); });
    return out;
  };

  // copy a course (curve, difficulties and the answer log) so settings can be compared
  Course.prototype.clone = function (seed) {
    var c = new Course(this.bands, { curve: this.curve, seed: seed });
    for (var i = 0; i < this.ids.length; i++) { c.addQuestion(this.ids[i], this.band[i]); c.offset[i] = this.offset[i]; }
    for (var r = 0; r < this.log.length; r++) c._record(this.log[r][0], this.log[r][1], this.log[r][2]);
    return c;
  };

  // ---- Session ---------------------------------------------------------
  function Session(course, student, o) {
    this.course = course;
    this.student = student;
    this.mode = o.mode || "adaptive";
    this.nCalibration = o.nCalibration == null ? 20 : o.nCalibration;
    this.confidence = o.confidence == null ? 0.9 : o.confidence;
    this.maxQuestions = o.maxQuestions == null ? 25 : o.maxQuestions;
    this.rangeMass = o.rangeMass == null ? 0.9 : o.rangeMass;
    this.allowBorderline = o.allowBorderline !== false;
    this.borderlineSd = 0.3;
    var G = course.grid.length;
    this.logPrior = new Float64Array(G);
    if (o.priorBand != null) {
      for (var g = 0; g < G; g++) this.logPrior[g] = -0.5 * Math.pow((course.grid[g] - (o.priorBand - 0.5)) / 1.5, 2);
    }
    this.answers = (course.byStudent[student] || []).slice();
    this.seen = {};
    for (var i = 0; i < this.answers.length; i++) this.seen[this.answers[i][0]] = true;
    this.logpost = null; this.version = -1; this.order = [];
  }

  Session.prototype._ensure = function () {
    var c = this.course;
    if (this.logpost && this.version === c.version) return;
    var pc = c.pc(), lp = Float64Array.from(this.logPrior), G = lp.length;
    for (var i = 0; i < this.answers.length; i++) {
      var q = this.answers[i][0], ok = this.answers[i][1];
      for (var g = 0; g < G; g++) lp[g] += Math.log(ok ? pc[q][g] : 1 - pc[q][g]);
    }
    this.logpost = lp; this.version = c.version;
  };

  Session.prototype._post = function () {
    this._ensure();
    var lp = this.logpost, G = lp.length, m = -Infinity, g;
    for (g = 0; g < G; g++) if (lp[g] > m) m = lp[g];
    var p = new Float64Array(G), s = 0;
    for (g = 0; g < G; g++) { p[g] = Math.exp(lp[g] - m); s += p[g]; }
    for (g = 0; g < G; g++) p[g] /= s;
    return p;
  };

  Session.prototype.record = function (id, correct) {
    var c = this.course;
    if (!Object.prototype.hasOwnProperty.call(c.index, id)) throw new Error("unknown question " + id);
    var q = c.index[id];
    if (this.seen[q]) throw new Error(id + " was already answered by this student");
    this._ensure();
    var row = c.pc()[q];
    for (var g = 0; g < this.logpost.length; g++) this.logpost[g] += Math.log(correct ? row[g] : 1 - row[g]);
    this.answers.push([q, !!correct]);
    this.seen[q] = true;
    c._record(this.student, q, !!correct);
  };

  Session.prototype.estimate = function () {
    var c = this.course, B = c.bands, post = this._post(), probs = [], b, g;
    for (b = 0; b < B; b++) {
      var s = 0;
      for (g = 0; g < POINTS; g++) s += post[b * POINTS + g];
      probs.push(s);
    }
    var top = 0;
    for (b = 1; b < B; b++) if (probs[b] > probs[top]) top = b;
    var lo = top, hi = top, mass = probs[top];
    while (mass < this.rangeMass && (lo > 0 || hi < B - 1)) {
      var left = lo > 0 ? probs[lo - 1] : -1, right = hi < B - 1 ? probs[hi + 1] : -1;
      if (left >= right) { lo--; mass += left; } else { hi++; mass += right; }
    }
    var n = this.answers.length, used = Object.keys(this.seen).length >= c.ids.length;
    var done, why;
    if (this.mode === "calibrate") {
      done = n >= this.nCalibration || used; why = done ? "calibration questions finished" : "";
    } else {
      var pair = 0;
      for (b = 0; b < B - 1; b++) pair = Math.max(pair, probs[b] + probs[b + 1]);
      var mean = 0, sq = 0;
      for (g = 0; g < post.length; g++) { mean += post[g] * c.grid[g]; sq += post[g] * c.grid[g] * c.grid[g]; }
      var sd = Math.sqrt(Math.max(0, sq - mean * mean));
      var sure = probs[top] >= this.confidence;
      var border = this.allowBorderline && pair >= this.confidence && sd <= this.borderlineSd;
      done = sure || border || n >= this.maxQuestions || used;
      why = sure ? "sure" : border ? "borderline" : n >= this.maxQuestions ? "limit" : used ? "out of questions" : "";
    }
    return { band: top + 1, bandRange: [lo + 1, hi + 1], confidence: probs[top],
      rangeConfidence: Math.min(mass, 1), nAnswered: n, done: !!done, why: why, bandProbs: probs, post: post };
  };

  Session.prototype.nextQuestion = function () {
    if (this.estimate().done) return null;
    var c = this.course, unseen = [], i;
    for (i = 0; i < c.ids.length; i++) if (!this.seen[i]) unseen.push(i);
    return c.ids[this.mode === "calibrate" ? this._pickStratified(unseen) : this._pickInformative(unseen)];
  };

  Session.prototype._pickStratified = function (unseen) {
    var c = this.course;
    for (var guard = 0; guard < 1000; guard++) {
      if (!this.order.length) {
        var bs = [];
        for (var b = 1; b <= c.bands; b++) bs.push(b);
        for (var i = bs.length - 1; i > 0; i--) {   // shuffle
          var j = Math.floor(c.rand() * (i + 1)), t = bs[i]; bs[i] = bs[j]; bs[j] = t;
        }
        this.order = bs;
      }
      var want = this.order.pop();
      var pool = unseen.filter(function (q) { return c.band[q] === want; });
      if (pool.length) return pool[Math.floor(c.rand() * pool.length)];
      if (!unseen.some(function (q) { return c.band[q] === want; }) &&
          !unseen.some(function (q) { return true; })) break;
    }
    return unseen[Math.floor(c.rand() * unseen.length)];
  };

  Session.prototype._pickInformative = function (unseen) {
    var c = this.course, B = c.bands, post = this._post(), pc = c.pc(), best = -1, bestH = Infinity;
    for (var u = 0; u < unseen.length; u++) {
      var row = pc[unseen[u]], pr = 0, right = new Float64Array(B), wrong = new Float64Array(B);
      for (var b = 0; b < B; b++) {
        for (var g = 0; g < POINTS; g++) {
          var k = b * POINTS + g, w = row[k] * post[k];
          right[b] += w; wrong[b] += post[k] - w; pr += w;
        }
      }
      var h = pr * entropy(right, pr) + (1 - pr) * entropy(wrong, 1 - pr) + c.rand() * 1e-9;
      if (h < bestH) { bestH = h; best = unseen[u]; }
    }
    return best;
  };

  function entropy(mass, total) {
    var h = 0;
    for (var i = 0; i < mass.length; i++) { var p = mass[i] / total; h -= p * Math.log(p + 1e-12); }
    return h;
  }

  root.BandTest = { Course: Course, makeRng: makeRng, gauss: gauss, pRight: pRight, POINTS: POINTS,
    AUTO_DIFFICULTY_STUDENTS: AUTO_DIFFICULTY_STUDENTS };
})(typeof window !== "undefined" ? window : globalThis);
