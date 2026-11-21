/* ui.js: tiny helpers shared by the demo pages (no dependencies). */
(function (root) {
  "use strict";
  var NS = "http://www.w3.org/2000/svg";

  function s(tag, attrs, text) {
    var e = document.createElementNS(NS, tag);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (text != null) e.textContent = text;
    return e;
  }

  function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }

  function pct(x, digits) { return (x * 100).toFixed(digits || 0) + "%"; }

  /* Belief chart: the library's belief about the student's level, 0..bands.
   * est: result of session.estimate(); opts: {bands, truth, label} */
  function beliefChart(svg, est, opts) {
    var W = 720, H = 230, L = 58, R = 10, T = 18, B = 34;
    var bands = opts.bands, post = est.post, G = post.length;
    var pw = W - L - R, ph = H - T - B, max = 0, g;
    for (g = 0; g < G; g++) if (post[g] > max) max = post[g];
    clear(svg);
    svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    svg.setAttribute("role", "img");
    svg.setAttribute("aria-label", opts.label || "Belief about the student's level");
    var x = function (level) { return L + (level / bands) * pw; };
    var started = est.nAnswered > 0;
    // reported range
    if (started) svg.appendChild(s("rect", { x: x(est.bandRange[0] - 1), y: T, width: x(est.bandRange[1]) - x(est.bandRange[0] - 1), height: ph, fill: "var(--accent-soft)" }));
    // bars
    var bw = pw / G;
    for (g = 0; g < G; g++) {
      var h = max ? (post[g] / max) * (ph - 6) : 0;
      var inBand = started && Math.floor(g / 10) + 1 === est.band;
      svg.appendChild(s("rect", { x: L + g * bw + 0.4, y: T + ph - h, width: Math.max(bw - 0.8, 0.5), height: h, fill: inBand ? "var(--accent)" : "var(--bar)", opacity: inBand || !started ? 1 : 0.55 }));
    }
    // band lines and labels
    svg.appendChild(s("line", { x1: L, x2: W - R, y1: T + ph, y2: T + ph, "class": "axis" }));
    for (var b = 0; b <= bands; b++) {
      svg.appendChild(s("line", { x1: x(b), x2: x(b), y1: T, y2: T + ph + 4, "class": "axis", "stroke-dasharray": b === 0 || b === bands ? "" : "3 3" }));
      if (b < bands) svg.appendChild(s("text", { x: x(b + 0.5), y: T + ph + 20, "text-anchor": "middle" }, "Band " + (b + 1)));
    }
    svg.appendChild(s("text", { x: L - 6, y: T + 10, "text-anchor": "end" }, "likely"));
    svg.appendChild(s("text", { x: L - 6, y: T + ph, "text-anchor": "end" }, "unlikely"));
    // true level
    if (opts.truth != null) {
      svg.appendChild(s("line", { x1: x(opts.truth), x2: x(opts.truth), y1: T - 4, y2: T + ph, stroke: "var(--good)", "stroke-width": 2.5 }));
      svg.appendChild(s("text", { x: Math.min(Math.max(x(opts.truth), L + 40), W - 50), y: T - 4, "text-anchor": "middle", fill: "var(--good)", style: "fill:var(--good);font-weight:600" }, "real level"));
    }
  }

  /* Line chart of several curves. series: [{name, color, dash, points:[[x,y],...]}] */
  function lineChart(svg, series, o) {
    var W = 720, H = 250, L = 44, R = 14, T = 12, B = 38;
    var pw = W - L - R, ph = H - T - B;
    clear(svg);
    svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    svg.setAttribute("role", "img");
    svg.setAttribute("aria-label", o.label);
    var X = function (v) { return L + ((v - o.xmin) / (o.xmax - o.xmin)) * pw; };
    var Y = function (v) { return T + (1 - (v - o.ymin) / (o.ymax - o.ymin)) * ph; };
    var i, v;
    for (i = 0; i <= 4; i++) {
      v = o.ymin + (i / 4) * (o.ymax - o.ymin);
      svg.appendChild(s("line", { x1: L, x2: W - R, y1: Y(v), y2: Y(v), "class": "axis" }));
      svg.appendChild(s("text", { x: L - 6, y: Y(v) + 4, "text-anchor": "end" }, pct(v)));
    }
    for (v = Math.ceil(o.xmin); v <= o.xmax; v++) {
      svg.appendChild(s("text", { x: X(v), y: T + ph + 16, "text-anchor": "middle" }, (v > 0 ? "+" : "") + v));
    }
    svg.appendChild(s("line", { x1: X(0), x2: X(0), y1: T, y2: T + ph, "class": "axis", "stroke-dasharray": "3 3" }));
    svg.appendChild(s("text", { x: L + pw / 2, y: H - 4, "text-anchor": "middle" }, o.xlabel));
    series.forEach(function (sr) {
      var d = sr.points.map(function (p, k) { return (k ? "L" : "M") + X(p[0]).toFixed(1) + " " + Y(p[1]).toFixed(1); }).join(" ");
      svg.appendChild(s("path", { d: d, fill: "none", stroke: sr.color, "stroke-width": 2.6, "stroke-dasharray": sr.dash || "", "stroke-linecap": "round" }));
    });
  }

  function legend(node, series) {
    clear(node);
    series.forEach(function (sr) {
      var sp = document.createElement("span");
      sp.style.setProperty("--c", sr.color);
      sp.textContent = sr.name;
      node.appendChild(sp);
    });
  }

  function download(name, text) {
    var a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([text], { type: "text/csv" }));
    a.download = name;
    document.body.appendChild(a); a.click(); a.remove();
  }

  function csvLine(cells) {
    return cells.map(function (c) {
      c = String(c);
      return /[",\n]/.test(c) ? '"' + c.replace(/"/g, '""') + '"' : c;
    }).join(",");
  }

  root.UI = { s: s, clear: clear, pct: pct, beliefChart: beliefChart, lineChart: lineChart, legend: legend, download: download, csvLine: csvLine };
})(window);
