/* Seeded randomness, the stats every chapter reports, and number formats.
 *
 * Every file in js/engine/ is DOM-free and attaches to one global, ML. The same files load as
 * classic <script> tags in the browser (so the page works from file://) and through require()
 * in Node for the tests, with no build step.
 */
(function (ML) {
  'use strict';

  // Mulberry32: small, fast, seedable. Every dataset here is reproducible.
  ML.rng = function (seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  };
  ML.randn = function (r) {
    let u = 0;
    while (u === 0) u = r();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r());
  };
  ML.sigmoid = (z) => 1 / (1 + Math.exp(-z));
  ML.clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  ML.mean = (a) => a.reduce((s, v) => s + v, 0) / (a.length || 1);
  ML.sd = function (a) {
    const m = ML.mean(a);
    return Math.sqrt(a.reduce((s, v) => s + (v - m) * (v - m), 0) / Math.max(1, a.length - 1));
  };
  ML.shuffle = function (arr, r) {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(r() * (i + 1));
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
    return arr;
  };

  // Area under the ROC curve via the Mann-Whitney rank sum (ties count half).
  ML.auc = function (scores, labels) {
    const idx = scores.map((_, i) => i).sort((a, b) => scores[a] - scores[b]);
    let rankSum = 0, nPos = 0, i = 0;
    while (i < idx.length) {
      let j = i;
      while (j + 1 < idx.length && scores[idx[j + 1]] === scores[idx[i]]) j++;
      const avg = (i + j) / 2 + 1;
      for (let k = i; k <= j; k++) if (labels[idx[k]]) { rankSum += avg; nPos++; }
      i = j + 1;
    }
    const nNeg = idx.length - nPos;
    if (!nPos || !nNeg) return NaN;
    return (rankSum - (nPos * (nPos + 1)) / 2) / (nPos * nNeg);
  };

  // ROC points from the strictest threshold to the loosest.
  ML.roc = function (scores, labels) {
    const idx = scores.map((_, i) => i).sort((a, b) => scores[b] - scores[a]);
    const P = labels.filter(Boolean).length, N = labels.length - P;
    const pts = [{ fpr: 0, tpr: 0 }];
    let tp = 0, fp = 0, i = 0;
    while (i < idx.length) {
      const s = scores[idx[i]];
      while (i < idx.length && scores[idx[i]] === s) { labels[idx[i]] ? tp++ : fp++; i++; }
      pts.push({ fpr: fp / N, tpr: tp / P, t: s });
    }
    return pts;
  };

  // Confusion counts and the usual clinical metrics at one threshold (flag if score >= t).
  ML.confusion = function (scores, labels, t) {
    let tp = 0, fp = 0, tn = 0, fn = 0;
    for (let i = 0; i < scores.length; i++) {
      const flag = scores[i] >= t;
      if (labels[i]) flag ? tp++ : fn++;
      else flag ? fp++ : tn++;
    }
    const div = (a, b) => (b ? a / b : NaN);
    return {
      tp, fp, tn, fn,
      sens: div(tp, tp + fn), spec: div(tn, tn + fp),
      ppv: div(tp, tp + fp), npv: div(tn, tn + fn),
      acc: div(tp + tn, scores.length),
    };
  };

  ML.ticks = function (a, b, n) {
    const step0 = (b - a) / Math.max(1, n);
    const mag = Math.pow(10, Math.floor(Math.log10(step0)));
    const err = step0 / mag;
    const step = (err >= 7.5 ? 10 : err >= 3.5 ? 5 : err >= 1.5 ? 2 : 1) * mag;
    const out = [];
    for (let v = Math.ceil(a / step - 1e-9) * step; v <= b + 1e-9; v += step) out.push(+v.toFixed(10));
    return out;
  };

  ML.fmt = (v, d = 2) => (Number.isFinite(v) ? v.toFixed(d) : '–');
  ML.pct = (v, d = 0) => (Number.isFinite(v) ? (v * 100).toFixed(d) + '%' : '–');
  ML.signed = (v, d = 2) => (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(d);
})(globalThis.ML = globalThis.ML || {});
