/* Chapter 1: simulated adults, a logistic regression for 5-year type 2 diabetes risk, the
 * threshold presets, and the same model at a second hospital. No DOM.
 */
(function (ML) {
  'use strict';
  const { clamp } = ML;
  const TRUE_B0 = -1.1;

  // ---------- simulated patients ----------

  // Draws the latent pieces separately so a site can be re-simulated with shifted
  // prevalence or lab bias while keeping the same underlying people.
  function latent(r) {
    const zg = ML.randn(r);
    return { zg, zb: 0.35 * zg + 0.937 * ML.randn(r), u: r() };
  }
  function realize(L, b0, bias) {
    const gTrue = clamp(122 + 28 * L.zg, 60, 199);
    const bmi = clamp(29 + 5.5 * L.zb, 17, 49);
    const logit = b0 + (1.6 * (gTrue - 122)) / 28 + (0.9 * (bmi - 29)) / 5.5;
    return {
      g: Math.round(clamp(gTrue + bias, 40, 240)),
      bmi: Math.round(bmi * 10) / 10,
      y: L.u < ML.sigmoid(logit) ? 1 : 0,
      risk: ML.sigmoid(logit),
    };
  }
  function cohort(seed, n) {
    const r = ML.rng(seed);
    return Array.from({ length: n }, () => realize(latent(r), TRUE_B0, 0));
  }

  const base = cohort(11, 220);
  const test = cohort(12, 150);
  const siteLatent = (() => { const r = ML.rng(13); return Array.from({ length: 1500 }, () => latent(r)); })();

  // Standardize with the original training data so weights keep their meaning as patients are added.
  const MG = ML.mean(base.map((p) => p.g)), SG = ML.sd(base.map((p) => p.g));
  const MB = ML.mean(base.map((p) => p.bmi)), SB = ML.sd(base.map((p) => p.bmi));
  const xg = (p) => (p.g - MG) / SG;
  const xb = (p) => (p.bmi - MB) / SB;

  // ---------- model ----------

  const score = (p, w1, w2, b) => w1 * xg(p) + w2 * xb(p) + b;
  const prob = (p, w1, w2, b) => ML.sigmoid(score(p, w1, w2, b));
  function logLoss(pts, w1, w2, b) {
    let s = 0;
    for (const p of pts) {
      const q = clamp(prob(p, w1, w2, b), 1e-12, 1 - 1e-12);
      s -= p.y ? Math.log(q) : Math.log(1 - q);
    }
    return s / pts.length;
  }
  function accuracy(pts, w1, w2, b) {
    let k = 0;
    for (const p of pts) if ((prob(p, w1, w2, b) >= 0.5) === !!p.y) k++;
    return k / pts.length;
  }
  // Gradient of the mean log loss with respect to [w1, w2, b].
  function gradient(pts, w1, w2, b) {
    let g1 = 0, g2 = 0, g0 = 0;
    for (const p of pts) {
      const e = prob(p, w1, w2, b) - p.y;
      g1 += e * xg(p); g2 += e * xb(p); g0 += e;
    }
    const n = pts.length;
    return [g1 / n, g2 / n, g0 / n];
  }
  // The fully trained model: plain gradient descent from zero, run long enough to converge.
  function fit(pts) {
    let w1 = 0, w2 = 0, b = 0;
    for (let i = 0; i < 3000; i++) {
      const [g1, g2, g0] = gradient(pts, w1, w2, b);
      w1 -= 1.5 * g1; w2 -= 1.5 * g2; b -= 1.5 * g0;
    }
    return { w1, w2, b, loss: logLoss(pts, w1, w2, b) };
  }

  // ---------- thresholds (flag if risk >= t) ----------

  // The step 4 presets, each a scan over thresholds from 1% to 99%.
  // Screening: the highest threshold that still flags at least 90% of future cases.
  function screeningThreshold(scores, labels) {
    let t = 0.01;
    for (let v = 0.99; v >= 0.01; v -= 0.01) if (ML.confusion(scores, labels, v).sens >= 0.9) { t = v; break; }
    return t;
  }
  // Few false alarms: the lowest threshold that leaves at least 90% of non-cases alone.
  function specificThreshold(scores, labels) {
    let t = 0.99;
    for (let v = 0.01; v <= 0.99; v += 0.01) if (ML.confusion(scores, labels, v).spec >= 0.9) { t = v; break; }
    return t;
  }
  // Balanced: the threshold with the highest sensitivity + specificity (Youden's J).
  function balancedThreshold(scores, labels) {
    let t = 0.5, bj = -1;
    for (let v = 0.01; v <= 0.99; v += 0.01) {
      const m = ML.confusion(scores, labels, v);
      if (m.sens + m.spec > bj) { bj = m.sens + m.spec; t = v; }
    }
    return t;
  }
  // If a missed case costs as much as c unneeded referrals, flagging pays off once
  // risk × c > (1 − risk) × 1, that is above 1 / (1 + c).
  const costThreshold = (c) => 1 / (1 + c);

  // ---------- a second hospital ----------

  // The original hospital's share developing diabetes, rounded to the slider's 1% steps.
  const siteDefault = (() => {
    let s = 0;
    for (const L of siteLatent) s += realize(L, TRUE_B0, 0).risk;
    return Math.round((s / siteLatent.length) * 100) / 100;
  })();
  // The baseline risk that makes `prev` of the site's patients develop diabetes on average.
  function interceptFor(prev) {
    let lo = -8, hi = 6;
    for (let k = 0; k < 40; k++) {
      const mid = (lo + hi) / 2;
      let s = 0;
      for (const L of siteLatent) s += realize(L, mid, 0).risk;
      if (s / siteLatent.length < prev) lo = mid; else hi = mid;
    }
    return (lo + hi) / 2;
  }
  // The same 1,500 people at a site with a different prevalence and a glucose assay off by `bias`.
  function site(prev, bias) {
    const b0 = Math.abs(prev - siteDefault) < 0.005 ? TRUE_B0 : interceptFor(prev);
    return siteLatent.map((L) => realize(L, b0, bias));
  }
  // Calibration curve: predicted against observed rate in tenths of risk, skipping bins under 10.
  function calibration(scores, labels) {
    const bins = Array.from({ length: 10 }, () => ({ n: 0, s: 0, y: 0 }));
    scores.forEach((s, i) => { const b = bins[Math.min(9, Math.floor(s * 10))]; b.n++; b.s += s; b.y += labels[i]; });
    return bins.filter((b) => b.n >= 10).map((b) => ({ x: b.s / b.n, y: b.y / b.n, n: b.n }));
  }
  // Recalibration in the large: the shift d added to every score so the model's average
  // predicted risk on `pts` matches what happened, found with Newton's method. The weights,
  // and so the ranking (AUC), don't change.
  function recalibrate(pts, w1, w2, b) {
    const zs = pts.map((p) => score(p, w1, w2, b));
    let d = 0;
    for (let k = 0; k < 25; k++) {
      let g = 0, h = 0;
      zs.forEach((z, i) => { const q = ML.sigmoid(z + d); g += q - pts[i].y; h += q * (1 - q); });
      d -= g / Math.max(h, 1e-9);
    }
    return d;
  }

  ML.logistic = {
    TRUE_B0, cohort, base, test, MG, SG, MB, SB, xg, xb,
    score, prob, logLoss, accuracy, gradient, fit,
    screeningThreshold, specificThreshold, balancedThreshold, costThreshold,
    siteDefault, interceptFor, site, calibration, recalibrate,
  };
})(globalThis.ML = globalThis.ML || {});
