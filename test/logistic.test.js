const test = require('node:test');
const assert = require('node:assert/strict');
const ML = require('./load.js');

const LR = ML.logistic;
const model = LR.fit(LR.base);
const scoresOf = (pts, d = 0) => pts.map((p) => ML.sigmoid(LR.score(p, model.w1, model.w2, model.b) + d));
const labelsOf = (pts) => pts.map((p) => p.y);
const testScores = scoresOf(LR.test), testLabels = labelsOf(LR.test);

// What the step 5 table compares: the 1,300 site patients not used for recalibration.
function atSite(prev, bias, d = 0) {
  const pts = LR.site(prev, bias).slice(200);
  const s = scoresOf(pts, d), y = labelsOf(pts);
  return { auc: ML.auc(s, y), gap: ML.mean(s) - ML.mean(y) };
}

test('the simulated cohorts are the same on every load', () => {
  assert.deepEqual(LR.cohort(11, 220), LR.base);
  assert.deepEqual(LR.cohort(12, 150), LR.test);
  assert.equal(LR.base.length, 220);
  assert.equal(LR.test.length, 150); // "We've also locked away 150 more patients"
});

test('the gradient matches finite differences of the log loss', () => {
  const r = ML.rng(3), eps = 1e-6;
  for (let k = 0; k < 5; k++) {
    const w = [r() * 8 - 4, r() * 8 - 4, r() * 8 - 4];
    const g = LR.gradient(LR.base, ...w);
    for (let i = 0; i < 3; i++) {
      const up = w.slice(), dn = w.slice();
      up[i] += eps; dn[i] -= eps;
      const numeric = (LR.logLoss(LR.base, ...up) - LR.logLoss(LR.base, ...dn)) / (2 * eps);
      assert.ok(Math.abs(numeric - g[i]) < 1e-6, `weights ${w}, component ${i}: ${numeric} vs ${g[i]}`);
    }
  }
});

test('the fully trained model has converged', () => {
  assert.ok(Math.hypot(...LR.gradient(LR.base, model.w1, model.w2, model.b)) < 1e-6);
});

test('it beats the step 2 target and, like the simulation, weighs glucose more than BMI', () => {
  assert.ok(model.loss < 0.5, `log loss ${model.loss}`); // "Getting under 0.50 is good going"
  assert.ok(model.w1 > model.w2 && model.w2 > 0);
});

test('it ranks held-out patients well', () => {
  assert.ok(ML.auc(testScores, testLabels) > 0.8);
});

test('the screening preset is the highest threshold that still flags 90% of cases', () => {
  const t = LR.screeningThreshold(testScores, testLabels);
  assert.ok(ML.confusion(testScores, testLabels, t).sens >= 0.9);
  assert.ok(ML.confusion(testScores, testLabels, t + 0.01).sens < 0.9);
});

test('the few-false-alarms preset is the lowest threshold that leaves 90% of non-cases alone', () => {
  const t = LR.specificThreshold(testScores, testLabels);
  assert.ok(ML.confusion(testScores, testLabels, t).spec >= 0.9);
  assert.ok(ML.confusion(testScores, testLabels, t - 0.01).spec < 0.9);
});

test('the balanced preset maximizes sensitivity plus specificity', () => {
  const j = (t) => { const m = ML.confusion(testScores, testLabels, t); return m.sens + m.spec; };
  const best = j(LR.balancedThreshold(testScores, testLabels));
  for (let k = 1; k <= 99; k++) assert.ok(j(k / 100) <= best + 1e-12, `threshold ${k}%`);
});

test('the cost threshold is where flagging and not flagging cost the same', () => {
  for (const c of [0.5, 1, 4, 20]) {
    const t = LR.costThreshold(c);
    assert.ok(Math.abs(t * c - (1 - t)) < 1e-12); // expected cost of a miss = cost of a referral
  }
});

test('at the original hospital the model is calibrated', () => {
  // The page calls rates within 2.5 points a match.
  assert.ok(Math.abs(atSite(LR.siteDefault, 0).gap) < 0.025);
});

test('the site simulator hits the prevalence you ask for', () => {
  for (const prev of [0.05, 0.2, 0.45, 0.6]) {
    const risk = ML.mean(LR.site(prev, 0).map((p) => p.risk));
    assert.ok(Math.abs(risk - prev) < 1e-4, `${prev}: ${risk}`);
  }
});

test('when prevalence changes, AUC barely moves while calibration slips', () => {
  const home = atSite(LR.siteDefault, 0);
  for (const prev of [0.1, 0.5]) {
    const s = atSite(prev, 0);
    assert.ok(Math.abs(s.auc - home.auc) < 0.05, `prevalence ${prev}: AUC ${s.auc} vs ${home.auc}`);
    assert.ok(Math.abs(s.gap) > 0.1, `prevalence ${prev}: gap ${s.gap}`);
  }
});

test('a biased glucose assay leaves the ranking untouched but shifts every risk', () => {
  const home = atSite(LR.siteDefault, 0);
  const high = atSite(LR.siteDefault, 20), low = atSite(LR.siteDefault, -20);
  assert.equal(high.auc, home.auc);
  assert.equal(low.auc, home.auc);
  assert.ok(high.gap > 0.1); // reads high, so the model overestimates
  assert.ok(low.gap < -0.1);
});

test('recalibration matches predicted to observed on the 200 patients it was fit to, and keeps the ranking', () => {
  for (const [prev, bias] of [[0.1, 0], [0.5, 0], [LR.siteDefault, 20], [0.45, -15]]) {
    const local = LR.site(prev, bias).slice(0, 200);
    const d = LR.recalibrate(local, model.w1, model.w2, model.b);
    assert.ok(Math.abs(ML.mean(scoresOf(local, d)) - ML.mean(labelsOf(local))) < 1e-9);
    assert.equal(atSite(prev, bias, d).auc, atSite(prev, bias).auc);
  }
});

test('recalibration shrinks a large calibration gap on the patients it never saw', () => {
  for (let prev = 0.05; prev <= 0.6; prev += 0.05) {
    for (let bias = -25; bias <= 25; bias += 5) {
      const p = Math.round(prev * 100) / 100;
      const before = atSite(p, bias).gap;
      if (Math.abs(before) < 0.1) continue;
      const d = LR.recalibrate(LR.site(p, bias).slice(0, 200), model.w1, model.w2, model.b);
      const after = atSite(p, bias, d).gap;
      // 200 patients pin a rate down to within a few points, not exactly.
      assert.ok(Math.abs(after) < 0.04, `prevalence ${p}, bias ${bias}: ${before} -> ${after}`);
    }
  }
});

test('calibration bins skip thin bins and average what is in them', () => {
  const scores = [...Array(12).fill(0.15), ...Array(5).fill(0.55), ...Array(10).fill(0.85)];
  const labels = [...Array(3).fill(1), ...Array(9).fill(0), ...Array(5).fill(1), ...Array(10).fill(1)];
  const bins = LR.calibration(scores, labels);
  assert.deepEqual(bins.map((b) => b.n), [12, 10]); // the 50–60% bin has only 5 patients
  assert.deepEqual(bins.map((b) => b.y), [0.25, 1]);
  assert.ok(Math.abs(bins[0].x - 0.15) < 1e-12 && Math.abs(bins[1].x - 0.85) < 1e-12);
});
