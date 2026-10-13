const test = require('node:test');
const assert = require('node:assert/strict');
const ML = require('./load.js');

// AUC by its definition: the chance a random case outscores a random non-case, ties counting half.
function aucByPairs(scores, labels) {
  let wins = 0, pairs = 0;
  scores.forEach((s, i) => {
    if (!labels[i]) return;
    scores.forEach((t, j) => {
      if (labels[j]) return;
      pairs++;
      wins += s > t ? 1 : s === t ? 0.5 : 0;
    });
  });
  return wins / pairs;
}

test('the seeded RNG replays the same sequence and stays in [0, 1)', () => {
  const a = ML.rng(42), b = ML.rng(42), c = ML.rng(43);
  const xs = Array.from({ length: 1000 }, a);
  assert.deepEqual(Array.from({ length: 1000 }, b), xs);
  assert.notDeepEqual(Array.from({ length: 1000 }, c), xs);
  assert.ok(xs.every((x) => x >= 0 && x < 1));
});

test('randn draws look standard normal', () => {
  const r = ML.rng(7);
  const xs = Array.from({ length: 20000 }, () => ML.randn(r));
  assert.ok(Math.abs(ML.mean(xs)) < 0.03);
  assert.ok(Math.abs(ML.sd(xs) - 1) < 0.03);
});

test('AUC matches the pairwise definition, with ties counting half', () => {
  for (const seed of [1, 2, 3, 4, 5]) {
    const r = ML.rng(seed);
    // Coarse scores so plenty of them tie.
    const scores = Array.from({ length: 80 }, () => Math.round(r() * 10) / 10);
    const labels = scores.map((s) => (r() < 0.3 + 0.4 * s ? 1 : 0));
    assert.ok(Math.abs(ML.auc(scores, labels) - aucByPairs(scores, labels)) < 1e-12, `seed ${seed}`);
  }
});

test('AUC is 1 for a perfect ranking, 0 reversed, 0.5 when every score ties, and undefined with one class', () => {
  assert.equal(ML.auc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]), 1);
  assert.equal(ML.auc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]), 0);
  assert.equal(ML.auc([0.5, 0.5, 0.5, 0.5], [0, 1, 0, 1]), 0.5);
  assert.ok(Number.isNaN(ML.auc([0.1, 0.2], [1, 1])));
});

test('the ROC curve runs from (0, 0) to (1, 1) without stepping back, and its area is the AUC', () => {
  const r = ML.rng(11);
  const scores = Array.from({ length: 200 }, () => Math.round(r() * 20) / 20);
  const labels = scores.map((s) => (r() < s ? 1 : 0));
  const pts = ML.roc(scores, labels);
  assert.deepEqual([pts[0].fpr, pts[0].tpr], [0, 0]);
  assert.deepEqual([pts.at(-1).fpr, pts.at(-1).tpr], [1, 1]);
  let area = 0;
  for (let i = 1; i < pts.length; i++) {
    assert.ok(pts[i].fpr >= pts[i - 1].fpr && pts[i].tpr >= pts[i - 1].tpr);
    area += (pts[i].fpr - pts[i - 1].fpr) * (pts[i].tpr + pts[i - 1].tpr) / 2;
  }
  assert.ok(Math.abs(area - ML.auc(scores, labels)) < 1e-12);
});

test('the confusion matrix flags scores at or above the threshold', () => {
  const m = ML.confusion([0.9, 0.6, 0.5, 0.4, 0.3, 0.1], [1, 0, 1, 1, 0, 0], 0.5);
  assert.deepEqual([m.tp, m.fp, m.fn, m.tn], [2, 1, 1, 2]);
  assert.equal(m.sens, 2 / 3);
  assert.equal(m.spec, 2 / 3);
  assert.equal(m.ppv, 2 / 3);
  assert.equal(m.npv, 2 / 3);
  assert.equal(m.acc, 4 / 6);
  // Nobody flagged: PPV has no denominator.
  assert.ok(Number.isNaN(ML.confusion([0.1, 0.2], [1, 0], 0.5).ppv));
});

test('axis ticks land on round steps inside the range', () => {
  assert.deepEqual(ML.ticks(0, 1, 5), [0, 0.2, 0.4, 0.6, 0.8, 1]);
  for (const [a, b, n] of [[50, 205, 6], [15, 50, 5], [-6, 6, 6], [0.5, 12.5, 6], [-2, 2.2, 4]]) {
    const t = ML.ticks(a, b, n);
    const step = t[1] - t[0];
    const lead = step / Math.pow(10, Math.floor(Math.log10(step) + 1e-9));
    assert.ok([1, 2, 5].some((k) => Math.abs(lead - k) < 1e-9), `${a}..${b} step ${step}`);
    assert.ok(t[0] >= a - 1e-9 && t.at(-1) <= b + 1e-9);
    for (let i = 1; i < t.length; i++) assert.ok(Math.abs(t[i] - t[i - 1] - step) < 1e-9);
  }
});

test('number formats show a dash for missing values and a real minus sign', () => {
  assert.equal(ML.fmt(0.12345, 3), '0.123');
  assert.equal(ML.fmt(NaN), '–');
  assert.equal(ML.pct(0.256), '26%');
  assert.equal(ML.pct(0.256, 1), '25.6%');
  assert.equal(ML.pct(Infinity), '–');
  assert.equal(ML.signed(1.5), '+1.50');
  assert.equal(ML.signed(-1.5), '−1.50');
});
