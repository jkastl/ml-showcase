const test = require('node:test');
const assert = require('node:assert/strict');
const ML = require('./load.js');

const TR = ML.tree;
const { train, test: heldOut } = TR;
const ALL = train.map((_, i) => i);
const aucByDepth = (minLeaf) => Array.from({ length: 12 }, (_, i) => {
  const root = TR.build(train, i + 1, minLeaf);
  return { depth: i + 1, train: TR.aucOf(root, train), test: TR.aucOf(root, heldOut) };
});

// Weighted Gini of a cut, counted directly instead of incrementally.
function weightedGini(pts, f, t) {
  const left = pts.filter((p) => p.x[f] <= t), right = pts.filter((p) => p.x[f] > t);
  const pos = (a) => a.filter((p) => p.y).length;
  return (left.length * TR.gini(pos(left), left.length) + right.length * TR.gini(pos(right), right.length)) / pts.length;
}

test('the simulated cohorts are the same on every load', () => {
  assert.deepEqual(TR.cohort(21, 300), train);
  assert.equal(train.length, 300); // "These are 300 simulated emergency patients"
  assert.equal(heldOut.length, 120); // "120 held-out patients"
});

test('Gini impurity is 0 for a pure group and 0.5 at a 50/50 split', () => {
  assert.equal(TR.gini(0, 40), 0);
  assert.equal(TR.gini(40, 40), 0);
  assert.equal(TR.gini(20, 40), 0.5);
  assert.equal(TR.gini(0, 0), 0);
});

test('the scan tries a cut between every pair of distinct values and scores it correctly', () => {
  for (const f of [0, 1]) {
    const cuts = TR.scan(train, ALL, f);
    const values = [...new Set(train.map((p) => p.x[f]))].sort((a, b) => a - b);
    assert.deepEqual(cuts.map((c) => c.t), values.slice(1).map((v, i) => (values[i] + v) / 2));
    for (const c of cuts) {
      assert.equal(c.nl + c.nr, train.length);
      assert.ok(Math.abs(c.wg - weightedGini(train, f, c.t)) < 1e-12);
    }
  }
});

test('the best split is the lowest weighted Gini on either feature', () => {
  const s = TR.bestSplit(train, ALL, 1);
  const lowest = Math.min(...[0, 1].flatMap((f) => TR.scan(train, ALL, f).map((c) => c.wg)));
  assert.equal(s.wg, lowest);
});

test('a built tree keeps to its limits, and its leaves split the patients without overlap', () => {
  for (const [depth, minLeaf] of [[3, 15], [4, 8], [12, 1]]) {
    const ls = TR.leaves(TR.build(train, depth, minLeaf));
    const seen = ls.flatMap((L) => L.idx).sort((a, b) => a - b);
    assert.deepEqual(seen, ALL);
    for (const L of ls) {
      assert.ok(L.depth <= depth);
      assert.ok(L.n >= minLeaf); // "no leaf may hold fewer patients than this"
      assert.equal(L.risk, L.pos / L.n);
    }
  }
});

test('every patient follows the questions down to the leaf that holds them', () => {
  const root = TR.build(train, 6, 3);
  for (const L of TR.leaves(root)) for (const i of L.idx) assert.equal(TR.leafFor(root, train[i].x), L);
});

test('training AUC only goes up with depth', () => {
  for (const minLeaf of [1, 5]) {
    const c = aucByDepth(minLeaf);
    for (let i = 1; i < c.length; i++) assert.ok(c[i].train >= c[i - 1].train - 1e-12, `min leaf ${minLeaf}, depth ${c[i].depth}`);
  }
});

test('held-out AUC peaks and then falls as the tree memorizes the training set', () => {
  const c = aucByDepth(1);
  const peak = c.reduce((a, b) => (b.test > a.test ? b : a));
  assert.ok(peak.depth <= 6, `peak at depth ${peak.depth}`);
  assert.ok(c.at(-1).train > 0.99);
  assert.ok(c.at(-1).test < peak.test - 0.1, `depth 12 ${c.at(-1).test} vs peak ${peak.test}`);
});

test('growing one best split at a time ends with the same tree as building it depth-first', () => {
  // Step 2's "Grow to depth 4" against the tree build() makes with the same limits.
  const root = TR.makeNode(train, ALL, 0, TR.fullBox(), null);
  let next;
  while ((next = TR.nextSplit(root, train, 4, 8))) TR.applySplit(train, next.node, next.s);
  const shape = (r) => TR.leaves(r).map((L) => `${TR.ruleFor(L)}: ${L.pos}/${L.n}`).sort();
  assert.deepEqual(shape(root), shape(TR.build(train, 4, 8)));
});

test('the depth-3 tree finds the qSOFA respiratory-rate cutoff on its own', () => {
  const root = TR.build(train, 3, 15);
  assert.equal(TR.F[root.split.f].short, 'RR');
  assert.equal(TR.rightLabel(root.split.f, root.split.t), 'RR ≥ 22'); // qSOFA: RR ≥ 22
});

test('each rule describes exactly the patients in its leaf', () => {
  const root = TR.build(train, 4, 5);
  // Read "RR 22–25", "SBP ≥ 103" or "RR ≤ 21" back as [feature, low, high]. Vitals are whole numbers.
  const bounds = (cond) => {
    const [name, a, b] = cond.split(' ');
    const f = TR.F.findIndex((F) => F.short === name);
    if (a === '≤') return [f, -Infinity, Number(b)];
    if (a === '≥') return [f, Number(b), Infinity];
    const [lo, hi] = a.split('–').map(Number);
    return [f, lo, hi];
  };
  const holds = (rule, x) => rule === 'everyone' ||
    rule.split(' and ').map(bounds).every(([f, lo, hi]) => x[f] >= lo && x[f] <= hi);
  for (const L of TR.leaves(root)) {
    const rule = TR.ruleFor(L);
    train.forEach((p, i) => assert.equal(holds(rule, p.x), L.idx.includes(i), `${rule}, patient ${i}`));
  }
});
