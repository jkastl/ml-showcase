const test = require('node:test');
const assert = require('node:assert/strict');
const ML = require('./load.js');

const KM = ML.kmeans;
const { pts, STD, RAW } = KM;

test('the patients are 320 people from four hidden groups', () => {
  assert.equal(pts.length, 320);
  assert.deepEqual(KM.GROUPS.map((g, i) => pts.filter((p) => p.g === i).length), [60, 80, 90, 90]);
});

test('assign picks the nearest center, and move puts each center at the mean of its patients', () => {
  const few = [{ x: 0, y: 0 }, { x: 2, y: 0 }, { x: 10, y: 10 }, { x: 10, y: 14 }];
  const lab = KM.assign(few, [{ x: 1, y: 1 }, { x: 9, y: 9 }], RAW);
  assert.deepEqual(lab, [0, 0, 1, 1]);
  // A center nobody is assigned to stays where it was.
  assert.deepEqual(KM.centroids(few, lab, [{ x: 1, y: 1 }, { x: 9, y: 9 }, { x: 50, y: 50 }]),
    [{ x: 1, y: 0 }, { x: 10, y: 12 }, { x: 50, y: 50 }]);
});

test('neither the assign step nor the move step ever increases inertia', () => {
  const starts = [KM.BAD_START, ...[1, 2, 3].map((s) => KM.kpp(pts, 3 + s, STD, ML.rng(s)))];
  for (const start of starts) {
    let cents = start, lab = KM.assign(pts, cents, STD);
    let last = KM.inertia(pts, lab, cents, STD);
    for (let i = 0; i < 30; i++) {
      cents = KM.centroids(pts, lab, cents);
      const moved = KM.inertia(pts, lab, cents, STD);
      lab = KM.assign(pts, cents, STD);
      const assigned = KM.inertia(pts, lab, cents, STD);
      assert.ok(moved <= last + 1e-9 && assigned <= moved + 1e-9);
      last = assigned;
    }
  }
});

test('a finished run is stable: nobody switches and no center moves', () => {
  const res = KM.run(pts, KM.kpp(pts, 4, STD, ML.rng(9)), STD);
  assert.deepEqual(KM.assign(pts, res.cents, STD), res.lab);
  const again = KM.centroids(pts, res.lab, res.cents);
  again.forEach((c, k) => assert.ok(Math.abs(c.x - res.cents[k].x) < 1e-9 && Math.abs(c.y - res.cents[k].y) < 1e-9));
});

test('k-means++ starts on k different patients', () => {
  for (let s = 0; s < 10; s++) {
    const cents = KM.kpp(pts, 6, STD, ML.rng(s));
    assert.equal(new Set(cents.map((c) => `${c.x},${c.y}`)).size, 6);
    for (const c of cents) assert.ok(pts.some((p) => p.x === c.x && p.y === c.y));
  }
});

test('the bad start gets stuck, while k-means++ starts usually find the best answer', () => {
  const best = KM.bestOf(pts, 4, STD, 10, 99);
  // The page says "Stuck" when a run ends more than 5% above the best k = 4 inertia.
  assert.ok(KM.run(pts, KM.BAD_START, STD).inertia > best.inertia * 1.05);
  let found = 0;
  for (let s = 0; s < 20; s++) if (KM.run(pts, KM.kpp(pts, 4, STD, ML.rng(s)), STD).inertia <= best.inertia * 1.01) found++;
  assert.ok(found >= 15, `${found} of 20`);
});

test('inertia falls as k grows, and silhouette peaks at the four simulated groups', () => {
  // The same runs as the step 3 charts.
  const sweep = [1, 2, 3, 4, 5, 6].map((k) => {
    const res = KM.bestOf(pts, k, STD, 10, 100 + k);
    return { k, inertia: res.inertia, sil: KM.silhouette(pts, res.lab, k, STD) };
  });
  for (let i = 1; i < sweep.length; i++) assert.ok(sweep[i].inertia < sweep[i - 1].inertia);
  const top = sweep.slice(1).reduce((a, b) => (b.sil > a.sil ? b : a));
  assert.equal(top.k, 4);
});

test('silhouette is near 1 for two far-apart tight groups and always within −1 to 1', () => {
  const r = ML.rng(5);
  const blobs = Array.from({ length: 40 }, (_, i) => ({ x: (i < 20 ? 0 : 100) + r(), y: r(), g: i < 20 ? 0 : 1 }));
  assert.ok(KM.silhouette(blobs, blobs.map((p) => p.g), 2, RAW) > 0.95);
  const random = blobs.map(() => Math.floor(r() * 3));
  const s = KM.silhouette(blobs, random, 3, RAW);
  assert.ok(s >= -1 && s <= 1);
  assert.ok(Number.isNaN(KM.silhouette(blobs, blobs.map(() => 0), 1, RAW)));
});

test('standardized units recover the hidden groups far better than raw units', () => {
  const raw = KM.agreement(pts, KM.bestOf(pts, 4, RAW, 10, 400).lab);
  const std = KM.agreement(pts, KM.bestOf(pts, 4, STD, 10, 400).lab);
  assert.ok(std > 0.9, `standardized ${std}`);
  assert.ok(raw < 0.75, `raw ${raw}`);
});

test('clusters are numbered by age, so a cluster keeps its color as k or units change', () => {
  const res = KM.bestOf(pts, 5, STD, 5, 1);
  for (let k = 1; k < res.cents.length; k++) assert.ok(res.cents[k].x >= res.cents[k - 1].x);
  assert.deepEqual(KM.assign(pts, res.cents, STD), res.lab);
});

test('agreement matches clusters to groups whatever the clusters are called', () => {
  const truth = pts.map((p) => p.g);
  assert.equal(KM.agreement(pts, truth), 1);
  assert.equal(KM.agreement(pts, truth.map((g) => [2, 0, 3, 1][g])), 1);
});
