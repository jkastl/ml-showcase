/* Chapter 2: simulated emergency patients with suspected infection, and a CART decision tree
 * (Gini impurity, greedy splits, depth and leaf-size limits). No DOM.
 */
(function (ML) {
  'use strict';
  const { clamp } = ML;
  const F = [
    { name: 'Respiratory rate', short: 'RR', unit: '/min', dom: [8, 42] },
    { name: 'Systolic BP', short: 'SBP', unit: 'mmHg', dom: [65, 185] },
  ];

  // ---------- simulated patients ----------

  function cohort(seed, n) {
    const r = ML.rng(seed);
    return Array.from({ length: n }, () => {
      const sev = ML.randn(r); // shared illness severity couples fast breathing with low pressure
      const rr = Math.round(clamp(20 + 3.5 * sev + 3.8 * ML.randn(r), 10, 40));
      const sbp = Math.round(clamp(120 - 11 * sev + 19 * ML.randn(r), 70, 180));
      const a = ML.sigmoid((rr - 22.5) / 1.0), b = ML.sigmoid((100.5 - sbp) / 3.5);
      const logit = -3.4 + 3.4 * a + 3.4 * b + 0.6 * a * b;
      return { x: [rr, sbp], y: r() < ML.sigmoid(logit) ? 1 : 0 };
    });
  }
  const train = cohort(21, 300);
  const test = cohort(22, 120);

  // ---------- CART ----------

  // Gini impurity of a group with `pos` cases out of `n`: 0 when pure, 0.5 at 50/50.
  const gini = (pos, n) => { if (!n) return 0; const p = pos / n; return 2 * p * (1 - p); };
  const leftLabel = (f, t) => `${F[f].short} ≤ ${Math.floor(t)}`;
  const rightLabel = (f, t) => `${F[f].short} ≥ ${Math.floor(t) + 1}`;

  // Every cut between distinct values of feature f, with the size-weighted Gini of the two sides.
  function scan(pts, idx, f) {
    const s = idx.slice().sort((a, b) => pts[a].x[f] - pts[b].x[f]);
    const n = s.length;
    let P = 0;
    for (const i of s) P += pts[i].y;
    const out = [];
    let nl = 0, pl = 0;
    for (let k = 0; k < n - 1; k++) {
      nl++;
      pl += pts[s[k]].y;
      const v = pts[s[k]].x[f], v2 = pts[s[k + 1]].x[f];
      if (v === v2) continue;
      const nr = n - nl, pr = P - pl;
      out.push({ f, t: (v + v2) / 2, nl, pl, nr, pr, wg: (nl * gini(pl, nl) + nr * gini(pr, nr)) / n });
    }
    return out;
  }
  function bestSplit(pts, idx, minLeaf) {
    let best = null;
    for (const f of [0, 1]) for (const c of scan(pts, idx, f)) {
      if (c.nl < minLeaf || c.nr < minLeaf) continue;
      if (!best || c.wg < best.wg - 1e-12) best = c;
    }
    return best;
  }

  let nextId = 1;
  function makeNode(pts, idx, depth, box, parent) {
    let pos = 0;
    for (const i of idx) pos += pts[i].y;
    return { id: nextId++, idx, n: idx.length, pos, risk: idx.length ? pos / idx.length : 0, gini: gini(pos, idx.length), depth, box, parent, split: null, left: null, right: null };
  }
  function applySplit(pts, node, s) {
    const L = [], R = [];
    for (const i of node.idx) (pts[i].x[s.f] <= s.t ? L : R).push(i);
    const bl = node.box.slice(), br = node.box.slice();
    // box = [x0, x1, y0, y1]; feature 0 is x, feature 1 is y
    if (s.f === 0) { bl[1] = s.t; br[0] = s.t; } else { bl[3] = s.t; br[2] = s.t; }
    node.split = { f: s.f, t: s.t };
    node.left = makeNode(pts, L, node.depth + 1, bl, node);
    node.right = makeNode(pts, R, node.depth + 1, br, node);
  }
  const fullBox = () => [F[0].dom[0], F[0].dom[1], F[1].dom[0], F[1].dom[1]];
  function build(pts, maxDepth, minLeaf) {
    const root = makeNode(pts, pts.map((_, i) => i), 0, fullBox(), null);
    (function grow(node) {
      if (node.depth >= maxDepth || node.gini === 0) return;
      const s = bestSplit(pts, node.idx, minLeaf);
      if (!s || node.gini - s.wg < 1e-9) return;
      applySplit(pts, node, s);
      grow(node.left);
      grow(node.right);
    })(root);
    return root;
  }
  function leaves(node, out = []) {
    if (!node.split) out.push(node); else { leaves(node.left, out); leaves(node.right, out); }
    return out;
  }
  function leafFor(node, x) {
    while (node.split) node = x[node.split.f] <= node.split.t ? node.left : node.right;
    return node;
  }
  function pathTo(node) { const p = []; while (node) { p.push(node.id); node = node.parent; } return new Set(p); }
  function aucOf(root, pts) {
    return ML.auc(pts.map((p) => leafFor(root, p.x).risk), pts.map((p) => p.y));
  }

  // Step 2 grows one split at a time: of every leaf's best split, the one that removes the most
  // impurity (Gini drop × patients). Each leaf's best split is worked out once and kept on it.
  function nextSplit(root, pts, maxDepth, minLeaf) {
    let best = null;
    for (const L of leaves(root)) {
      if (L.depth >= maxDepth || L.gini === 0) continue;
      if (L._best === undefined) L._best = bestSplit(pts, L.idx, minLeaf);
      const s = L._best;
      if (!s) continue;
      const gain = L.n * (L.gini - s.wg);
      if (gain > 1e-9 && (!best || gain > best.gain)) best = { node: L, s, gain };
    }
    return best;
  }

  function findNode(root, nid) {
    let hit = null;
    (function walk(n) { if (n.id === nid) hit = n; if (n.split) { walk(n.left); walk(n.right); } })(root);
    return hit;
  }

  // A leaf as a readable rule: the tightest bounds its path puts on each feature.
  function ruleFor(leaf) {
    const lo = [null, null], hi = [null, null];
    let n = leaf;
    while (n.parent) {
      const s = n.parent.split;
      if (n === n.parent.left) hi[s.f] = hi[s.f] === null ? Math.floor(s.t) : Math.min(hi[s.f], Math.floor(s.t));
      else lo[s.f] = lo[s.f] === null ? Math.floor(s.t) + 1 : Math.max(lo[s.f], Math.floor(s.t) + 1);
      n = n.parent;
    }
    const conds = [];
    for (const f of [0, 1]) {
      if (lo[f] !== null && hi[f] !== null) conds.push(`${F[f].short} ${lo[f]}–${hi[f]}`);
      else if (lo[f] !== null) conds.push(`${F[f].short} ≥ ${lo[f]}`);
      else if (hi[f] !== null) conds.push(`${F[f].short} ≤ ${hi[f]}`);
    }
    return conds.join(' and ') || 'everyone';
  }

  ML.tree = {
    F, cohort, train, test, gini, leftLabel, rightLabel, scan, bestSplit, makeNode, applySplit, fullBox,
    build, leaves, leafFor, pathTo, aucOf, nextSplit, findNode, ruleFor,
  };
})(globalThis.ML = globalThis.ML || {});
