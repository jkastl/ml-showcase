/* Chapter 3: simulated patients at diabetes diagnosis, built from four hidden groups, and
 * k-means (Lloyd's algorithm, k-means++ starts, silhouette). No DOM.
 *
 * Points are { x: age, y: HbA1c }. `sc` is the unit of distance for each axis: [1, 1] measures
 * raw years and HbA1c points, STD measures each in its own standard deviations.
 */
(function (ML) {
  'use strict';
  const { clamp } = ML;

  // ---------- simulated patients ----------

  const GROUPS = [
    { age: 34, sa: 6, a1c: 11.2, sb: 0.8, n: 60, name: 'A' },
    { age: 52, sa: 6, a1c: 9.3, sb: 0.55, n: 80, name: 'B' },
    { age: 50, sa: 7, a1c: 7.0, sb: 0.45, n: 90, name: 'C' },
    { age: 70, sa: 5, a1c: 6.9, sb: 0.5, n: 90, name: 'D' },
  ];
  const pts = [];
  (() => {
    const r = ML.rng(31);
    GROUPS.forEach((g, gi) => {
      for (let i = 0; i < g.n; i++) {
        pts.push({
          x: Math.round(clamp(g.age + g.sa * ML.randn(r), 20, 85)),
          y: Math.round(clamp(g.a1c + g.sb * ML.randn(r), 5.4, 13.5) * 10) / 10,
          g: gi,
        });
      }
    });
    ML.shuffle(pts, r);
  })();
  const STD = [ML.sd(pts.map((p) => p.x)), ML.sd(pts.map((p) => p.y))];
  const RAW = [1, 1];

  // A start that lands in a poor local minimum: one group ends up split, two others merged.
  const BAD_START = [{ x: 25, y: 9.1 }, { x: 51, y: 12.4 }, { x: 54, y: 11.2 }, { x: 42, y: 10 }];

  // ---------- k-means ----------

  const d2 = (p, c, sc) => { const dx = (p.x - c.x) / sc[0], dy = (p.y - c.y) / sc[1]; return dx * dx + dy * dy; };
  function nearest(p, cents, sc) {
    let best = 0, bd = Infinity;
    for (let k = 0; k < cents.length; k++) { const d = d2(p, cents[k], sc); if (d < bd) { bd = d; best = k; } }
    return best;
  }
  // The assign step: every point to its nearest center.
  const assign = (pts, cents, sc) => pts.map((p) => nearest(p, cents, sc));
  // The move step: every center to the mean of its points. A center with no points stays put.
  function centroids(pts, lab, cents) {
    return cents.map((c, k) => {
      let sx = 0, sy = 0, n = 0;
      for (let i = 0; i < pts.length; i++) if (lab[i] === k) { sx += pts[i].x; sy += pts[i].y; n++; }
      return n ? { x: sx / n, y: sy / n } : { x: c.x, y: c.y };
    });
  }
  // Total squared distance from each point to its center.
  function inertia(pts, lab, cents, sc) {
    let s = 0;
    for (let i = 0; i < pts.length; i++) s += d2(pts[i], cents[lab[i]], sc);
    return s;
  }
  // k-means++: each new center is a point drawn with probability proportional to its squared
  // distance from the nearest center so far, which spreads the starts apart.
  function kpp(pts, k, sc, r) {
    const cents = [Object.assign({}, pts[Math.floor(r() * pts.length)])];
    while (cents.length < k) {
      const ds = pts.map((p) => Math.min(...cents.map((c) => d2(p, c, sc))));
      let u = r() * ds.reduce((a, b) => a + b, 0), i = 0;
      while (u > ds[i] && i < ds.length - 1) u -= ds[i++];
      cents.push({ x: pts[i].x, y: pts[i].y });
    }
    return cents;
  }
  // Assign and move until nobody switches cluster.
  function run(pts, cents, sc) {
    let lab = assign(pts, cents, sc);
    for (let it = 0; it < 100; it++) {
      cents = centroids(pts, lab, cents);
      const nl = assign(pts, cents, sc);
      if (nl.every((v, i) => v === lab[i])) break;
      lab = nl;
    }
    return { cents, lab, inertia: inertia(pts, lab, cents, sc) };
  }
  // Relabel clusters by mean age so a cluster keeps its color as k or units change.
  function ordered(res) {
    const order = res.cents.map((c, k) => k).sort((a, b) => res.cents[a].x - res.cents[b].x || res.cents[a].y - res.cents[b].y);
    const inv = [];
    order.forEach((k, i) => { inv[k] = i; });
    return { cents: order.map((k) => res.cents[k]), lab: res.lab.map((l) => inv[l]), inertia: res.inertia };
  }
  // The lowest-inertia result of several k-means++ runs, as real software does it.
  function bestOf(pts, k, sc, runs, seed) {
    const r = ML.rng(seed);
    let best = null;
    for (let i = 0; i < runs; i++) {
      const res = run(pts, kpp(pts, k, sc, r), sc);
      if (!best || res.inertia < best.inertia) best = res;
    }
    return ordered(best);
  }
  // Mean silhouette: for each point, (b − a) / max(a, b), where a is its mean distance to its own
  // cluster and b its mean distance to the nearest other cluster. From −1 to 1; higher is better.
  function silhouette(pts, lab, k, sc) {
    if (k < 2) return NaN;
    let total = 0;
    for (let i = 0; i < pts.length; i++) {
      const sum = new Array(k).fill(0), cnt = new Array(k).fill(0);
      for (let j = 0; j < pts.length; j++) {
        if (i === j) continue;
        sum[lab[j]] += Math.sqrt(d2(pts[i], pts[j], sc));
        cnt[lab[j]]++;
      }
      const a = cnt[lab[i]] ? sum[lab[i]] / cnt[lab[i]] : 0;
      let b = Infinity;
      for (let c = 0; c < k; c++) if (c !== lab[i] && cnt[c]) b = Math.min(b, sum[c] / cnt[c]);
      total += cnt[lab[i]] ? (b - a) / Math.max(a, b) : 0;
    }
    return total / pts.length;
  }
  // Share of points in their true group (p.g) under the best one-to-one matching of 4 clusters
  // to 4 groups (all 24 permutations).
  function agreement(pts, lab) {
    const perms = [];
    (function permute(a, l) { if (l === a.length) { perms.push(a.slice()); return; } for (let i = l; i < a.length; i++) { [a[l], a[i]] = [a[i], a[l]]; permute(a, l + 1); [a[l], a[i]] = [a[i], a[l]]; } })([0, 1, 2, 3], 0);
    let best = 0;
    for (const pm of perms) {
      let ok = 0;
      for (let i = 0; i < pts.length; i++) if (pm[lab[i]] === pts[i].g) ok++;
      best = Math.max(best, ok);
    }
    return best / pts.length;
  }

  ML.kmeans = {
    GROUPS, pts, STD, RAW, BAD_START,
    d2, nearest, assign, centroids, inertia, kpp, run, ordered, bestOf, silhouette, agreement,
  };
})(globalThis.ML = globalThis.ML || {});
