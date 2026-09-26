/* Chapter 2: a CART decision tree for 48-hour deterioration in suspected infection. */
(function () {
  'use strict';
  const { id, text, fmt, pct, clamp } = ML;
  const T = ML.theme;
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
  const ALL = train.map((_, i) => i);

  // ---------- CART ----------

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

  // ---------- drawing ----------

  function scatter(host, extra = {}) {
    return new ML.Plot(host, Object.assign({
      aspect: 0.62, maxHeight: 440, minHeight: 250, x: F[0].dom, y: F[1].dom,
      xLabel: 'Respiratory rate (breaths/min)', yLabel: 'Systolic BP (mmHg)',
      label: 'Scatter plot of emergency patients by respiratory rate and systolic blood pressure',
    }, extra));
  }
  function drawPts(p, pts, opt = {}) {
    const c = p.ctx, r = opt.r || 4;
    for (const q of pts) {
      const inside = !opt.box || (q.x[0] >= opt.box[0] && q.x[0] <= opt.box[1] && q.x[1] >= opt.box[2] && q.x[1] <= opt.box[3]);
      const col = q.y ? T.s2 : T.s1;
      ML.marker(c, 'circle', p.sx(q.x[0]), p.sy(q.x[1]), r, inside ? col : ML.rgba(col, 0.2), inside ? T.surface : null);
    }
  }
  function drawLeaves(p, root, opt = {}) {
    const c = p.ctx;
    for (const L of leaves(root)) {
      const [x0, x1, y0, y1] = L.box;
      const X0 = p.sx(x0), X1 = p.sx(x1), Y0 = p.sy(y1), Y1 = p.sy(y0);
      c.fillStyle = ML.rgba(T.s2, 0.04 + 0.5 * L.risk);
      c.fillRect(X0, Y0, X1 - X0, Y1 - Y0);
      c.strokeStyle = T.axis;
      c.lineWidth = 1;
      c.strokeRect(X0 + 0.5, Y0 + 0.5, X1 - X0 - 1, Y1 - Y0 - 1);
    }
  }
  function drawLeafLabels(p, root) {
    const c = p.ctx;
    c.font = '600 11px ' + ML.FONT;
    c.textAlign = 'center';
    c.textBaseline = 'middle';
    for (const L of leaves(root)) {
      const [x0, x1, y0, y1] = L.box;
      const X0 = p.sx(x0), X1 = p.sx(x1), Y0 = p.sy(y1), Y1 = p.sy(y0);
      if (X1 - X0 < 40 || Y1 - Y0 < 22) continue;
      const s = pct(L.risk), w = c.measureText(s).width + 10, cx = (X0 + X1) / 2, cy = (Y0 + Y1) / 2;
      c.fillStyle = ML.rgba(T.surface, 0.85);
      c.beginPath();
      c.roundRect(cx - w / 2, cy - 9, w, 18, 5);
      c.fill();
      c.fillStyle = T.ink;
      c.fillText(s, cx, cy + 0.5);
    }
  }
  function outlineBox(p, box, color) {
    const c = p.ctx;
    const X0 = p.sx(box[0]), X1 = p.sx(box[1]), Y0 = p.sy(box[3]), Y1 = p.sy(box[2]);
    c.strokeStyle = color;
    c.lineWidth = 2;
    c.strokeRect(X0 + 1, Y0 + 1, X1 - X0 - 2, Y1 - Y0 - 2);
  }
  function tipFor(q) {
    return `<span class="k">RR</span> ${q.x[0]}/min · <span class="k">SBP</span> ${q.x[1]} mmHg<br>${q.y ? 'Deteriorated' : 'Stable'}`;
  }
  function hoverPts(plot, pts) {
    plot.on('move', (e) => {
      if (e.dragging) return;
      const i = plot.nearest(pts, e, (q) => q.x[0], (q) => q.x[1], 10);
      if (i >= 0) plot.showTip(e.px, e.py, tipFor(pts[i])); else plot.hideTip();
    });
  }

  // Tree diagram as SVG: leaves spread evenly, parents centered over children.
  const NW = 124, NH = 50, COL = 132, ROW = 84;
  function renderTree(host, root, opt = {}) {
    const ls = leaves(root);
    const pos = new Map();
    let k = 0, maxDepth = 0;
    (function place(n) {
      maxDepth = Math.max(maxDepth, n.depth);
      if (!n.split) { pos.set(n.id, (k++ + 0.5) * COL); return; }
      place(n.left); place(n.right);
      pos.set(n.id, (pos.get(n.left.id) + pos.get(n.right.id)) / 2);
    })(root);
    const W = Math.max(ls.length * COL, NW + 10), H = maxDepth * ROW + NH + 8;
    const onPath = opt.path || new Set();
    const parts = [];
    (function edges(n) {
      if (!n.split) return;
      for (const [ch, lbl, side] of [[n.left, leftLabel(n.split.f, n.split.t), -1], [n.right, rightLabel(n.split.f, n.split.t), 1]]) {
        const x1 = pos.get(n.id), y1 = n.depth * ROW + NH + 4, x2 = pos.get(ch.id), y2 = ch.depth * ROW + 4;
        const on = onPath.has(n.id) && onPath.has(ch.id);
        parts.push(`<path class="tedge${on ? ' path' : ''}" d="M${x1},${y1} C${x1},${(y1 + y2) / 2} ${x2},${(y1 + y2) / 2} ${x2},${y2}"/>`);
        const mx = (x1 + x2) / 2 + side * 4, my = (y1 + y2) / 2;
        parts.push(`<text class="tlabel" x="${mx}" y="${my}" text-anchor="${side < 0 ? 'end' : 'start'}" dominant-baseline="middle">${lbl}</text>`);
        edges(ch);
      }
    })(root);
    (function nodes(n) {
      const x = pos.get(n.id) - NW / 2, y = n.depth * ROW + 4;
      const cls = ['tnode'];
      if (opt.sel === n.id) cls.push('sel');
      if (onPath.has(n.id)) cls.push('path');
      const title = n.split ? `${F[n.split.f].name}?` : `Leaf: risk ${pct(n.risk)}`;
      const sub = `${n.n} patients · ${pct(n.risk)}`;
      parts.push(`<g class="${cls.join(' ')}" data-id="${n.id}" tabindex="0" role="button" aria-label="${title} ${sub}">` +
        `<rect class="box" x="${x}" y="${y}" width="${NW}" height="${NH}" rx="7"/>` +
        `<text class="q" x="${x + 8}" y="${y + 16}">${title}</text>` +
        `<text x="${x + 8}" y="${y + 30}">${sub}</text>` +
        `<rect x="${x + 8}" y="${y + 38}" width="${NW - 16}" height="5" rx="2" fill="var(--grid)"/>` +
        `<rect x="${x + 8}" y="${y + 38}" width="${Math.max(0.01, (NW - 16) * n.risk)}" height="5" rx="2" fill="var(--s2)"/>` +
        `</g>`);
      if (n.split) { nodes(n.left); nodes(n.right); }
    })(root);
    host.innerHTML = `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="group" aria-label="Decision tree diagram">${parts.join('')}</svg>`;
    // Keep the node of interest in view when the tree is wider than its panel.
    const focus = opt.sel || (onPath.size ? [...onPath][0] : root.id);
    if (W > host.clientWidth && pos.has(focus)) host.scrollLeft = pos.get(focus) - host.clientWidth / 2;
    if (opt.onClick) {
      for (const g of host.querySelectorAll('.tnode')) {
        const go = () => opt.onClick(+g.dataset.id);
        g.addEventListener('click', go);
        g.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); go(); } });
      }
    }
  }
  function findNode(root, nid) {
    let hit = null;
    (function walk(n) { if (n.id === nid) hit = n; if (n.split) { walk(n.left); walk(n.right); } })(root);
    return hit;
  }

  // ---------- step 1: one split ----------

  const S1 = { f: 0, t: 16.5, best: null };
  const rootG = gini(train.filter((q) => q.y).length, train.length);
  const scans = [scan(train, ALL, 0), scan(train, ALL, 1)];
  const bestFor = (f) => scans[f].reduce((a, b) => (b.wg < a.wg ? b : a));
  function cutAt(f, t) {
    // Nearest real cut at or around t (cuts sit between observed values).
    let best = scans[f][0], bd = Infinity;
    for (const c of scans[f]) { const d = Math.abs(c.t - t); if (d < bd) { bd = d; best = c; } }
    return best;
  }

  const featSeg = ML.seg('tr-feat', (v) => {
    S1.f = +v;
    S1.t = S1.f === 0 ? 16.5 : 140.5;
    S1.best = null;
    split1Changed();
  });
  ML.btn('tr-best', () => { const b = bestFor(S1.f); S1.t = b.t; S1.best = b; split1Changed(); });

  const pSplit = scatter('tr-split', { drag: true });
  pSplit.draw = (c, p) => {
    const cut = cutAt(S1.f, S1.t);
    p.axes();
    p.clip();
    const X = p.sx(cut.t), Y = p.sy(cut.t);
    const rl = cut.pl / cut.nl, rr = cut.pr / cut.nr;
    c.fillStyle = ML.rgba(T.s2, 0.04 + 0.45 * rl);
    if (S1.f === 0) c.fillRect(p.m.l, p.m.t, X - p.m.l, p.ih); else c.fillRect(p.m.l, Y, p.iw, p.m.t + p.ih - Y);
    c.fillStyle = ML.rgba(T.s2, 0.04 + 0.45 * rr);
    if (S1.f === 0) c.fillRect(X, p.m.t, p.m.l + p.iw - X, p.ih); else c.fillRect(p.m.l, p.m.t, p.iw, Y - p.m.t);
    drawPts(p, train);
    c.strokeStyle = T.ink;
    c.lineWidth = 2;
    c.beginPath();
    if (S1.f === 0) { c.moveTo(X, p.m.t); c.lineTo(X, p.m.t + p.ih); } else { c.moveTo(p.m.l, Y); c.lineTo(p.m.l + p.iw, Y); }
    c.stroke();
    c.restore();
    // grab handle
    if (S1.f === 0) ML.marker(c, 'circle', X, p.m.t + 8, 7, T.ink, T.surface);
    else ML.marker(c, 'circle', p.m.l + p.iw - 8, Y, 7, T.ink, T.surface);
    c.fillStyle = T.ink;
    c.font = '600 11px ' + ML.FONT;
    c.textBaseline = 'middle';
    if (S1.f === 0) {
      c.textAlign = 'right'; c.fillText(leftLabel(0, cut.t), X - 12, p.m.t + 8);
      c.textAlign = 'left'; c.fillText(rightLabel(0, cut.t), X + 12, p.m.t + 8);
    } else {
      c.textAlign = 'right';
      c.fillText(rightLabel(1, cut.t), p.m.l + p.iw - 20, Y - 11);
      c.fillText(leftLabel(1, cut.t), p.m.l + p.iw - 20, Y + 11);
    }
  };
  const dragSplit = (e) => { S1.t = S1.f === 0 ? e.x : e.y; S1.best = null; split1Changed(); };
  pSplit.on('down', (e) => { if (e.inside) dragSplit(e); });
  pSplit.on('move', (e) => { if (e.dragging) { pSplit.hideTip(); dragSplit(e); return; } const i = pSplit.nearest(train, e, (q) => q.x[0], (q) => q.x[1], 10); if (i >= 0) pSplit.showTip(e.px, e.py, tipFor(train[i])); else pSplit.hideTip(); });

  const pImp = new ML.Plot('tr-imp', {
    height: 190, x: F[0].dom, y: [0, Math.ceil(rootG * 11) / 10], drag: 'x', yTicks: 3,
    margin: { l: 40, b: 36, t: 10, r: 12 }, label: 'Weighted Gini impurity at every threshold',
  });
  pImp.draw = (c, p) => {
    const f = S1.f, sc = scans[f];
    p.o.x = F[f].dom;
    p.o.xLabel = F[f].name + ' cut';
    p.axes();
    p.refLine(F[f].dom[0], rootG, F[f].dom[1], rootG, `before split ${fmt(rootG)}`, 'right');
    p.clip();
    const xs = [], ys = [];
    for (let k = 0; k < sc.length; k++) {
      const a = k ? (sc[k - 1].t + sc[k].t) / 2 : F[f].dom[0];
      const b = k < sc.length - 1 ? (sc[k].t + sc[k + 1].t) / 2 : F[f].dom[1];
      xs.push(a, b); ys.push(sc[k].wg, sc[k].wg);
    }
    p.line(xs, ys, T.s1, 2);
    const cut = cutAt(f, S1.t);
    c.restore();
    ML.marker(c, 'circle', p.sx(cut.t), p.sy(cut.wg), 5.5, T.ink, T.surface);
    if (S1.best) {
      c.fillStyle = T['ink-2'];
      c.font = '11px ' + ML.FONT;
      c.textAlign = 'center';
      c.fillText('lowest', p.sx(cut.t), p.sy(cut.wg) + 16);
    }
  };
  const dragImp = (e) => { S1.t = e.x; S1.best = null; split1Changed(); };
  pImp.on('down', (e) => { if (e.inside) dragImp(e); });
  pImp.on('move', (e) => { if (e.dragging) dragImp(e); });

  function split1Changed() {
    const cut = cutAt(S1.f, S1.t);
    const side = (lbl, n, pos) => `
      <div style="margin-bottom:.7rem">
        <div style="display:flex;justify-content:space-between;font-size:.88rem"><b>${lbl}</b><span>${n} patients</span></div>
        <div style="display:flex;height:12px;border-radius:0 4px 4px 0;overflow:hidden;margin:.3rem 0;gap:2px;background:var(--surface)">
          <span style="flex:${n - pos};background:var(--s1)"></span><span style="flex:${pos};background:var(--s2)"></span>
        </div>
        <div style="font-size:.82rem;color:var(--ink-2)">${pct(pos / n)} deteriorated · Gini ${fmt(gini(pos, n))}</div>
      </div>`;
    const drop = rootG - cut.wg;
    const bestAll = Math.min(bestFor(0).wg, bestFor(1).wg);
    id('tr-sides').innerHTML =
      side(leftLabel(S1.f, cut.t), cut.nl, cut.pl) + side(rightLabel(S1.f, cut.t), cut.nr, cut.pr) +
      `<div class="stats"><div class="stat"><span class="stat-label">Weighted impurity</span><span class="stat-value">${fmt(cut.wg, 3)}</span><span class="stat-note">down ${fmt(drop, 3)} from ${fmt(rootG, 3)}</span></div></div>` +
      `<p class="callout">${cut.wg <= bestAll + 1e-9 ? '<span class="good-text">That’s the best cut on either feature. It becomes the root of the tree.</span>' : cut.wg <= bestFor(S1.f).wg + 1e-9 ? 'Best cut for this vital sign. Is the other one better?' : 'Keep dragging: a lower impurity is possible.'}</p>`;
    pSplit.update();
    pImp.update();
  }

  // ---------- step 2: grow ----------

  const S2 = { root: null, history: [], sel: null };
  const MIN_LEAF2 = 8, MAX_D2 = 4;
  function growReset() {
    S2.root = makeNode(train, ALL, 0, fullBox(), null);
    S2.history = [];
    S2.sel = S2.root.id;
    growChanged();
  }
  function nextCandidate() {
    let best = null;
    for (const L of leaves(S2.root)) {
      if (L.depth >= MAX_D2 || L.gini === 0) continue;
      if (L._best === undefined) L._best = bestSplit(train, L.idx, MIN_LEAF2);
      const s = L._best;
      if (!s) continue;
      const gain = L.n * (L.gini - s.wg);
      if (gain > 1e-9 && (!best || gain > best.gain)) best = { node: L, s, gain };
    }
    return best;
  }
  function growOne() {
    const c = nextCandidate();
    if (!c) return false;
    applySplit(train, c.node, c.s);
    S2.history.push(c.node);
    S2.sel = c.node.id;
    return true;
  }
  ML.btn('tr-next', () => { growOne(); growChanged(); });
  ML.btn('tr-undo', () => {
    const n = S2.history.pop();
    if (!n) return;
    n.split = n.left = n.right = null;
    S2.sel = n.id;
    growChanged();
  });
  ML.btn('tr-full', () => { while (growOne()); growChanged(); });
  ML.btn('tr-reset', growReset);

  const pGrow = scatter('tr-grow');
  pGrow.draw = (c, p) => {
    p.axes();
    p.clip();
    drawLeaves(p, S2.root);
    const sel = S2.sel && findNode(S2.root, S2.sel);
    drawPts(p, train, { r: 3.5, box: sel && sel !== S2.root ? sel.box : null });
    drawLeafLabels(p, S2.root);
    if (sel && sel !== S2.root) outlineBox(p, sel.box, T.ink);
    c.restore();
  };
  hoverPts(pGrow, train);

  function growChanged() {
    const nl = leaves(S2.root).length;
    const next = nextCandidate();
    id('tr-next').disabled = !next;
    id('tr-full').disabled = !next;
    id('tr-undo').disabled = !S2.history.length;
    text('tr-grow-info', `${S2.history.length} question${S2.history.length === 1 ? '' : 's'} · ${nl} leaves · training AUC ${fmt(aucOf(S2.root, train))}` +
      (next ? '' : ' · done (depth limit 4, minimum leaf size 8)'));
    renderTree(id('tr-diagram'), S2.root, { sel: S2.sel, onClick: (nid) => { S2.sel = nid; growChanged(); } });
    pGrow.update();
  }

  // ---------- step 3: overfitting ----------

  const S3 = { depth: 3, minLeaf: 5, root: null, curve: [] };
  const depthR = ML.range('tr-depth', (v) => { S3.depth = v; overChanged(); });
  const leafR = ML.range('tr-minleaf', (v) => { S3.minLeaf = v; overChanged(true); });
  function overChanged(recurve) {
    S3.root = build(train, S3.depth, S3.minLeaf);
    if (recurve || !S3.curve.length) {
      S3.curve = [];
      for (let d = 1; d <= 12; d++) {
        const r = build(train, d, S3.minLeaf);
        S3.curve.push({ d, tr: aucOf(r, train), te: aucOf(r, test) });
      }
    }
    const pt = S3.curve[S3.depth - 1];
    text('tr-leaves', leaves(S3.root).length);
    text('tr-auc-train', fmt(pt.tr));
    text('tr-auc-test', fmt(pt.te));
    pOver.update();
    pAuc.update();
  }
  const pOver = scatter('tr-over', { aspect: 0.8, maxHeight: 360, margin: { l: 44, r: 10 } });
  pOver.draw = (c, p) => {
    p.axes();
    p.clip();
    drawLeaves(p, S3.root);
    drawPts(p, train, { r: 3 });
    c.restore();
  };
  hoverPts(pOver, train);
  const pAuc = new ML.Plot('tr-auc', {
    aspect: 0.8, maxHeight: 360, x: [0.5, 12.5], y: [0.5, 1], xLabel: 'maximum depth', yLabel: 'AUC', xTicks: 6,
    margin: { l: 44, r: 12 }, drag: 'x', label: 'Training and held-out AUC by tree depth',
  });
  pAuc.draw = (c, p) => {
    p.axes({ xTicks: [1, 2, 4, 6, 8, 10, 12] });
    const X = p.sx(S3.depth);
    c.fillStyle = ML.rgba(T.ink, 0.06);
    c.fillRect(X - 10, p.m.t, 20, p.ih);
    const cv = S3.curve;
    p.line(cv.map((q) => q.d), cv.map((q) => q.tr), T.s1, 2);
    p.line(cv.map((q) => q.d), cv.map((q) => q.te), T.s2, 2);
    const pk = cv.reduce((a, b) => (b.te > a.te ? b : a));
    for (const q of cv) {
      ML.marker(c, 'circle', p.sx(q.d), p.sy(q.tr), q.d === S3.depth ? 5 : 3, T.s1, T.surface);
      ML.marker(c, 'circle', p.sx(q.d), p.sy(q.te), q.d === S3.depth ? 5 : 3, T.s2, T.surface);
    }
    c.fillStyle = T['ink-2'];
    c.font = '11px ' + ML.FONT;
    c.textAlign = 'center';
    c.fillText('held-out peak', p.sx(pk.d), p.sy(pk.te) + 22);
  };
  const pickDepth = (e) => { const d = clamp(Math.round(e.x), 1, 12); if (d !== S3.depth) { depthR.set(d); S3.depth = d; overChanged(); } };
  pAuc.on('down', pickDepth);
  pAuc.on('move', (e) => {
    if (e.dragging) { pickDepth(e); return; }
    const q = S3.curve[clamp(Math.round(e.x), 1, 12) - 1];
    if (q && e.inside) pAuc.showTip(e.px, e.py, `<b>Depth ${q.d}</b><br><span class="k">Training</span> ${fmt(q.tr)}<br><span class="k">Held out</span> ${fmt(q.te)}<br><span class="k">tap to select</span>`);
    else pAuc.hideTip();
  });

  // ---------- step 4: trace one patient ----------

  const S4 = { root: build(train, 3, 15), rr: 26, sbp: 96, qsofa: false };
  const rrR = ML.range('tr-rr', (v) => { S4.rr = v; traceChanged(); }, (v) => v + '/min');
  const sbpR = ML.range('tr-sbp', (v) => { S4.sbp = v; traceChanged(); }, (v) => v + ' mmHg');
  id('tr-qsofa').addEventListener('change', (e) => { S4.qsofa = e.target.checked; traceChanged(); });

  const pTrace = scatter('tr-trace', { aspect: 0.8, maxHeight: 400, drag: true, margin: { l: 44, r: 10 } });
  pTrace.draw = (c, p) => {
    p.axes();
    p.clip();
    drawLeaves(p, S4.root);
    drawPts(p, train, { r: 2.5 });
    drawLeafLabels(p, S4.root);
    const leaf = leafFor(S4.root, [S4.rr, S4.sbp]);
    outlineBox(p, leaf.box, T.ink);
    c.restore();
    if (S4.qsofa) {
      p.refLine(22, F[1].dom[0], 22, F[1].dom[1], 'qSOFA RR 22');
      p.refLine(F[0].dom[0], 100, F[0].dom[1], 100, 'qSOFA SBP 100', 'right');
    }
    const X = p.sx(S4.rr), Y = p.sy(S4.sbp);
    c.beginPath();
    c.arc(X, Y, 11, 0, Math.PI * 2);
    c.lineWidth = 2.5;
    c.strokeStyle = T.ink;
    c.stroke();
    ML.marker(c, 'circle', X, Y, 5, T.ink, T.surface);
  };
  const dragProbe = (e) => {
    S4.rr = Math.round(clamp(e.x, 10, 40));
    S4.sbp = Math.round(clamp(e.y, 70, 180));
    rrR.set(S4.rr); sbpR.set(S4.sbp);
    traceChanged();
  };
  pTrace.on('down', (e) => { if (e.inside) dragProbe(e); });
  pTrace.on('move', (e) => { if (e.dragging) dragProbe(e); });

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
  function traceChanged() {
    const leaf = leafFor(S4.root, [S4.rr, S4.sbp]);
    const q1 = S4.rr >= 22, q2 = S4.sbp <= 100;
    const tick = (b) => (b ? '<span class="bad-text">✓ met</span>' : '<span style="color:var(--muted)">✗ not met</span>');
    id('tr-trace-out').innerHTML = `
      <div class="stats">
        <div class="stat"><span class="stat-label">Tree's predicted risk</span><span class="stat-value">${pct(leaf.risk)}</span><span class="stat-note">${leaf.pos} of ${leaf.n} similar patients</span></div>
      </div>
      <p class="callout"><b>qSOFA-style check</b><br>RR ≥ 22: ${tick(q1)}<br>SBP ≤ 100: ${tick(q2)}<br>${q1 + q2} of these 2 criteria${q1 && q2 ? ' (qSOFA ≥ 2 flags high risk)' : ''}</p>`;
    const ls = leaves(S4.root);
    id('tr-rules').innerHTML = ls.map((L) =>
      `<span class="${L === leaf ? 'hit' : ''}">${L === leaf ? '▶ ' : '  '}IF ${ruleFor(L)}\n     THEN risk ${pct(L.risk)}  (${L.pos}/${L.n} patients)</span>`).join('\n');
    renderTree(id('tr-diagram2'), S4.root, { path: pathTo(leaf) });
    pTrace.update();
  }

  // ---------- routing ----------

  ML.register('triage', {
    enter(step) {
      if (step === 0) { featSeg.set(String(S1.f)); split1Changed(); }
      if (step === 1) { if (!S2.root) growReset(); else growChanged(); }
      if (step === 2) overChanged(true);
      if (step === 3) traceChanged();
    },
  });
  void leafR;
})();
