/* Chapter 1: logistic regression for 5-year type 2 diabetes risk. */
(function () {
  'use strict';
  const { id, text, fmt, pct, clamp } = ML;
  const T = ML.theme;
  const G = [50, 205]; // glucose axis (mg/dL)
  const B = [15, 50]; // BMI axis
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

  const S = {
    added: [],
    dataVersion: 0,
    w1: 0, w2: 0, b: 0, // hand-fit weights (step 2)
    handVersion: 0,
    best: Infinity,
    hover: -1,
    thr: 0.5,
    recal: null,
  };
  const train = () => base.concat(S.added);
  let active = 0;

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
  function gradient(pts, w1, w2, b) {
    let g1 = 0, g2 = 0, g0 = 0;
    for (const p of pts) {
      const e = prob(p, w1, w2, b) - p.y;
      g1 += e * xg(p); g2 += e * xb(p); g0 += e;
    }
    const n = pts.length;
    return [g1 / n, g2 / n, g0 / n];
  }
  let fitted = null;
  function model() {
    if (fitted && fitted.v === S.dataVersion) return fitted;
    const pts = train();
    let w1 = 0, w2 = 0, b = 0;
    for (let i = 0; i < 3000; i++) {
      const [g1, g2, g0] = gradient(pts, w1, w2, b);
      w1 -= 1.5 * g1; w2 -= 1.5 * g2; b -= 1.5 * g0;
    }
    fitted = { v: S.dataVersion, w1, w2, b, loss: logLoss(pts, w1, w2, b) };
    return fitted;
  }

  // ---------- shared drawing ----------

  function scatterPlot(host, extra = {}) {
    return new ML.Plot(host, Object.assign({
      aspect: 0.62, maxHeight: 440, minHeight: 250, x: G, y: B,
      xLabel: '2-hour glucose (mg/dL)', yLabel: 'BMI (kg/m²)',
      label: 'Scatter plot of patients by glucose and BMI, colored by outcome',
    }, extra));
  }
  function drawShading(p, w1, w2, b) {
    const c = p.ctx, cell = 6;
    for (let px = p.m.l; px < p.m.l + p.iw; px += cell) {
      for (let py = p.m.t; py < p.m.t + p.ih; py += cell) {
        const q = ML.sigmoid(w1 * ((p.ix(px + cell / 2) - MG) / SG) + w2 * ((p.iy(py + cell / 2) - MB) / SB) + b);
        const a = Math.abs(q - 0.5) * 2 * 0.32;
        if (a < 0.01) continue;
        c.fillStyle = ML.rgba(q > 0.5 ? T.s2 : T.s1, a);
        c.fillRect(px, py, cell, cell);
      }
    }
  }
  function drawBoundary(p, w1, w2, b) {
    // Line where z = 0, in raw units: a*g + c*bmi + d = 0.
    const a = w1 / SG, cc = w2 / SB, d = b - (w1 * MG) / SG - (w2 * MB) / SB;
    if (Math.abs(a) < 1e-9 && Math.abs(cc) < 1e-9) return;
    let xs, ys;
    if (Math.abs(cc * (B[1] - B[0])) > Math.abs(a * (G[1] - G[0]))) {
      xs = [G[0] - 50, G[1] + 50];
      ys = xs.map((g) => -(a * g + d) / cc);
    } else {
      ys = [B[0] - 50, B[1] + 50];
      xs = ys.map((v) => -(cc * v + d) / a);
    }
    p.line(xs, ys, T.ink, 2);
  }
  function drawPatients(p, pts, hi) {
    const c = p.ctx;
    for (let i = 0; i < pts.length; i++) {
      const q = pts[i];
      ML.marker(c, 'circle', p.sx(q.g), p.sy(q.bmi), q.added ? 5 : 4, q.y ? T.s2 : T.s1, q.added ? T.ink : T.surface);
    }
    if (hi >= 0 && pts[hi]) {
      const q = pts[hi];
      c.beginPath();
      c.arc(p.sx(q.g), p.sy(q.bmi), 9, 0, Math.PI * 2);
      c.lineWidth = 2;
      c.strokeStyle = T.ink;
      c.stroke();
    }
  }
  function patientTip(q, i, extra = '') {
    return `<b>${q.added ? 'Your patient' : 'Patient ' + (i + 1)}</b><br>` +
      `<span class="k">2-h glucose</span> ${q.g} mg/dL<br><span class="k">BMI</span> ${q.bmi.toFixed(1)}<br>` +
      `${q.y ? 'Developed diabetes' : 'No diabetes at 5 years'}${extra}`;
  }
  function hoverPatients(p, e) {
    const pts = train();
    const i = p.nearest(pts, e, (q) => q.g, (q) => q.bmi, 12);
    return { pts, i };
  }

  // ---------- step 1: the data ----------

  const addSeg = ML.seg('lr-add', () => {});
  const p1 = scatterPlot('lr-data');
  p1.draw = (c, p) => {
    p.axes();
    p.clip();
    drawPatients(p, train(), S.hover);
    c.restore();
    p.refLine(140, B[0], 140, B[1], 'Impaired tolerance ≥ 140');
  };
  p1.on('move', (e) => {
    const { pts, i } = hoverPatients(p1, e);
    if (i !== S.hover) { S.hover = i; p1.update(); }
    if (i >= 0) p1.showTip(e.px, e.py, patientTip(pts[i], i)); else p1.hideTip();
  });
  p1.on('leave', () => { S.hover = -1; p1.update(); });
  p1.on('down', (e) => {
    if (!e.inside) return;
    S.added.push({ g: Math.round(e.x), bmi: Math.round(e.y * 10) / 10, y: +addSeg.value, added: true });
    dataChanged();
  });
  ML.btn('lr-reset-data', () => { S.added = []; dataChanged(); });

  function dataChanged() {
    S.dataVersion++;
    S.best = Infinity;
    updateStats1();
    p1.update();
  }
  function updateStats1() {
    const pts = train();
    const hi = pts.filter((p) => p.g >= 140), lo = pts.filter((p) => p.g < 140);
    const rate = (a) => (a.length ? a.filter((p) => p.y).length / a.length : NaN);
    text('lr-n', pts.length);
    text('lr-prev', pct(rate(pts)));
    text('lr-igt-rate', pct(rate(hi)));
    text('lr-norm-rate', pct(rate(lo)));
  }

  // ---------- step 2: fit by hand ----------

  const p2 = scatterPlot('lr-fit');
  p2.draw = (c, p) => {
    p.axes();
    p.clip();
    drawShading(p, S.w1, S.w2, S.b);
    drawBoundary(p, S.w1, S.w2, S.b);
    drawPatients(p, train(), S.hover);
    c.restore();
  };
  p2.on('move', (e) => {
    const { pts, i } = hoverPatients(p2, e);
    if (i !== S.hover) { S.hover = i; p2.update(); pSig.update(); explain(i >= 0 ? pts[i] : null); }
    if (i >= 0) p2.showTip(e.px, e.py, patientTip(pts[i], i, `<br><span class="k">Predicted risk</span> ${pct(prob(pts[i], S.w1, S.w2, S.b))}`));
    else p2.hideTip();
  });
  p2.on('leave', () => { S.hover = -1; p2.update(); pSig.update(); explain(null); });

  const pSig = new ML.Plot('lr-sig', {
    height: 170, x: [-6, 6], y: [-0.08, 1.08], xLabel: 'score z', yTicks: 2, xTicks: 6,
    margin: { l: 40, b: 36, t: 8, r: 8 }, label: 'Sigmoid curve with patients placed by score',
  });
  pSig.draw = (c, p) => {
    p.axes({ yTicks: [0, 0.5, 1] });
    p.clip();
    const zs = [], ps = [];
    for (let z = -6; z <= 6.001; z += 0.1) { zs.push(z); ps.push(ML.sigmoid(z)); }
    const pts = train();
    const r = ML.rng(5);
    pts.forEach((q, i) => {
      const z = clamp(score(q, S.w1, S.w2, S.b), -5.9, 5.9);
      const jitter = (r() - 0.5) * 0.09;
      if (i === S.hover) return;
      ML.marker(c, 'circle', p.sx(z), p.sy(q.y + (q.y ? -0.04 : 0.04) + jitter), 2.5, ML.rgba(q.y ? T.s2 : T.s1, 0.75), null);
    });
    p.line(zs, ps, T.ink, 2);
    const q = pts[S.hover];
    if (q) {
      const z = clamp(score(q, S.w1, S.w2, S.b), -5.9, 5.9), pr = ML.sigmoid(z);
      c.strokeStyle = T.muted;
      c.lineWidth = 1;
      c.beginPath();
      c.moveTo(p.sx(z), p.sy(q.y)); c.lineTo(p.sx(z), p.sy(pr)); c.lineTo(p.m.l, p.sy(pr));
      c.stroke();
      ML.marker(c, 'circle', p.sx(z), p.sy(pr), 5, T.ink, T.surface);
      ML.marker(c, 'circle', p.sx(z), p.sy(q.y), 5, q.y ? T.s2 : T.s1, T.surface);
    }
    c.restore();
  };

  function explain(q) {
    const el = id('lr-formula');
    if (!q) { el.textContent = 'Hover a patient to see their score.'; return; }
    const a = xg(q), bb = xb(q), z = score(q, S.w1, S.w2, S.b), pr = ML.sigmoid(z);
    const pActual = q.y ? pr : 1 - pr;
    el.textContent =
      `z = ${fmt(S.w1)} × glucose(${ML.signed(a)} SD) + ${fmt(S.w2)} × BMI(${ML.signed(bb)} SD) ${S.b < 0 ? '−' : '+'} ${fmt(Math.abs(S.b))}\n` +
      `  = ${ML.signed(z)}   →   risk = sigmoid(z) = ${pct(pr)}\n` +
      `${q.y ? 'Developed diabetes' : 'No diabetes'}, so loss = −log(${fmt(pActual)}) = ${fmt(-Math.log(Math.max(1e-12, pActual)))}`;
  }

  function handChanged() {
    S.handVersion++;
    const pts = train();
    const L = logLoss(pts, S.w1, S.w2, S.b);
    S.best = Math.min(S.best, L);
    text('lr-loss2', fmt(L, 3));
    text('lr-best2', fmt(S.best, 3));
    text('lr-acc2', pct(accuracy(pts, S.w1, S.w2, S.b)));
    p2.update();
    pSig.update();
    if (S.hover >= 0) explain(pts[S.hover]);
  }
  const f2 = (v) => fmt(v);
  const sw1 = ML.range('lr-w1', (v) => { S.w1 = v; handChanged(); }, f2);
  const sw2 = ML.range('lr-w2', (v) => { S.w2 = v; handChanged(); }, f2);
  const sb = ML.range('lr-b', (v) => { S.b = v; handChanged(); }, f2);

  // ---------- step 3: gradient descent ----------

  const GD = { w1: 0, w2: 0, b: 0, iter: 0, hist: [], path: [], running: false, from: -1, status: '' };
  const LW = [-4, 4];
  const rate = ML.range('lr-rate', () => {}, (v) => (v < 0.1 ? v.toFixed(3) : v < 1 ? v.toFixed(2) : v.toFixed(1)),
    (v) => Math.pow(10, v), (v) => Math.log10(v));

  function gdReset(w1, w2, b) {
    Object.assign(GD, { w1, w2, b, iter: 0, status: '' });
    GD.hist = [logLoss(train(), w1, w2, b)];
    GD.path = [[w1, w2]];
    gdRefresh();
  }
  function gdStep() {
    const pts = train();
    const [g1, g2, g0] = gradient(pts, GD.w1, GD.w2, GD.b);
    const lr = rate.value;
    GD.last = { g1, g2, g0, lr, from: [GD.w1, GD.w2, GD.b] };
    GD.w1 = clamp(GD.w1 - lr * g1, -30, 30);
    GD.w2 = clamp(GD.w2 - lr * g2, -30, 30);
    GD.b = clamp(GD.b - lr * g0, -30, 30);
    GD.iter++;
    const L = logLoss(pts, GD.w1, GD.w2, GD.b);
    GD.hist.push(L);
    GD.path.push([GD.w1, GD.w2]);
    if (GD.path.length > 2000) GD.path.shift();
    return Math.hypot(g1, g2, g0);
  }
  function gdRefresh() {
    text('lr-iter', GD.iter);
    text('lr-loss3', fmt(GD.hist[GD.hist.length - 1], 3));
    const g = GD.last;
    const el = id('lr-grad');
    if (!g) el.textContent = 'Each step: new weight = old weight − learning rate × gradient.\nPress "One step" to see the numbers.';
    else {
      el.textContent =
        `gradient: glucose ${ML.signed(g.g1, 3)}   BMI ${ML.signed(g.g2, 3)}   baseline ${ML.signed(g.g0, 3)}\n` +
        `step ${GD.iter}: glucose weight ${fmt(g.from[0])} → ${fmt(GD.w1)}, BMI weight ${fmt(g.from[1])} → ${fmt(GD.w2)}, baseline ${fmt(g.from[2])} → ${fmt(GD.b)}` +
        (GD.status ? `\n${GD.status}` : '');
    }
    id('lr-train').textContent = GD.running ? 'Pause' : 'Train';
    p3.update(); pLand.update(); pCurve.update();
  }
  function gdLoop() {
    if (!GD.running) return;
    let gn = 0;
    const per = GD.iter < 40 ? 1 : 3;
    for (let k = 0; k < per; k++) gn = gdStep();
    if (gn < 2e-4) { GD.running = false; GD.status = 'Converged: the gradient is essentially zero.'; }
    else if (GD.iter >= 3000) { GD.running = false; GD.status = 'Stopped after 3,000 steps.'; }
    gdRefresh();
    if (GD.running) requestAnimationFrame(gdLoop);
  }
  function gdStop() { GD.running = false; id('lr-train').textContent = 'Train'; }
  ML.btn('lr-train', () => {
    GD.running = !GD.running;
    GD.status = '';
    gdRefresh();
    if (GD.running) requestAnimationFrame(gdLoop);
  });
  ML.btn('lr-step', () => { gdStop(); gdStep(); gdRefresh(); });
  ML.btn('lr-reset-gd', () => { gdStop(); gdReset(S.w1, S.w2, S.b); });

  const p3 = scatterPlot('lr-gd', { aspect: 0.8, maxHeight: 360, margin: { l: 44, r: 10 } });
  p3.draw = (c, p) => {
    p.axes();
    p.clip();
    drawShading(p, GD.w1, GD.w2, GD.b);
    drawBoundary(p, GD.w1, GD.w2, GD.b);
    drawPatients(p, train(), -1);
    c.restore();
  };

  let landCache = null;
  function landscape(b) {
    const key = S.dataVersion + ':' + b.toFixed(2);
    if (landCache && landCache.key === key) return landCache;
    const N = 44, grid = new Float64Array(N * N), pts = train();
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
      const w1 = LW[0] + ((i + 0.5) / N) * (LW[1] - LW[0]);
      const w2 = LW[0] + ((j + 0.5) / N) * (LW[1] - LW[0]);
      const L = logLoss(pts, w1, w2, b);
      grid[j * N + i] = L;
      lo = Math.min(lo, L); hi = Math.max(hi, L);
    }
    landCache = { key, N, grid, lo, hi };
    return landCache;
  }
  const pLand = new ML.Plot('lr-land', {
    aspect: 0.8, maxHeight: 360, x: LW, y: LW, xLabel: 'weight on glucose', yLabel: 'weight on BMI',
    margin: { l: 44, r: 10 }, label: 'Heat map of log loss across glucose and BMI weights, with the path gradient descent took',
  });
  pLand.draw = (c, p) => {
    const L = landscape(Math.round(GD.b * 50) / 50);
    const cap = Math.min(L.hi, L.lo + 1.6);
    const cw = p.iw / L.N, ch = p.ih / L.N;
    for (let i = 0; i < L.N; i++) for (let j = 0; j < L.N; j++) {
      const t = 1 - clamp((L.grid[j * L.N + i] - L.lo) / (cap - L.lo), 0, 1);
      c.fillStyle = ML.mix(T['seq-lo'], T['seq-hi'], Math.pow(t, 1.6));
      c.fillRect(p.m.l + i * cw, p.m.t + p.ih - (j + 1) * ch, cw + 0.6, ch + 0.6);
    }
    p.axes({ yGrid: false });
    p.clip();
    if (GD.path.length > 1) p.line(GD.path.map((q) => q[0]), GD.path.map((q) => q[1]), T.s2, 2);
    const s = GD.path[0];
    ML.marker(c, 'circle', p.sx(s[0]), p.sy(s[1]), 4, T.surface, T.s2);
    ML.marker(c, 'circle', p.sx(GD.w1), p.sy(GD.w2), 5.5, T.s2, T.surface);
    c.restore();
  };
  pLand.on('down', (e) => {
    if (!e.inside) return;
    gdStop();
    gdReset(clamp(e.x, -4, 4), clamp(e.y, -4, 4), GD.b);
  });
  pLand.on('move', (e) => {
    if (!e.inside) { pLand.hideTip(); return; }
    pLand.showTip(e.px, e.py, `<span class="k">glucose</span> ${fmt(e.x)} · <span class="k">BMI</span> ${fmt(e.y)}<br>log loss ${fmt(logLoss(train(), e.x, e.y, GD.b), 3)}<br><span class="k">click to start here</span>`);
  });

  const pCurve = new ML.Plot('lr-curve', {
    height: 170, x: [0, 50], y: [0, 1], xLabel: 'training step', margin: { l: 44, b: 36, t: 10, r: 12 },
    label: 'Line chart of log loss over training steps',
  });
  pCurve.draw = (c, p) => {
    const n = GD.hist.length;
    p.o.x = [0, Math.max(50, n - 1)];
    p.o.y = [0, Math.max(1, Math.min(4, Math.max(...GD.hist) * 1.05))];
    p.axes();
    p.clip();
    const best = model().loss;
    c.restore();
    p.refLine(0, best, p.o.x[1], best, `best possible ≈ ${fmt(best, 3)}`, 'right');
    p.clip();
    p.line(GD.hist.map((_, i) => i), GD.hist, T.s1, 2);
    ML.marker(c, 'circle', p.sx(n - 1), p.sy(GD.hist[n - 1]), 4, T.s1, T.surface);
    c.restore();
  };
  pCurve.on('move', (e) => {
    const i = Math.round(clamp(e.x, 0, GD.hist.length - 1));
    if (!GD.hist.length) return;
    pCurve.showTip(e.px, e.py, `step ${i}<br>log loss ${fmt(GD.hist[i], 3)}`);
  });

  // ---------- step 4: threshold ----------

  let testScores = [], testLabels = [], roc = [], auc = NaN;
  function scoreTest() {
    const m = model();
    testScores = test.map((p) => prob(p, m.w1, m.w2, m.b));
    testLabels = test.map((p) => p.y);
    roc = ML.roc(testScores, testLabels);
    auc = ML.auc(testScores, testLabels);
    text('lr-auc', fmt(auc, 2));
  }

  const thr = ML.range('lr-thr', (v) => { S.thr = v; thrChanged(); }, (v) => fmt(v));
  function setThr(v) {
    S.thr = clamp(v, 0.005, 0.995);
    thr.input.value = S.thr.toFixed(2);
    ML.$('output[for="lr-thr"]').textContent = fmt(S.thr);
    thrChanged();
  }

  const pHist = new ML.Plot('lr-hist', {
    height: 240, x: [0, 1], y: [-1, 1], xLabel: 'predicted 5-year risk', drag: true,
    xFmt: (v) => Math.round(v * 100) + '%', margin: { l: 34, t: 22 },
    label: 'Dot plot of predicted risk for held-out patients, split by outcome, with a draggable threshold',
  });
  let histDots = [];
  pHist.draw = (c, p) => {
    // Stack one dot per patient in narrow risk bins; cases above the axis, non-cases below.
    const nb = Math.max(20, Math.floor(p.iw / 9));
    const stacks = [new Array(nb).fill(0), new Array(nb).fill(0)];
    const order = testScores.map((s, i) => i).sort((a, b) => testScores[a] - testScores[b]);
    let maxStack = 1;
    const place = [];
    for (const i of order) {
      const bin = Math.min(nb - 1, Math.floor(testScores[i] * nb));
      const k = stacks[testLabels[i]][bin]++;
      maxStack = Math.max(maxStack, k + 1);
      place.push({ i, bin, k });
    }
    const binPx = p.iw / nb;
    const r = Math.max(1.8, Math.min(4.2, binPx / 2 - 0.4, (p.ih / 2 - 4) / (maxStack * 2)));
    const mid = p.m.t + p.ih / 2;
    const T0 = ML.theme;
    p.axes({ yTicks: [], noY: true });
    c.strokeStyle = T0.axis;
    c.beginPath(); c.moveTo(p.m.l, mid + 0.5); c.lineTo(p.m.l + p.iw, mid + 0.5); c.stroke();
    c.fillStyle = T0.muted;
    c.font = '11px ' + ML.FONT;
    c.textAlign = 'right';
    c.textBaseline = 'top';
    c.fillText('developed diabetes ↑', p.m.l + p.iw - 4, p.m.t + 4);
    c.textBaseline = 'bottom';
    c.fillText('didn’t ↓', p.m.l + p.iw - 4, p.m.t + p.ih - 4);
    c.textBaseline = 'alphabetic';
    histDots = [];
    for (const d of place) {
      const y = testLabels[d.i];
      const x = p.m.l + (d.bin + 0.5) * binPx;
      const yy = y ? mid - r - 1 - d.k * 2 * r : mid + r + 1 + d.k * 2 * r;
      const flagged = testScores[d.i] >= S.thr;
      const col = y ? T0.s2 : T0.s1;
      ML.marker(c, 'circle', x, yy, r - 0.3, flagged ? col : ML.rgba(col, 0.28), null);
      histDots.push({ x, y: yy, i: d.i });
    }
    const tx = p.sx(S.thr);
    c.fillStyle = ML.rgba(T0.ink, 0.04);
    c.fillRect(tx, p.m.t, p.m.l + p.iw - tx, p.ih);
    c.strokeStyle = T0.ink;
    c.lineWidth = 2;
    c.beginPath(); c.moveTo(tx, p.m.t - 4); c.lineTo(tx, p.m.t + p.ih); c.stroke();
    ML.marker(c, 'circle', tx, p.m.t - 6, 6, T0.ink, T0.surface);
    c.fillStyle = T0.ink;
    c.font = '600 11px ' + ML.FONT;
    c.textAlign = tx > p.m.l + p.iw - 90 ? 'right' : 'left';
    c.fillText(`flag ≥ ${Math.round(S.thr * 100)}%`, tx + (c.textAlign === 'right' ? -10 : 10), p.m.t - 12);
  };
  pHist.on('down', (e) => { if (e.px > pHist.m.l - 10) setThr(e.x); });
  pHist.on('move', (e) => {
    if (e.dragging) { setThr(e.x); pHist.hideTip(); return; }
    let best = -1, bd = 64;
    histDots.forEach((d, k) => { const dd = (d.x - e.px) ** 2 + (d.y - e.py) ** 2; if (dd < bd) { bd = dd; best = k; } });
    if (best < 0) { pHist.hideTip(); return; }
    const i = histDots[best].i, q = test[i];
    pHist.showTip(e.px, e.py, `<b>Held-out patient</b><br><span class="k">Predicted risk</span> ${pct(testScores[i])}<br>` +
      `<span class="k">Glucose</span> ${q.g} · <span class="k">BMI</span> ${q.bmi.toFixed(1)}<br>${q.y ? 'Developed diabetes' : 'No diabetes'} · ${testScores[i] >= S.thr ? 'flagged' : 'not flagged'}`);
  });

  const pRoc = new ML.Plot('lr-roc', {
    aspect: 1, maxHeight: 300, x: [0, 1], y: [0, 1], xLabel: 'false-positive rate (1 − specificity)', yLabel: 'sensitivity',
    xTicks: 4, yTicks: 4, xFmt: (v) => Math.round(v * 100) + '%', yFmt: (v) => Math.round(v * 100) + '%', drag: true,
    margin: { l: 50 }, label: 'ROC curve with the current threshold marked',
  });
  pRoc.draw = (c, p) => {
    p.axes();
    p.refLine(0, 0, 1, 1);
    p.clip();
    c.fillStyle = ML.rgba(T.s1, 0.1);
    c.beginPath();
    c.moveTo(p.sx(0), p.sy(0));
    for (const q of roc) c.lineTo(p.sx(q.fpr), p.sy(q.tpr));
    c.lineTo(p.sx(1), p.sy(0));
    c.closePath();
    c.fill();
    p.line(roc.map((q) => q.fpr), roc.map((q) => q.tpr), T.s1, 2);
    const m = ML.confusion(testScores, testLabels, S.thr);
    c.restore();
    ML.marker(c, 'circle', p.sx(1 - m.spec), p.sy(m.sens), 6, T.ink, T.surface);
    c.fillStyle = T.muted;
    c.font = '11px ' + ML.FONT;
    c.textAlign = 'left';
    c.fillText('chance', p.sx(0.66), p.sy(0.56));
  };
  function rocPick(e) {
    let best = null, bd = Infinity;
    for (const q of roc) {
      if (q.t === undefined) continue;
      const d = (p0(q.fpr) - e.px) ** 2 + (p1y(q.tpr) - e.py) ** 2;
      if (d < bd) { bd = d; best = q; }
    }
    if (best) setThr(best.t);
  }
  const p0 = (v) => pRoc.sx(v), p1y = (v) => pRoc.sy(v);
  pRoc.on('down', rocPick);
  pRoc.on('move', (e) => { if (e.dragging) rocPick(e); });

  function thrChanged() {
    const m = ML.confusion(testScores, testLabels, S.thr);
    text('lr-sens', pct(m.sens));
    text('lr-spec', pct(m.spec));
    text('lr-ppv', pct(m.ppv));
    text('lr-npv', pct(m.npv));
    id('lr-cm').innerHTML = `
      <div class="cm" style="grid-template-columns: auto 1fr 1fr">
        <div class="h"></div><div class="h">Flagged</div><div class="h">Not flagged</div>
        <div class="h rowh">Developed<br>diabetes</div>
        <div class="c"><b>${m.tp}</b><small>caught</small></div>
        <div class="c"><b>${m.fn}</b><small>missed</small></div>
        <div class="h rowh">Didn't</div>
        <div class="c"><b>${m.fp}</b><small>unneeded referral</small></div>
        <div class="c"><b>${m.tn}</b><small>correctly left alone</small></div>
      </div>`;
    pHist.update();
    pRoc.update();
    if (active === 4) updateSites();
  }

  ML.btn('lr-p-screen', () => {
    let t = 0.01;
    for (let v = 0.99; v >= 0.01; v -= 0.01) if (ML.confusion(testScores, testLabels, v).sens >= 0.9) { t = v; break; }
    setThr(t);
  });
  ML.btn('lr-p-spec', () => {
    let t = 0.99;
    for (let v = 0.01; v <= 0.99; v += 0.01) if (ML.confusion(testScores, testLabels, v).spec >= 0.9) { t = v; break; }
    setThr(t);
  });
  ML.btn('lr-p-bal', () => {
    let t = 0.5, bj = -1;
    for (let v = 0.01; v <= 0.99; v += 0.01) {
      const m = ML.confusion(testScores, testLabels, v);
      if (m.sens + m.spec > bj) { bj = m.sens + m.spec; t = v; }
    }
    setThr(t);
  });
  const cost = ML.range('lr-cost', costNote, (v) => (v === 1 ? '1 referral' : v + ' referrals'));
  function costNote() {
    const c = cost.value;
    text('lr-cost-note', `Flag anyone above 1 ÷ (1 + ${c}) = ${pct(1 / (1 + c), 1)} risk.`);
  }
  ML.btn('lr-usecost', () => setThr(1 / (1 + cost.value)));
  costNote();

  // ---------- step 5: a new hospital ----------

  const siteDefault = (() => {
    let s = 0;
    for (const L of siteLatent) s += realize(L, TRUE_B0, 0).risk;
    return Math.round((s / siteLatent.length) * 100) / 100;
  })();
  const prevNew = ML.range('lr-prevnew', () => { S.recal = null; updateSites(); }, (v) => pct(v));
  const shift = ML.range('lr-shift', () => { S.recal = null; updateSites(); }, (v) => (v > 0 ? '+' : v < 0 ? '−' : '±') + Math.abs(v) + ' mg/dL');
  prevNew.set(siteDefault);

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
  function site(prev, bias) {
    const b0 = Math.abs(prev - siteDefault) < 0.005 ? TRUE_B0 : interceptFor(prev);
    return siteLatent.map((L) => realize(L, b0, bias));
  }
  function calCurve(scores, labels) {
    const bins = Array.from({ length: 10 }, () => ({ n: 0, s: 0, y: 0 }));
    scores.forEach((s, i) => { const b = bins[Math.min(9, Math.floor(s * 10))]; b.n++; b.s += s; b.y += labels[i]; });
    return bins.filter((b) => b.n >= 10).map((b) => ({ x: b.s / b.n, y: b.y / b.n, n: b.n }));
  }

  let calLines = [];
  const pCal = new ML.Plot('lr-cal', {
    aspect: 0.9, maxHeight: 360, x: [0, 1], y: [0, 1], xLabel: 'predicted risk', yLabel: 'observed rate',
    xTicks: 5, yTicks: 5, xFmt: (v) => Math.round(v * 100) + '%', yFmt: (v) => Math.round(v * 100) + '%',
    margin: { l: 50 }, label: 'Calibration plot comparing predicted and observed diabetes rates',
  });
  pCal.draw = (c, p) => {
    p.axes();
    p.refLine(0, 0, 1, 1, 'perfect', 'right');
    p.clip();
    for (const L of calLines) {
      p.line(L.pts.map((q) => q.x), L.pts.map((q) => q.y), L.color, 2);
      for (const q of L.pts) ML.marker(c, 'circle', p.sx(q.x), p.sy(q.y), 4, L.color, T.surface);
    }
    c.restore();
  };
  pCal.on('move', (e) => {
    let hit = null, bd = 144;
    for (const L of calLines) for (const q of L.pts) {
      const d = (pCal.sx(q.x) - e.px) ** 2 + (pCal.sy(q.y) - e.py) ** 2;
      if (d < bd) { bd = d; hit = { L, q }; }
    }
    if (!hit) { pCal.hideTip(); return; }
    pCal.showTip(e.px, e.py, `<b>${hit.L.name}</b><br>${hit.q.n} patients predicted ${pct(hit.q.x)} on average<br>${pct(hit.q.y)} developed diabetes`);
  });

  ML.btn('lr-recal', () => {
    if (S.recal !== null) { S.recal = null; updateSites(); return; }
    const m = model();
    const pts = site(prevNew.value, shift.value).slice(0, 200);
    const zs = pts.map((p) => score(p, m.w1, m.w2, m.b));
    let d = 0;
    for (let k = 0; k < 25; k++) {
      let g = 0, h = 0;
      zs.forEach((z, i) => { const q = ML.sigmoid(z + d); g += q - pts[i].y; h += q * (1 - q); });
      d -= g / Math.max(h, 1e-9);
    }
    S.recal = d;
    updateSites();
  });
  ML.btn('lr-sim-reset', () => { prevNew.set(siteDefault); shift.set(0); S.recal = null; updateSites(); });

  function updateSites() {
    const m = model();
    const orig = site(siteDefault, 0).slice(200);
    const neu = site(prevNew.value, shift.value).slice(200);
    const d = S.recal || 0;
    const so = orig.map((p) => prob(p, m.w1, m.w2, m.b));
    const sn = neu.map((p) => ML.sigmoid(score(p, m.w1, m.w2, m.b) + d));
    const yo = orig.map((p) => p.y), yn = neu.map((p) => p.y);
    const mo = ML.confusion(so, yo, S.thr), mn = ML.confusion(sn, yn, S.thr);
    const newName = S.recal !== null ? 'New site, recalibrated' : 'New site';
    calLines = [
      { name: 'Original hospital', color: T.s1, pts: calCurve(so, yo) },
      { name: newName, color: S.recal !== null ? T.s3 : T.s2, pts: calCurve(sn, yn) },
    ];
    id('lr-cal-legend').innerHTML = calLines.map((L) => `<span><i class="sw line" style="background:${L.color}"></i>${L.name}</span>`).join('');
    const rows = [
      ['Developed diabetes', pct(ML.mean(yo)), pct(ML.mean(yn))],
      ['Average predicted risk', pct(ML.mean(so)), pct(ML.mean(sn))],
      ['AUC', fmt(ML.auc(so, yo)), fmt(ML.auc(sn, yn))],
      ['Share flagged', pct(ML.mean(so.map((s) => +(s >= S.thr)))), pct(ML.mean(sn.map((s) => +(s >= S.thr))))],
      ['Sensitivity', pct(mo.sens), pct(mn.sens)],
      ['PPV', pct(mo.ppv), pct(mn.ppv)],
    ];
    id('lr-sites').innerHTML = rows.map((r) => `<tr><td>${r[0]}</td><td>${r[1]}</td><td>${r[2]}</td></tr>`).join('');
    ML.$('#lr-sites').closest('table').querySelector('th:last-child').textContent = S.recal !== null ? 'Recalibrated' : 'New site';
    text('lr-cal-thr', pct(S.thr));
    id('lr-recal').textContent = S.recal !== null ? 'Undo recalibration' : 'Recalibrate on 200 local patients';
    const gap = ML.mean(sn) - ML.mean(yn);
    let note;
    if (Math.abs(gap) < 0.025) note = S.recal !== null
      ? 'Recalibrated: predicted and observed rates line up again, and the ranking (AUC) never changed.'
      : 'Predicted and observed rates match here, so the model is well calibrated at this site.';
    else note = `At the new site the model ${gap > 0 ? 'overestimates' : 'underestimates'} risk by ${Math.round(Math.abs(gap) * 100)} percentage points on average, while its AUC barely changes. It still ranks patients well; its numbers are just wrong.`;
    text('lr-cal-note', note);
    pCal.update();
  }

  // ---------- routing ----------

  ML.register('risk', {
    enter(step) {
      active = step;
      if (step === 0) updateStats1();
      if (step === 1) handChanged();
      if (step === 2) {
        if (GD.from !== S.handVersion + ':' + S.dataVersion) { GD.from = S.handVersion + ':' + S.dataVersion; gdReset(S.w1, S.w2, S.b); }
        else gdRefresh();
      }
      if (step === 3) { scoreTest(); thrChanged(); }
      if (step === 4) { scoreTest(); updateSites(); }
    },
    leave(step) { if (step === 2) gdStop(); },
  });
  document.addEventListener('ml-theme', () => { if (active === 4) updateSites(); });

  updateStats1();
  void sw1; void sw2; void sb;
})();
