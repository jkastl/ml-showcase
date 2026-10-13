/* Chapter 3 page: k-means clustering of simulated patients at diabetes diagnosis. The algorithm is in js/engine/kmeans.js. */
(function () {
  'use strict';
  const { id, text, fmt, clamp } = ML;
  const T = ML.theme;
  const KM = ML.kmeans;
  const { GROUPS, pts, STD, RAW } = KM;
  const XD = [15, 90], YD = [5, 14];
  const MAXK = 6; // one color and marker shape per cluster

  // ---------- drawing ----------

  function scatter(host, extra = {}) {
    return new ML.Plot(host, Object.assign({
      aspect: 0.62, maxHeight: 440, minHeight: 260, x: XD, y: YD,
      xLabel: 'Age at diagnosis (years)', yLabel: 'HbA1c (%)',
      label: 'Scatter plot of patients by age at diagnosis and HbA1c',
    }, extra));
  }
  function drawTerritories(p, cents, sc) {
    if (!cents.length) return;
    const c = p.ctx, cell = 5;
    for (let px = p.m.l; px < p.m.l + p.iw; px += cell) {
      for (let py = p.m.t; py < p.m.t + p.ih; py += cell) {
        const k = KM.nearest({ x: p.ix(px + cell / 2), y: p.iy(py + cell / 2) }, cents, sc);
        c.fillStyle = ML.rgba(ML.series(k), 0.13);
        c.fillRect(px, py, cell, cell);
      }
    }
  }
  function drawPts(p, lab) {
    const c = p.ctx;
    pts.forEach((q, i) => {
      const k = lab ? lab[i] : -1;
      ML.marker(c, k < 0 ? 'circle' : ML.SHAPES[k], p.sx(q.x), p.sy(q.y), 3.6, k < 0 ? T.muted : ML.series(k), T.surface);
    });
  }
  function drawCents(p, cents, hot) {
    const c = p.ctx;
    cents.forEach((q, k) => {
      const X = p.sx(q.x), Y = p.sy(q.y);
      ML.marker(c, ML.SHAPES[k], X, Y, k === hot ? 11 : 9, ML.series(k), T.ink);
      ML.marker(c, 'circle', X, Y, 2.2, T.ink, null);
    });
  }
  function drawTrails(p, trails) {
    for (const tr of trails) {
      if (tr.length > 1) p.line(tr.map((q) => q.x), tr.map((q) => q.y), ML.rgba(T.ink, 0.55), 1.5);
    }
  }
  function ptTip(plot, e, lab, extra) {
    const i = plot.nearest(pts, e, (q) => q.x, (q) => q.y, 10);
    if (i < 0) { plot.hideTip(); return; }
    const q = pts[i];
    plot.showTip(e.px, e.py, `<span class="k">Age at diagnosis</span> ${q.x}<br><span class="k">HbA1c</span> ${q.y.toFixed(1)}%` +
      (lab ? `<br>In cluster ${lab[i] + 1}` : '') + (extra ? extra(i) : ''));
  }

  // ---------- steps 1 & 2: place and iterate ----------

  const K = { placed: [], cents: [], lab: null, phase: 'assign', trails: [], hist: [], running: 0, drag: -1, converged: false };
  const liveLab = () => (K.placed.length ? KM.assign(pts, K.placed, STD) : null);

  const pPlace = scatter('km-place', { drag: true });
  pPlace.draw = (c, p) => {
    p.axes();
    p.clip();
    drawTerritories(p, K.placed, STD);
    drawPts(p, liveLab());
    c.restore();
    drawCents(p, K.placed, K.drag);
    if (!K.placed.length) {
      c.fillStyle = T['ink-2'];
      c.font = '600 13px ' + ML.FONT;
      c.textAlign = 'center';
      c.fillText('Tap to drop a center', p.m.l + p.iw / 2, p.m.t + 22);
    }
  };
  function hitCent(plot, cents, e) {
    for (let k = cents.length - 1; k >= 0; k--) {
      if (Math.hypot(plot.sx(cents[k].x) - e.px, plot.sy(cents[k].y) - e.py) < (plot.touch ? 24 : 16)) return k;
    }
    return -1;
  }
  pPlace.on('down', (e) => {
    if (!e.inside) return;
    const k = hitCent(pPlace, K.placed, e);
    if (k >= 0) K.drag = k;
    else if (K.placed.length < MAXK) { K.placed.push({ x: e.x, y: e.y }); K.drag = K.placed.length - 1; }
    placedChanged();
  });
  pPlace.on('move', (e) => {
    if (K.drag >= 0 && e.dragging) {
      K.placed[K.drag] = { x: clamp(e.x, XD[0], XD[1]), y: clamp(e.y, YD[0], YD[1]) };
      pPlace.hideTip();
      placedChanged();
      return;
    }
    pPlace.canvas.style.cursor = hitCent(pPlace, K.placed, e) >= 0 ? 'grab' : K.placed.length < MAXK ? 'crosshair' : 'default';
    ptTip(pPlace, e, liveLab());
  });
  pPlace.on('up', () => { K.drag = -1; pPlace.update(); });
  ML.btn('km-rand', () => { K.placed = KM.kpp(pts, 4, STD, ML.rng(Date.now() & 0xffff)).map((q) => ({ x: q.x, y: q.y })); placedChanged(); });
  ML.btn('km-undo', () => { K.placed.pop(); placedChanged(); });
  ML.btn('km-clear', () => { K.placed = []; placedChanged(); });

  function placedChanged() {
    K.stale = true;
    placedRender();
  }
  function placedRender() {
    const lab = liveLab();
    text('km-k', K.placed.length);
    text('km-inertia', lab ? fmt(KM.inertia(pts, lab, K.placed, STD), 1) : '–');
    id('km-undo').disabled = !K.placed.length;
    id('km-clear').disabled = !K.placed.length;
    pPlace.update();
  }

  function iterStart(cents, note) {
    stopRun();
    K.cents = cents.map((q) => ({ x: q.x, y: q.y }));
    K.lab = null;
    K.phase = 'assign';
    K.trails = K.cents.map((q) => [q]);
    K.hist = [];
    K.converged = false;
    K.note = note || '';
    iterChanged();
  }
  function doAssign() {
    const nl = KM.assign(pts, K.cents, STD);
    const same = K.lab && nl.every((v, i) => v === K.lab[i]);
    K.lab = nl;
    K.hist.push(KM.inertia(pts, K.lab, K.cents, STD));
    if (same) K.converged = true;
    K.phase = 'move';
  }
  function doMove() {
    K.cents = KM.centroids(pts, K.lab, K.cents);
    K.cents.forEach((q, k) => K.trails[k].push(q));
    K.hist.push(KM.inertia(pts, K.lab, K.cents, STD));
    K.phase = 'assign';
  }
  function stopRun() { clearInterval(K.running); K.running = 0; id('km-run').textContent = 'Run'; }
  ML.btn('km-assign', () => { stopRun(); doAssign(); iterChanged(); });
  ML.btn('km-move', () => { stopRun(); doMove(); iterChanged(); });
  ML.btn('km-run', () => {
    if (K.running) { stopRun(); iterChanged(); return; }
    if (K.converged) iterStart(K.trails.map((t) => t[0]), K.note);
    id('km-run').textContent = 'Pause';
    K.running = setInterval(() => {
      if (K.phase === 'assign') doAssign(); else doMove();
      if (K.converged) stopRun();
      iterChanged();
    }, 420);
  });
  ML.btn('km-restart', () => iterStart(K.placed.length >= 2 ? K.placed : KM.kpp(pts, 4, STD, ML.rng(3)), ''));
  ML.btn('km-bad', () => iterStart(KM.BAD_START, 'bad'));
  let ppSeed = 7;
  ML.btn('km-pp', () => iterStart(KM.kpp(pts, Math.max(2, K.cents.length || 4), STD, ML.rng(ppSeed++)), 'pp'));

  const best4 = KM.bestOf(pts, 4, STD, 10, 99);
  const pIter = scatter('km-iter');
  pIter.draw = (c, p) => {
    p.axes();
    p.clip();
    drawTerritories(p, K.cents, STD);
    drawPts(p, K.lab);
    drawTrails(p, K.trails);
    c.restore();
    drawCents(p, K.cents, -1);
  };
  pIter.on('move', (e) => ptTip(pIter, e, K.lab));

  const pInertia = new ML.Plot('km-inertia-chart', {
    height: 170, x: [0, 10], y: [0, 1], xLabel: 'step', margin: { l: 48, b: 36, t: 10, r: 12 },
    label: 'Inertia after each assign or move step',
  });
  pInertia.draw = (c, p) => {
    const h = K.hist;
    p.o.x = [0, Math.max(10, h.length)];
    p.o.y = [0, Math.max(100, ...h) * 1.1];
    p.axes();
    if (K.cents.length === 4) p.refLine(0, best4.inertia, p.o.x[1], best4.inertia, `best known for k = 4: ${fmt(best4.inertia, 0)}`, 'right');
    if (!h.length) return;
    p.line(h.map((_, i) => i + 1), h, T.s1, 2);
    h.forEach((v, i) => ML.marker(c, i % 2 ? 'square' : 'circle', p.sx(i + 1), p.sy(v), 3.5, T.s1, T.surface));
  };
  pInertia.on('move', (e) => {
    const i = Math.round(e.x) - 1;
    if (i < 0 || i >= K.hist.length) { pInertia.hideTip(); return; }
    pInertia.showTip(e.px, e.py, `step ${i + 1} (${i % 2 ? 'move' : 'assign'})<br>inertia ${fmt(K.hist[i], 1)}`);
  });

  function iterChanged() {
    id('km-assign').disabled = K.phase !== 'assign' || K.converged || !!K.running;
    id('km-move').disabled = K.phase !== 'move' || K.converged || !!K.running;
    const rounds = Math.floor(K.hist.length / 2);
    let msg;
    if (K.converged) {
      const I = K.hist[K.hist.length - 1];
      msg = `Converged after ${rounds} rounds: nobody switched clusters. Inertia ${fmt(I, 1)}.`;
      if (K.cents.length === 4) msg += I > best4.inertia * 1.05 ? ' Stuck: the best k = 4 answer is lower.' : ' That matches the best k = 4 answer.';
    } else if (!K.lab && K.note === 'auto') msg = 'Step 1 had fewer than 2 centers, so we started with 4. Press Assign.';
    else if (!K.lab) msg = K.note === 'bad' ? 'All four centers start bunched among the younger patients. Press Assign.' : 'Press Assign to give every patient to their nearest center.';
    else msg = K.phase === 'move' ? 'Now move each center to the average of its patients.' : 'Centers moved. Assign again: some patients may switch.';
    text('km-status', msg);
    pIter.update();
    pInertia.update();
  }

  // ---------- step 3: choosing k ----------

  let sweep = null;
  const S3 = { k: 4 };
  function ensureSweep() {
    if (sweep) return;
    sweep = [];
    for (let k = 1; k <= MAXK; k++) {
      const res = KM.bestOf(pts, k, STD, 10, 100 + k);
      sweep.push({ k, res, inertia: res.inertia, sil: KM.silhouette(pts, res.lab, k, STD) });
    }
  }
  const kPlot = (host, key, yLabel, yd) => {
    const p = new ML.Plot(host, {
      aspect: 0.75, maxHeight: 260, x: [0.5, MAXK + 0.5], y: yd, xLabel: 'k (number of clusters)', yLabel,
      margin: { l: 50, r: 12 }, label: `${yLabel} by number of clusters`,
    });
    p.draw = (c, pl) => {
      if (!sweep) return;
      const rows = sweep.filter((r) => Number.isFinite(r[key]));
      if (key === 'inertia') pl.o.y = [0, Math.ceil(sweep[0].inertia / 100) * 100];
      pl.axes({ xTicks: sweep.map((r) => r.k) });
      c.fillStyle = ML.rgba(T.ink, 0.06);
      c.fillRect(pl.sx(S3.k) - 12, pl.m.t, 24, pl.ih);
      pl.line(rows.map((r) => r.k), rows.map((r) => r[key]), T.s1, 2);
      for (const r of rows) ML.marker(c, 'circle', pl.sx(r.k), pl.sy(r[key]), r.k === S3.k ? 6 : 4, r.k === S3.k ? T.ink : T.s1, T.surface);
    };
    p.on('down', (e) => { const k = clamp(Math.round(e.x), key === 'sil' ? 2 : 1, MAXK); S3.k = k; kChanged(); });
    p.on('move', (e) => {
      const k = Math.round(e.x);
      const r = sweep && sweep[k - 1];
      if (!r || !e.inside || !Number.isFinite(r[key])) { p.hideTip(); return; }
      p.showTip(e.px, e.py, `<b>k = ${k}</b><br><span class="k">inertia</span> ${fmt(r.inertia, 0)}<br><span class="k">silhouette</span> ${fmt(r.sil)}<br><span class="k">tap to view</span>`);
    });
    return p;
  };
  const pElbow = kPlot('km-elbow', 'inertia', 'inertia', [0, 700]);
  const pSil = kPlot('km-sil', 'sil', 'silhouette', [0, 0.7]);
  const pK = scatter('km-kview');
  pK.draw = (c, p) => {
    if (!sweep) return;
    const res = sweep[S3.k - 1].res;
    p.axes();
    p.clip();
    drawTerritories(p, res.cents, STD);
    drawPts(p, res.lab);
    c.restore();
    drawCents(p, res.cents, -1);
  };
  pK.on('move', (e) => ptTip(pK, e, sweep && sweep[S3.k - 1].res.lab));

  function describe(age, a1c) {
    const a = age < 42 ? 'Younger onset' : age < 60 ? 'Middle-aged onset' : 'Older onset';
    const h = a1c >= 10 ? 'very high HbA1c' : a1c >= 8 ? 'high HbA1c' : 'near-target HbA1c';
    let hint = '';
    if (a1c >= 10 && age < 45) hint = 'Echoes the severe, insulin-deficient subgroup.';
    else if (a1c < 8 && age >= 60) hint = 'Echoes the mild, age-related subgroup.';
    return { label: `${a}, ${h}`, hint };
  }
  function shapeSvg(k) {
    const pth = {
      circle: '<circle cx="7" cy="7" r="5"/>',
      square: '<rect x="2.5" y="2.5" width="9" height="9"/>',
      triangle: '<path d="M7 1.5L12.5 11.5H1.5Z"/>',
      diamond: '<path d="M7 1L13 7L7 13L1 7Z"/>',
      down: '<path d="M7 12.5L12.5 2.5H1.5Z"/>',
      plus: '<path d="M5 1.5h4v3.5h3.5v4H9v3.5H5V9H1.5V5H5Z"/>',
    }[ML.SHAPES[k]];
    return `<svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true" fill="${ML.series(k)}">${pth}</svg>`;
  }
  function kChanged() {
    ensureSweep();
    text('km-ksel', S3.k);
    const res = sweep[S3.k - 1].res;
    id('km-profiles').innerHTML = res.cents.map((cc, k) => {
      const n = res.lab.filter((l) => l === k).length;
      const d = describe(cc.x, cc.y);
      return `<div class="profile"><b>${shapeSvg(k)}Cluster ${k + 1} · ${n} patients</b>` +
        `Age ${Math.round(cc.x)} · HbA1c ${cc.y.toFixed(1)}%<br>${d.label}.${d.hint ? `<br><span style="color:var(--muted)">${d.hint}</span>` : ''}</div>`;
    }).join('');
    pElbow.update(); pSil.update(); pK.update();
  }

  // ---------- step 4: units ----------

  const S4 = { units: 'raw', truth: false, res: null };
  const cache4 = {};
  const unitsSeg = ML.seg('km-units', (v) => { S4.units = v; unitsChanged(); });
  id('km-truth').addEventListener('change', (e) => { S4.truth = e.target.checked; unitsChanged(); });

  function ellipses(p) {
    const c = p.ctx;
    GROUPS.forEach((g) => {
      const ms = pts.filter((q) => GROUPS[q.g] === g);
      const mx = ML.mean(ms.map((q) => q.x)), my = ML.mean(ms.map((q) => q.y));
      const sx = ML.sd(ms.map((q) => q.x)), sy = ML.sd(ms.map((q) => q.y));
      c.beginPath();
      c.ellipse(p.sx(mx), p.sy(my), Math.abs(p.sx(mx + 2 * sx) - p.sx(mx)), Math.abs(p.sy(my + 2 * sy) - p.sy(my)), 0, 0, Math.PI * 2);
      c.strokeStyle = T.ink;
      c.lineWidth = 1.5;
      c.stroke();
      c.fillStyle = T.ink;
      c.font = '600 12px ' + ML.FONT;
      c.textAlign = 'center';
      c.textBaseline = 'bottom';
      c.fillText('Group ' + g.name, p.sx(mx), p.sy(my + 2 * sy) - 3);
      c.textBaseline = 'alphabetic';
    });
  }
  const pScale = scatter('km-scale');
  pScale.draw = (c, p) => {
    if (!S4.res) return;
    const sc = S4.units === 'raw' ? RAW : STD;
    p.axes();
    p.clip();
    drawTerritories(p, S4.res.cents, sc);
    drawPts(p, S4.res.lab);
    if (S4.truth) ellipses(p);
    c.restore();
    drawCents(p, S4.res.cents, -1);
  };
  pScale.on('move', (e) => ptTip(pScale, e, S4.res && S4.res.lab, S4.truth ? (i) => `<br>Simulated as group ${GROUPS[pts[i].g].name}` : null));

  function unitsChanged() {
    const sc = S4.units === 'raw' ? RAW : STD;
    S4.res = cache4[S4.units] || (cache4[S4.units] = KM.bestOf(pts, 4, sc, 10, 400));
    text('km-agree', ML.pct(KM.agreement(pts, S4.res.lab)));
    text('km-unit-age', fmt(1 / sc[0]));
    text('km-unit-a1c', fmt(1 / sc[1]));
    text('km-scale-legend', S4.truth
      ? 'Shading and colored points: the 4 clusters k-means found. Rings: the 4 groups we actually simulated.'
      : 'Shading and colored points: the 4 clusters k-means found.');
    pScale.update();
  }

  // ---------- routing ----------

  ML.register('subgroups', {
    enter(step) {
      if (step === 0) placedRender();
      if (step === 1) {
        if (K.stale || !K.cents.length) {
          K.stale = false;
          iterStart(K.placed.length >= 2 ? K.placed : KM.kpp(pts, 4, STD, ML.rng(3)), K.placed.length >= 2 ? '' : 'auto');
        } else iterChanged();
      }
      if (step === 2) kChanged();
      if (step === 3) { unitsSeg.set(S4.units); unitsChanged(); }
    },
    leave(step) { if (step === 1) stopRun(); },
  });
})();
