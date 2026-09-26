/* Chapter 4: a small neural network that labels single ECG beats. */
(function () {
  'use strict';
  const { id, text, fmt, pct, clamp } = ML;
  const T = ML.theme;
  const N = 64; // samples per beat window
  const WIN_MS = 800;
  const CLS = ['Normal', 'PVC', 'Artifact'];
  const CSHAPE = ['circle', 'square', 'triangle'];
  const YD = [-2, 2.2];

  // ---------- simulated beats ----------

  const gauss = (t, mu, s) => Math.exp(-0.5 * ((t - mu) / s) ** 2);
  function makeBeat(cls, r, o = {}) {
    const w = o.qrs || 1, clean = !!o.clean;
    const j = clean ? 0 : (r() - 0.5) * 0.05;
    const sc = clean ? 1 : 0.85 + 0.3 * r();
    const parts = [], wander = [];
    if (cls === 0) {
      parts.push([0.12 + 0.08 * r(), 0.2 + j, 0.025]); // P
      parts.push([-0.12, 0.4 - 0.016 * w + j, 0.008 * w]); // Q
      parts.push([0.95 + 0.35 * r(), 0.4 + j, 0.011 * w]); // R
      parts.push([-0.28, 0.4 + 0.02 * w + j, 0.009 * w]); // S
      parts.push([0.25 + 0.15 * r(), 0.68 + j, 0.045]); // T
    } else if (cls === 1) {
      const pol = o.pol || (r() < 0.7 ? 1 : -1);
      parts.push([pol * (1.1 + 0.5 * r()), 0.42 + j, 0.03 * w]); // wide QRS
      parts.push([-pol * (0.35 + 0.2 * r()), 0.42 + 0.06 * w + j, 0.025 * w]);
      parts.push([-pol * (0.3 + 0.15 * r()), 0.72 + j, 0.06]); // discordant T
    } else {
      for (let k = 0; k < 3; k++) wander.push([0.15 + 0.35 * r(), 0.5 + 3.5 * r(), r() * 6.283]);
      if (r() < 0.45) parts.push([(r() < 0.5 ? -1 : 1) * (0.6 + 0.6 * r()), 0.15 + 0.7 * r(), 0.006 + 0.006 * r()]);
    }
    const noise = new Float64Array(N);
    if (!clean) {
      const sd = 0.015 + 0.03 * r();
      for (let i = 0; i < N; i++) noise[i] = sd * ML.randn(r);
      if (cls !== 2 && r() < 0.6) wander.push([0.03 + 0.1 * r(), 0.3 + 1.2 * r(), r() * 6.283]);
    }
    if (cls === 2) {
      const nb = 1 + Math.floor(r() * 3);
      for (let b = 0; b < nb; b++) {
        const c = r(), wd = 0.05 + 0.12 * r(), a = 0.25 + 0.45 * r();
        for (let i = 0; i < N; i++) noise[i] += a * gauss((i + 0.5) / N, c, wd) * ML.randn(r);
      }
    }
    return { cls, sc, parts, wander, noise };
  }
  function evalBeat(B, t) {
    let v = 0;
    for (const [a, mu, s] of B.parts) v += a * gauss(t, mu, s);
    v *= B.sc;
    for (const [a, f, ph] of B.wander) v += a * Math.sin(2 * Math.PI * f * t + ph);
    const u = t * N - 0.5;
    const i0 = clamp(Math.floor(u), 0, N - 1), i1 = Math.min(N - 1, i0 + 1), fr = clamp(u - i0, 0, 1);
    return v + B.noise[i0] * (1 - fr) + B.noise[i1] * fr;
  }
  const tAt = (i) => (i + 0.5) / N;
  const sample = (B) => Float64Array.from({ length: N }, (_, i) => evalBeat(B, tAt(i)));

  function dataset(seed, counts) {
    const r = ML.rng(seed), out = [];
    counts.forEach((n, cls) => { for (let i = 0; i < n; i++) { const B = makeBeat(cls, r); out.push({ B, x: sample(B), y: cls }); } });
    return ML.shuffle(out, r);
  }
  const TRAIN = dataset(41, [620, 460, 320]);
  const NEURON_SET = TRAIN.slice(0, 630); // step 2 plots a subset to stay light
  const VAL = dataset(42, [120, 90, 60]);

  // ---------- network: 64 → H (tanh) → 3 (softmax), trained with Adam ----------

  const WD = 0.3;

  class MLP {
    constructor(h, seed) {
      const r = ML.rng(seed);
      this.h = h;
      // Small first-layer init: templates start flat and visibly grow into shape as training runs.
      this.W1 = Float64Array.from({ length: h * N }, () => ML.randn(r) * 0.02);
      this.b1 = new Float64Array(h);
      this.W2 = Float64Array.from({ length: 3 * h }, () => ML.randn(r) * Math.sqrt(1 / h));
      this.b2 = new Float64Array(3);
      this.params = [this.W1, this.b1, this.W2, this.b2];
      this.m = this.params.map((p) => new Float64Array(p.length));
      this.v = this.params.map((p) => new Float64Array(p.length));
      this.t = 0;
    }
    forward(x) {
      const H = this.h, hid = new Float64Array(H), z = new Float64Array(3);
      for (let j = 0; j < H; j++) {
        let s = this.b1[j];
        const o = j * N;
        for (let i = 0; i < N; i++) s += this.W1[o + i] * x[i];
        hid[j] = Math.tanh(s);
      }
      for (let k = 0; k < 3; k++) {
        let s = this.b2[k];
        for (let j = 0; j < H; j++) s += this.W2[k * H + j] * hid[j];
        z[k] = s;
      }
      const mx = Math.max(z[0], z[1], z[2]);
      const e = z.map((v) => Math.exp(v - mx)), se = e[0] + e[1] + e[2];
      return { hid, p: e.map((v) => v / se) };
    }
    step(batch, lr) {
      const H = this.h;
      const g = this.params.map((p) => new Float64Array(p.length));
      for (const ex of batch) {
        const { hid, p } = this.forward(ex.x);
        const dz = [p[0], p[1], p[2]];
        dz[ex.y] -= 1;
        const dh = new Float64Array(H);
        for (let k = 0; k < 3; k++) {
          g[3][k] += dz[k];
          for (let j = 0; j < H; j++) { g[2][k * H + j] += dz[k] * hid[j]; dh[j] += dz[k] * this.W2[k * H + j]; }
        }
        for (let j = 0; j < H; j++) {
          const d = dh[j] * (1 - hid[j] * hid[j]);
          g[1][j] += d;
          const o = j * N;
          for (let i = 0; i < N; i++) g[0][o + i] += d * ex.x[i];
        }
      }
      this.t++;
      const b1 = 0.9, b2 = 0.999, n = batch.length;
      const c1 = 1 - Math.pow(b1, this.t), c2 = 1 - Math.pow(b2, this.t);
      this.params.forEach((P, pi) => {
        const G = g[pi], M = this.m[pi], V = this.v[pi];
        for (let i = 0; i < P.length; i++) {
          const gi = G[i] / n;
          M[i] = b1 * M[i] + (1 - b1) * gi;
          V[i] = b2 * V[i] + (1 - b2) * gi * gi;
          P[i] -= (lr * (M[i] / c1)) / (Math.sqrt(V[i] / c2) + 1e-8);
          if (pi % 2 === 0) P[i] -= lr * WD * P[i]; // decoupled weight decay on weights, not biases
        }
      });
    }
    epoch(data, lr, r) {
      const idx = ML.shuffle(data.map((_, i) => i), r);
      for (let s = 0; s < idx.length; s += 32) this.step(idx.slice(s, s + 32).map((i) => data[i]), lr);
    }
    evaluate(data) {
      let loss = 0, ok = 0;
      const cm = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
      for (const ex of data) {
        const { p } = this.forward(ex.x);
        loss -= Math.log(Math.max(1e-12, p[ex.y]));
        const pred = p.indexOf(Math.max(...p));
        if (pred === ex.y) ok++;
        cm[ex.y][pred]++;
      }
      return { loss: loss / data.length, acc: ok / data.length, cm };
    }
    // Gradient of log p(class c) with respect to each input sample.
    saliency(x, c) {
      const H = this.h, { hid, p } = this.forward(x);
      const u = new Float64Array(H);
      for (let j = 0; j < H; j++) {
        let s = 0;
        for (let k = 0; k < 3; k++) s += ((k === c ? 1 : 0) - p[k]) * this.W2[k * H + j];
        u[j] = s * (1 - hid[j] * hid[j]);
      }
      const gx = new Float64Array(N);
      for (let j = 0; j < H; j++) for (let i = 0; i < N; i++) gx[i] += u[j] * this.W1[j * N + i];
      return gx;
    }
  }

  // ---------- shared drawing ----------

  const tFmt = (v) => Math.round(v * WIN_MS) + ' ms';
  function tracePlot(host, extra = {}) {
    return new ML.Plot(host, Object.assign({
      aspect: 0.42, maxHeight: 300, minHeight: 200, x: [0, 1], y: YD, xLabel: 'time in beat window', yLabel: 'mV',
      xTicks: 4, yTicks: 4, xFmt: tFmt, margin: { l: 42, r: 12 }, label: 'ECG beat trace',
    }, extra));
  }
  function drawCurve(p, f, color, width) {
    const xs = [], ys = [];
    for (let k = 0; k <= 400; k++) { const t = k / 400; xs.push(t); ys.push(f(t)); }
    p.line(xs, ys, color, width);
  }
  const valAt = (x) => (t) => {
    const u = t * N - 0.5;
    const i0 = clamp(Math.floor(u), 0, N - 1), i1 = Math.min(N - 1, i0 + 1), fr = clamp(u - i0, 0, 1);
    return x[i0] * (1 - fr) + x[i1] * fr;
  };

  // ---------- step 1: a beat is numbers ----------

  const S1 = { cls: 0, pick: [0, 0, 0], res: 64, hover: -1 };
  const byClass = [0, 1, 2].map((c) => TRAIN.filter((e) => e.y === c));
  const cur1 = () => byClass[S1.cls][S1.pick[S1.cls] % byClass[S1.cls].length];
  ML.seg('ecg-cls', (v) => { S1.cls = +v; step1(); });
  ML.btn('ecg-next', () => { S1.pick[S1.cls]++; step1(); });
  ML.range('ecg-res', (v) => { S1.res = v; step1(); });

  const pWave = tracePlot('ecg-wave');
  pWave.draw = (c, p) => {
    const ex = cur1();
    p.axes();
    p.clip();
    drawCurve(p, (t) => evalBeat(ex.B, t), T.muted, 1.5);
    const n = S1.res;
    const ts = Array.from({ length: n }, (_, i) => (i + 0.5) / n);
    const vs = ts.map((t) => evalBeat(ex.B, t));
    p.line(ts, vs, T.accent, 2);
    ts.forEach((t, i) => ML.marker(c, 'circle', p.sx(t), p.sy(vs[i]), n > 40 ? 2.6 : 3.6, T.accent, T.surface));
    if (S1.hover >= 0) {
      const t = tAt(S1.hover);
      c.strokeStyle = T.ink;
      c.lineWidth = 1;
      c.beginPath(); c.moveTo(p.sx(t), p.m.t); c.lineTo(p.sx(t), p.m.t + p.ih); c.stroke();
      ML.marker(c, 'circle', p.sx(t), p.sy(ex.x[S1.hover]), 5, T.ink, T.surface);
    }
    c.restore();
    c.fillStyle = T['ink-2'];
    c.font = '600 12px ' + ML.FONT;
    c.textAlign = 'right';
    c.fillText(`${CLS[ex.y]} · ${n} samples`, p.m.l + p.iw - 4, p.m.t + 14);
  };

  function cellColor(v) {
    const a = clamp(Math.abs(v) / 1.2, 0, 1);
    return ML.mix(T.grid, v < 0 ? T.neg : T.pos, Math.pow(a, 0.7));
  }
  const pStrip = new ML.Plot('ecg-strip', {
    height: 78, x: [0, N], y: [0, 1], margin: { l: 42, r: 12, t: 6, b: 26 }, label: 'The 64 input values as colored cells',
  });
  pStrip.draw = (c, p) => {
    const ex = cur1(), w = p.iw / N;
    for (let i = 0; i < N; i++) {
      c.fillStyle = cellColor(ex.x[i]);
      c.fillRect(p.m.l + i * w + 0.5, p.m.t, Math.max(1, w - 1), p.ih);
    }
    if (S1.hover >= 0) {
      c.strokeStyle = T.ink;
      c.lineWidth = 2;
      c.strokeRect(p.m.l + S1.hover * w, p.m.t - 1, w, p.ih + 2);
    }
    c.fillStyle = T.muted;
    c.font = '11px ' + ML.FONT;
    c.textBaseline = 'top';
    c.textAlign = 'left'; c.fillText('input 1', p.m.l, p.m.t + p.ih + 6);
    c.textAlign = 'right'; c.fillText('input 64', p.m.l + p.iw, p.m.t + p.ih + 6);
  };
  pStrip.on('move', (e) => {
    const i = Math.floor(e.x);
    if (i < 0 || i >= N || !e.inside) { pStrip.hideTip(); if (S1.hover !== -1) { S1.hover = -1; pWave.update(); pStrip.update(); } return; }
    S1.hover = i;
    pStrip.showTip(e.px, e.py, `<b>Input ${i + 1}</b><br>${tFmt(tAt(i))} · ${cur1().x[i].toFixed(2)} mV`);
    pWave.update(); pStrip.update();
  });
  pStrip.on('leave', () => { S1.hover = -1; pWave.update(); pStrip.update(); });
  function step1() { pWave.update(); pStrip.update(); }

  // ---------- step 2: one neuron ----------

  function template(kind) {
    const w = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      const t = tAt(i);
      w[i] = kind === 'r' ? gauss(t, 0.4, 0.014)
        : kind === 'p' ? gauss(t, 0.2, 0.028)
        : kind === 'wide' ? gauss(t, 0.44, 0.05) - 0.9 * gauss(t, 0.4, 0.012)
        : kind === 't' ? -gauss(t, 0.71, 0.06)
        : 0;
    }
    if (kind !== 'zero') {
      const m = ML.mean(Array.from(w));
      for (let i = 0; i < N; i++) w[i] -= m;
      const mx = Math.max(...w.map(Math.abs));
      for (let i = 0; i < N; i++) w[i] /= mx;
    }
    return w;
  }
  const S2 = { w: template('r'), beat: 0, last: -1 };
  const act = (x, w) => { let s = 0; for (let i = 0; i < N; i++) s += x[i] * w[i]; return s; };
  const tplSeg = ML.seg('ecg-tpl', (v) => { S2.w = template(v); step2(); });
  const beatOrder = ML.shuffle(NEURON_SET.map((_, i) => i), ML.rng(3));
  ML.btn('ecg-next2', () => { S2.beat = (S2.beat + 1) % beatOrder.length; step2(); });
  const cur2 = () => NEURON_SET[beatOrder[S2.beat]];

  const pNeuron = tracePlot('ecg-neuron', { drag: true, yLabel: 'mV / weight', label: 'Beat with the neuron weight template overlaid; drag to draw weights' });
  pNeuron.draw = (c, p) => {
    p.axes();
    p.clip();
    p.line([0, 1], [0, 0], T.axis, 1);
    const ex = cur2();
    drawCurve(p, (t) => evalBeat(ex.B, t), T.muted, 2);
    const ts = Array.from({ length: N }, (_, i) => tAt(i));
    p.line(ts, Array.from(S2.w), T.accent, 2);
    ts.forEach((t, i) => ML.marker(c, 'circle', p.sx(t), p.sy(S2.w[i]), 2.4, T.accent, null));
    c.restore();
    c.fillStyle = T['ink-2'];
    c.font = '600 12px ' + ML.FONT;
    c.textAlign = 'right';
    c.fillText(`this beat: ${CLS[ex.y]}`, p.m.l + p.iw - 4, p.m.t + 14);
  };
  function drawW(e) {
    const i = clamp(Math.floor(e.x * N), 0, N - 1), v = clamp(e.y, -1.3, 1.8);
    if (S2.last >= 0 && S2.last !== i) {
      const a = S2.last, va = S2.w[a], step = i > a ? 1 : -1;
      for (let k = a; k !== i; k += step) S2.w[k] = va + ((v - va) * (k - a)) / (i - a);
    }
    S2.w[i] = v;
    S2.last = i;
    tplSeg.set(null);
    step2();
  }
  pNeuron.on('down', (e) => { S2.last = -1; if (e.inside) drawW(e); });
  pNeuron.on('move', (e) => { if (e.dragging) drawW(e); });
  pNeuron.on('up', () => { S2.last = -1; });

  const pProd = new ML.Plot('ecg-prod', {
    height: 150, x: [0, 1], y: [-1, 1], xLabel: 'time in beat window', xTicks: 4, yTicks: 2, xFmt: tFmt,
    margin: { l: 42, r: 12, t: 8, b: 36 }, label: 'Each input times its weight',
  });
  pProd.draw = (c, p) => {
    const x = cur2().x, pr = Array.from({ length: N }, (_, i) => x[i] * S2.w[i]);
    const m = Math.max(0.2, ...pr.map(Math.abs)) * 1.1;
    p.o.y = [-m, m];
    p.axes({ yTicks: [-m, 0, m].map((v) => +v.toFixed(1)) });
    const w = p.iw / N, bw = Math.max(1, Math.min(24, w - 2)), y0 = p.sy(0);
    pr.forEach((v, i) => {
      const X = p.m.l + i * w + (w - bw) / 2, Y = p.sy(v);
      c.fillStyle = v >= 0 ? T.pos : T.neg;
      c.fillRect(X, Math.min(Y, y0), bw, Math.max(0.5, Math.abs(Y - y0)));
    });
    c.strokeStyle = T.axis;
    c.beginPath(); c.moveTo(p.m.l, y0 + 0.5); c.lineTo(p.m.l + p.iw, y0 + 0.5); c.stroke();
  };
  pProd.on('move', (e) => {
    const i = Math.floor(e.x * N);
    if (i < 0 || i >= N || !e.inside) { pProd.hideTip(); return; }
    const x = cur2().x[i];
    pProd.showTip(e.px, e.py, `<b>Input ${i + 1}</b><br>${x.toFixed(2)} × ${S2.w[i].toFixed(2)} = ${(x * S2.w[i]).toFixed(2)}`);
  });

  let actDots = [];
  const pActs = new ML.Plot('ecg-acts', {
    height: 170, x: [-5, 5], y: [0, 3], xLabel: 'activation (sum of input × weight)', margin: { l: 70, r: 14, t: 8, b: 36 },
    label: 'Activation of the neuron for every training beat, one row per beat type',
  });
  pActs.draw = (c, p) => {
    const acts = NEURON_SET.map((e) => act(e.x, S2.w));
    let lo = Math.min(...acts), hi = Math.max(...acts);
    if (hi - lo < 1) { lo -= 0.5; hi += 0.5; }
    const pad = (hi - lo) * 0.05;
    p.o.x = [lo - pad, hi + pad];
    p.axes({ yTicks: [], noY: true, yGrid: false });
    c.fillStyle = T['ink-2'];
    c.font = '12px ' + ML.FONT;
    c.textAlign = 'right';
    c.textBaseline = 'middle';
    CLS.forEach((name, k) => c.fillText(name, p.m.l - 8, p.sy(2.5 - k)));
    const r = ML.rng(9);
    actDots = [];
    NEURON_SET.forEach((e, i) => {
      const X = p.sx(acts[i]), Y = p.sy(2.5 - e.y + (r() - 0.5) * 0.6);
      ML.marker(c, CSHAPE[e.y], X, Y, 2.8, ML.rgba(ML.series(e.y), 0.8), null);
      actDots.push({ X, Y, i });
    });
    const cur = beatOrder[S2.beat];
    const d = actDots[cur];
    c.beginPath(); c.arc(d.X, d.Y, 8, 0, Math.PI * 2); c.lineWidth = 2; c.strokeStyle = T.ink; c.stroke();
  };
  pActs.on('move', (e) => {
    let best = -1, bd = 64;
    actDots.forEach((d, k) => { const dd = (d.X - e.px) ** 2 + (d.Y - e.py) ** 2; if (dd < bd) { bd = dd; best = k; } });
    if (best < 0) { pActs.hideTip(); return; }
    const ex = NEURON_SET[actDots[best].i];
    pActs.showTip(e.px, e.py, `${CLS[ex.y]} beat<br>activation ${act(ex.x, S2.w).toFixed(2)}<br><span class="k">click to view it</span>`);
  });
  pActs.on('down', (e) => {
    let best = -1, bd = 64;
    actDots.forEach((d, k) => { const dd = (d.X - e.px) ** 2 + (d.Y - e.py) ** 2; if (dd < bd) { bd = dd; best = k; } });
    if (best >= 0) { S2.beat = beatOrder.indexOf(actDots[best].i); step2(); }
  });

  function step2() {
    const a = act(cur2().x, S2.w);
    text('ecg-act', `= ${a.toFixed(2)}`);
    const acts = NEURON_SET.map((e) => act(e.x, S2.w));
    const pair = (c1, c2) => {
      const s = [], l = [];
      NEURON_SET.forEach((e, i) => { if (e.y === c1 || e.y === c2) { s.push(acts[i]); l.push(e.y === c2 ? 1 : 0); } });
      const au = ML.auc(s, l);
      return Math.max(au, 1 - au);
    };
    const np = pair(0, 1), na = pair(0, 2);
    const word = (v) => (v >= 0.95 ? 'cleanly' : v >= 0.85 ? 'fairly well' : v >= 0.7 ? 'somewhat' : 'barely');
    text('ecg-sep', `On its own, this neuron separates PVCs from normal beats ${word(np)} (AUC ${fmt(np)}) and artifact from normal beats ${word(na)} (AUC ${fmt(na)}). A network combines many such neurons.`);
    pNeuron.update(); pProd.update(); pActs.update();
  }

  // ---------- step 3: train ----------

  const S3 = { net: null, h: 8, seed: 1, epochs: 0, hist: [], running: false, r: ML.rng(77), last: null };
  const lrR = ML.range('ecg-lr', () => {}, (v) => v.toPrecision(2), (v) => Math.pow(10, v), (v) => Math.log10(v));
  ML.range('ecg-hidden', (v) => { S3.h = v; resetNet(); });
  let neuronPlots = [];
  function resetNet() {
    stopTrain();
    S3.net = new MLP(S3.h, S3.seed++);
    S3.epochs = 0;
    S3.hist = [];
    S3.last = { tr: S3.net.evaluate(TRAIN), va: S3.net.evaluate(VAL) };
    buildNeurons();
    trainRefresh();
  }
  function buildNeurons() {
    const host = id('ecg-neurons');
    for (const p of neuronPlots) ML.plots.delete(p);
    host.innerHTML = '';
    neuronPlots = [];
    for (let j = 0; j < S3.h; j++) {
      const d = document.createElement('div');
      host.appendChild(d);
      const p = new ML.Plot(d, { height: 60, x: [0, 1], y: [-1.1, 1.1], margin: { l: 4, r: 26, t: 6, b: 6 }, label: `Hidden neuron ${j + 1} weights` });
      p.draw = (c, pl) => {
        const net = S3.net, H = net.h;
        const w = net.W1.subarray(j * N, j * N + N);
        const m = Math.max(1e-6, ...Array.from(w).map(Math.abs));
        pl.line([0, 1], [0, 0], T.grid, 1);
        pl.line(Array.from({ length: N }, (_, i) => tAt(i)), Array.from(w).map((v) => v / m), T.ink, 1.5);
        const vm = Math.max(1e-6, ...Array.from(net.W2).map(Math.abs));
        for (let k = 0; k < 3; k++) {
          const v = net.W2[k * H + j] / vm, X = pl.w - 22 + k * 7, mid = pl.h / 2;
          c.fillStyle = ML.series(k);
          c.fillRect(X, v >= 0 ? mid - v * (mid - 6) : mid, 5, Math.max(1, Math.abs(v) * (mid - 6)));
        }
        c.fillStyle = T.muted;
        c.font = '10px ' + ML.FONT;
        c.textAlign = 'left';
        c.fillText(String(j + 1), 6, 14);
      };
      neuronPlots.push(p);
    }
  }
  function stopTrain() { S3.running = false; id('ecg-train').textContent = S3.epochs ? 'Keep training' : 'Train'; }
  function trainLoop() {
    if (!S3.running) return;
    const lr = lrR.value;
    {
      S3.net.epoch(TRAIN, lr, S3.r);
      S3.epochs++;
      const tr = S3.net.evaluate(TRAIN), va = S3.net.evaluate(VAL);
      S3.hist.push({ tr: tr.loss, va: va.loss });
      S3.last = { tr, va };
    }
    if (S3.epochs >= 150) stopTrain();
    trainRefresh();
    if (S3.running) requestAnimationFrame(trainLoop);
  }
  ML.btn('ecg-train', () => {
    if (S3.running) { stopTrain(); return; }
    S3.running = true;
    id('ecg-train').textContent = 'Pause';
    requestAnimationFrame(trainLoop);
  });
  ML.btn('ecg-reset', resetNet);

  const pLoss = new ML.Plot('ecg-loss', {
    height: 200, x: [0, 20], y: [0, 1.2], xLabel: 'epoch (one pass through the training beats)', margin: { l: 40, r: 12, b: 38 },
    label: 'Training and validation loss by epoch',
  });
  pLoss.draw = (c, p) => {
    const h = S3.hist;
    p.o.x = [0, Math.max(20, h.length)];
    p.o.y = [0, Math.max(0.5, ...h.map((q) => Math.max(q.tr, q.va))) * 1.05];
    p.axes();
    if (!h.length) {
      c.fillStyle = T.muted; c.font = '12px ' + ML.FONT; c.textAlign = 'center';
      c.fillText('Press Train', p.m.l + p.iw / 2, p.m.t + p.ih / 2);
      return;
    }
    const xs = h.map((_, i) => i + 1);
    p.line(xs, h.map((q) => q.tr), T.s1, 2);
    p.line(xs, h.map((q) => q.va), T.s2, 2);
  };
  pLoss.on('move', (e) => {
    const i = Math.round(e.x) - 1, q = S3.hist[i];
    if (!q || !e.inside) { pLoss.hideTip(); return; }
    pLoss.showTip(e.px, e.py, `epoch ${i + 1}<br><span class="k">training</span> ${fmt(q.tr, 3)}<br><span class="k">validation</span> ${fmt(q.va, 3)}`);
  });

  function cmHtml(cm) {
    let h = '<div class="cm" style="grid-template-columns: auto repeat(3, 1fr)"><div class="h"></div>';
    h += CLS.map((c) => `<div class="h">says<br>${c}</div>`).join('');
    cm.forEach((row, i) => {
      const tot = row.reduce((a, b) => a + b, 0) || 1;
      h += `<div class="h rowh">${CLS[i]}</div>`;
      row.forEach((v, j) => {
        const f = v / tot;
        h += `<div class="c" style="background:${ML.rgba(i === j ? T.s1 : T.s2, 0.08 + 0.5 * f)}"><b>${v}</b><small>${pct(f)}</small></div>`;
      });
    });
    return h + '</div>';
  }
  function trainRefresh() {
    const L = S3.last;
    text('ecg-epoch', S3.epochs);
    text('ecg-tloss', fmt(L.tr.loss, 3));
    text('ecg-vloss', fmt(L.va.loss, 3));
    text('ecg-vacc', pct(L.va.acc, 1));
    id('ecg-cm').innerHTML = cmHtml(L.va.cm);
    pLoss.update();
    neuronPlots.forEach((p) => p.update());
  }

  // ---------- step 4: try to fool it ----------

  const S4 = { ex: '0', drawn: null, qrs: 1, noise: 0, wander: 0, sal: true, last: -1 };
  const exSeg = ML.seg('ecg-ex', (v) => { S4.ex = v; S4.drawn = null; step4(); });
  ML.range('ecg-qrs', (v) => { S4.qrs = v; if (S4.drawn) { S4.drawn = null; exSeg.set(S4.ex); } step4(); }, (v) => '×' + v.toFixed(1));
  ML.range('ecg-noise', (v) => { S4.noise = v; step4(); }, (v) => v.toFixed(2) + ' mV');
  ML.range('ecg-wander', (v) => { S4.wander = v; step4(); }, (v) => v.toFixed(2) + ' mV');
  id('ecg-sal').addEventListener('change', (e) => { S4.sal = e.target.checked; step4(); });
  const noiseVec = (() => { const r = ML.rng(123); return Float64Array.from({ length: N }, () => ML.randn(r)); })();

  function baseSignal() {
    if (S4.drawn) return S4.drawn;
    if (S4.ex === 'flat') return new Float64Array(N);
    const r = ML.rng(500 + +S4.ex);
    return sample(makeBeat(+S4.ex, r, { qrs: S4.qrs, clean: true, pol: 1 }));
  }
  function signal4() {
    const b = baseSignal();
    return Float64Array.from({ length: N }, (_, i) => b[i] + S4.noise * noiseVec[i] + S4.wander * Math.sin(2 * Math.PI * 0.9 * tAt(i) + 0.6));
  }
  let fallback = null;
  function model4() {
    if (S3.net && S3.epochs >= 20) { text('ecg-model-note', `Using the network you trained in step 3 (${S3.h} hidden neurons, ${S3.epochs} epochs).`); return S3.net; }
    if (!fallback) {
      fallback = new MLP(8, 1000);
      const r = ML.rng(5);
      for (let e = 0; e < 60; e++) fallback.epoch(TRAIN, 0.003, r);
    }
    text('ecg-model-note', 'Using a network we trained for you (8 hidden neurons, 60 epochs). Train your own in step 3 and it’ll be used here instead.');
    return fallback;
  }

  const pDraw = tracePlot('ecg-draw', { drag: true, label: 'Editable ECG beat; drag to draw, with saliency shading' });
  pDraw.draw = (c, p) => {
    const x = signal4(), net = model4();
    const { p: pr } = net.forward(x);
    const top = pr.indexOf(Math.max(...pr));
    p.axes();
    p.clip();
    if (S4.sal) {
      // SmoothGrad: average the gradient's size over noisy copies of the input, then smooth lightly.
      const r = ML.rng(1), raw = new Float64Array(N);
      for (let k = 0; k < 16; k++) {
        const g0 = net.saliency(x.map((v) => v + 0.08 * ML.randn(r)), top);
        for (let i = 0; i < N; i++) raw[i] += Math.abs(g0[i]);
      }
      const g = raw.map((v, i) => (raw[Math.max(0, i - 2)] + 2 * raw[Math.max(0, i - 1)] + 3 * v + 2 * raw[Math.min(N - 1, i + 1)] + raw[Math.min(N - 1, i + 2)]) / 9);
      const m = Math.max(1e-9, ...g), w = p.iw / N;
      g.forEach((v, i) => {
        c.fillStyle = ML.rgba(T.accent, 0.55 * Math.pow(v / m, 1.5));
        c.fillRect(p.m.l + i * w, p.m.t, w + 0.5, p.ih);
      });
    }
    p.line([0, 1], [0, 0], T.axis, 1);
    drawCurve(p, valAt(x), T.ink, 2);
    c.restore();
  };
  function drawBeat(e) {
    if (!S4.drawn) {
      // Start the drawing from what's on screen, minus the distortions (they stay adjustable).
      S4.drawn = baseSignal().slice();
      exSeg.set(null);
    }
    const i = clamp(Math.floor(e.x * N), 0, N - 1);
    const v = clamp(e.y - S4.noise * noiseVec[i] - S4.wander * Math.sin(2 * Math.PI * 0.9 * tAt(i) + 0.6), YD[0], YD[1]);
    if (S4.last >= 0 && S4.last !== i) {
      const a = S4.last, va = S4.drawn[a], step = i > a ? 1 : -1;
      for (let k = a; k !== i; k += step) S4.drawn[k] = va + ((v - va) * (k - a)) / (i - a);
    }
    S4.drawn[i] = v;
    S4.last = i;
    step4();
  }
  pDraw.on('down', (e) => { S4.last = -1; if (e.inside) drawBeat(e); });
  pDraw.on('move', (e) => { if (e.dragging) drawBeat(e); });
  pDraw.on('up', () => { S4.last = -1; });

  function step4() {
    const net = model4();
    const { p } = net.forward(signal4());
    const top = p.indexOf(Math.max(...p));
    id('ecg-probs').innerHTML = CLS.map((name, k) =>
      `<div class="pbar${k === top ? ' top' : ''}"><span class="lbl">${name}</span><span class="track"><span class="fill" style="display:block;width:${(p[k] * 100).toFixed(1)}%;background:${ML.series(k)}"></span></span><span class="v">${pct(p[k])}</span></div>`).join('');
    pDraw.update();
  }

  // ---------- routing ----------

  ML.register('ecg', {
    enter(step) {
      if (step === 0) step1();
      if (step === 1) step2();
      if (step === 2) { if (!S3.net) resetNet(); else trainRefresh(); }
      if (step === 3) step4();
    },
    leave(step) { if (step === 2) stopTrain(); },
  });
})();
