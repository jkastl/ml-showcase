/* Chapter 4: simulated single heartbeats (normal, PVC, artifact) and a small neural network
 * that labels them, with backpropagation, Adam and saliency. No DOM.
 */
(function (ML) {
  'use strict';
  const { clamp } = ML;
  const N = 64; // samples per beat window
  const WIN_MS = 800;
  const CLS = ['Normal', 'PVC', 'Artifact'];

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
    // Backpropagation: the gradient of the summed cross-entropy loss over `batch` with respect to
    // [W1, b1, W2, b2], in the same layout as this.params.
    gradients(batch) {
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
      return g;
    }
    // One Adam update on the batch's mean gradient.
    step(batch, lr) {
      const g = this.gradients(batch);
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

  // ---------- one neuron, by hand ----------

  // The step 2 weight templates, zero-mean and scaled so the largest weight is ±1.
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
  // A neuron's activation before the nonlinearity: the dot product of input and weights.
  const act = (x, w) => { let s = 0; for (let i = 0; i < N; i++) s += x[i] * w[i]; return s; };

  ML.mlp = { N, WIN_MS, CLS, gauss, makeBeat, evalBeat, tAt, sample, dataset, TRAIN, VAL, MLP, template, act };
})(globalThis.ML = globalThis.ML || {});
