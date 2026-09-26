/* Shared helpers: seeded randomness, stats, theme tokens, canvas plots, controls, routing. */
(function () {
  'use strict';
  const ML = (window.ML = {});

  // ---------- math & stats ----------

  // Mulberry32: small, fast, seedable. Every dataset here is reproducible.
  ML.rng = function (seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  };
  ML.randn = function (r) {
    let u = 0;
    while (u === 0) u = r();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r());
  };
  ML.sigmoid = (z) => 1 / (1 + Math.exp(-z));
  ML.clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  ML.lerp = (a, b, t) => a + (b - a) * t;
  ML.mean = (a) => a.reduce((s, v) => s + v, 0) / (a.length || 1);
  ML.sd = function (a) {
    const m = ML.mean(a);
    return Math.sqrt(a.reduce((s, v) => s + (v - m) * (v - m), 0) / Math.max(1, a.length - 1));
  };
  ML.shuffle = function (arr, r) {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(r() * (i + 1));
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
    return arr;
  };

  // Area under the ROC curve via the Mann-Whitney rank sum (ties count half).
  ML.auc = function (scores, labels) {
    const idx = scores.map((_, i) => i).sort((a, b) => scores[a] - scores[b]);
    let rankSum = 0, nPos = 0, i = 0;
    while (i < idx.length) {
      let j = i;
      while (j + 1 < idx.length && scores[idx[j + 1]] === scores[idx[i]]) j++;
      const avg = (i + j) / 2 + 1;
      for (let k = i; k <= j; k++) if (labels[idx[k]]) { rankSum += avg; nPos++; }
      i = j + 1;
    }
    const nNeg = idx.length - nPos;
    if (!nPos || !nNeg) return NaN;
    return (rankSum - (nPos * (nPos + 1)) / 2) / (nPos * nNeg);
  };

  // ROC points from the strictest threshold to the loosest.
  ML.roc = function (scores, labels) {
    const idx = scores.map((_, i) => i).sort((a, b) => scores[b] - scores[a]);
    const P = labels.filter(Boolean).length, N = labels.length - P;
    const pts = [{ fpr: 0, tpr: 0 }];
    let tp = 0, fp = 0, i = 0;
    while (i < idx.length) {
      const s = scores[idx[i]];
      while (i < idx.length && scores[idx[i]] === s) { labels[idx[i]] ? tp++ : fp++; i++; }
      pts.push({ fpr: fp / N, tpr: tp / P, t: s });
    }
    return pts;
  };

  // Confusion counts and the usual clinical metrics at one threshold (flag if score >= t).
  ML.confusion = function (scores, labels, t) {
    let tp = 0, fp = 0, tn = 0, fn = 0;
    for (let i = 0; i < scores.length; i++) {
      const flag = scores[i] >= t;
      if (labels[i]) flag ? tp++ : fn++;
      else flag ? fp++ : tn++;
    }
    const div = (a, b) => (b ? a / b : NaN);
    return {
      tp, fp, tn, fn,
      sens: div(tp, tp + fn), spec: div(tn, tn + fp),
      ppv: div(tp, tp + fp), npv: div(tn, tn + fn),
      acc: div(tp + tn, scores.length),
    };
  };

  ML.ticks = function (a, b, n) {
    const step0 = (b - a) / Math.max(1, n);
    const mag = Math.pow(10, Math.floor(Math.log10(step0)));
    const err = step0 / mag;
    const step = (err >= 7.5 ? 10 : err >= 3.5 ? 5 : err >= 1.5 ? 2 : 1) * mag;
    const out = [];
    for (let v = Math.ceil(a / step - 1e-9) * step; v <= b + 1e-9; v += step) out.push(+v.toFixed(10));
    return out;
  };

  ML.fmt = (v, d = 2) => (Number.isFinite(v) ? v.toFixed(d) : '–');
  ML.pct = (v, d = 0) => (Number.isFinite(v) ? (v * 100).toFixed(d) + '%' : '–');
  ML.signed = (v, d = 2) => (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(d);

  // ---------- theme ----------

  const TOKENS = ['surface', 'page', 'ink', 'ink-2', 'muted', 'grid', 'axis', 'accent',
    's1', 's2', 's3', 's4', 's5', 's6', 'good', 'critical', 'seq-lo', 'seq-hi', 'neg', 'pos'];
  ML.theme = {};
  ML.FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif';
  ML.readTheme = function () {
    const cs = getComputedStyle(document.documentElement);
    for (const t of TOKENS) ML.theme[t] = cs.getPropertyValue('--' + t).trim();
  };
  function hexRgb(hex) {
    const h = hex.replace('#', '');
    const n = parseInt(h.length === 3 ? h.replace(/./g, '$&$&') : h, 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }
  ML.rgb = hexRgb;
  ML.rgba = function (hex, a) {
    const [r, g, b] = hexRgb(hex);
    return `rgba(${r},${g},${b},${a})`;
  };
  ML.mix = function (a, b, t) {
    const A = hexRgb(a), B = hexRgb(b);
    return `rgb(${A.map((v, i) => Math.round(v + (B[i] - v) * t)).join(',')})`;
  };
  ML.series = (i) => ML.theme['s' + ((i % 6) + 1)];

  // ---------- plots ----------

  ML.plots = new Set();

  class Plot {
    constructor(host, o = {}) {
      this.host = typeof host === 'string' ? document.getElementById(host) : host;
      this.o = Object.assign({
        height: 300, aspect: 0, maxHeight: 0, minHeight: 0,
        x: [0, 1], y: [0, 1], xLabel: '', yLabel: '',
        xTicks: 6, yTicks: 5, xFmt: null, yFmt: null, xGrid: false, drag: false,
      }, o);
      this.m = Object.assign({ t: 12, r: 14, b: 40, l: 48 }, o.margin || {});
      this.host.classList.add('plot');
      this.canvas = document.createElement('canvas');
      this.canvas.setAttribute('role', 'img');
      if (o.label) this.canvas.setAttribute('aria-label', o.label);
      this.host.appendChild(this.canvas);
      this.tip = document.createElement('div');
      this.tip.className = 'tip';
      this.tip.hidden = true;
      this.host.appendChild(this.tip);
      this.ctx = this.canvas.getContext('2d');
      this.handlers = {};
      this.draw = o.draw || null;
      this.w = 0;
      this.h = 0;
      // drag: 'x' is a sideways-only drag, so a vertical swipe can still scroll the page.
      this.canvas.style.touchAction = this.o.drag && this.o.drag !== 'x' ? 'none' : 'pan-y';
      this._bindPointer();
      new ResizeObserver(() => this.resize()).observe(this.host);
      ML.plots.add(this);
    }
    resize() {
      const w = Math.floor(this.host.clientWidth);
      if (!w) return;
      let h = this.o.aspect ? Math.round(w * this.o.aspect) : this.o.height;
      if (this.o.maxHeight) h = Math.min(h, this.o.maxHeight);
      if (this.o.minHeight) h = Math.max(h, this.o.minHeight);
      const dpr = window.devicePixelRatio || 1;
      if (w === this.w && h === this.h && dpr === this.dpr) return;
      this.dpr = dpr;
      this.w = w;
      this.h = h;
      this.canvas.width = Math.round(w * dpr);
      this.canvas.height = Math.round(h * dpr);
      this.canvas.style.width = w + 'px';
      this.canvas.style.height = h + 'px';
      this.render();
    }
    get iw() { return this.w - this.m.l - this.m.r; }
    get ih() { return this.h - this.m.t - this.m.b; }
    sx(v) { const [a, b] = this.o.x; return this.m.l + ((v - a) / (b - a)) * this.iw; }
    sy(v) { const [a, b] = this.o.y; return this.m.t + (1 - (v - a) / (b - a)) * this.ih; }
    ix(px) { const [a, b] = this.o.x; return a + ((px - this.m.l) / this.iw) * (b - a); }
    iy(py) { const [a, b] = this.o.y; return a + (1 - (py - this.m.t) / this.ih) * (b - a); }
    update() {
      if (this._raf) return;
      this._raf = requestAnimationFrame(() => { this._raf = 0; this.render(); });
    }
    render() {
      if (!this.w) return;
      const c = this.ctx;
      c.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
      c.clearRect(0, 0, this.w, this.h);
      if (this.draw) this.draw(c, this);
    }
    clip() {
      const c = this.ctx;
      c.save();
      c.beginPath();
      c.rect(this.m.l, this.m.t, this.iw, this.ih);
      c.clip();
    }
    axes(opt = {}) {
      const c = this.ctx, T = ML.theme, m = this.m, o = this.o;
      const xt = opt.xTicks || ML.ticks(o.x[0], o.x[1], o.xTicks);
      const yt = opt.yTicks || ML.ticks(o.y[0], o.y[1], o.yTicks);
      const fx = o.xFmt || ((v) => String(v));
      const fy = o.yFmt || ((v) => String(v));
      const bottom = m.t + this.ih;
      c.save();
      c.lineWidth = 1;
      c.strokeStyle = T.grid;
      c.beginPath();
      if (opt.yGrid !== false) for (const v of yt) {
        const y = Math.round(this.sy(v)) + 0.5;
        c.moveTo(m.l, y); c.lineTo(m.l + this.iw, y);
      }
      if (o.xGrid) for (const v of xt) {
        const x = Math.round(this.sx(v)) + 0.5;
        c.moveTo(x, m.t); c.lineTo(x, bottom);
      }
      c.stroke();
      c.strokeStyle = T.axis;
      c.beginPath();
      c.moveTo(m.l, Math.round(bottom) + 0.5); c.lineTo(m.l + this.iw, Math.round(bottom) + 0.5);
      c.stroke();
      c.fillStyle = T.muted;
      c.font = '11px ' + ML.FONT;
      c.textAlign = 'center';
      c.textBaseline = 'top';
      if (!opt.noX) for (const v of xt) c.fillText(fx(v), this.sx(v), bottom + 6);
      c.textAlign = 'right';
      c.textBaseline = 'middle';
      if (!opt.noY) for (const v of yt) c.fillText(fy(v), m.l - 7, this.sy(v));
      c.fillStyle = T['ink-2'];
      c.font = '12px ' + ML.FONT;
      if (o.xLabel) {
        c.textAlign = 'center';
        c.textBaseline = 'bottom';
        c.fillText(o.xLabel, m.l + this.iw / 2, this.h - 2);
      }
      if (o.yLabel) {
        c.save();
        c.translate(12, m.t + this.ih / 2);
        c.rotate(-Math.PI / 2);
        c.textAlign = 'center';
        c.textBaseline = 'middle';
        c.fillText(o.yLabel, 0, 0);
        c.restore();
      }
      c.restore();
    }
    // A thin reference line (a published cutoff, the chance diagonal) with an optional label.
    refLine(x0, y0, x1, y1, label, align) {
      const c = this.ctx, T = ML.theme;
      c.save();
      c.strokeStyle = T.muted;
      c.lineWidth = 1;
      c.beginPath();
      c.moveTo(this.sx(x0), this.sy(y0));
      c.lineTo(this.sx(x1), this.sy(y1));
      c.stroke();
      if (label) {
        c.fillStyle = T.muted;
        c.font = '11px ' + ML.FONT;
        c.textAlign = align || 'left';
        c.textBaseline = 'top';
        c.fillText(label, this.sx(x1) + (align === 'right' ? -4 : 4), this.sy(y1) + 2);
      }
      c.restore();
    }
    line(xs, ys, color, width = 2) {
      const c = this.ctx;
      if (!xs.length) return;
      c.save();
      c.strokeStyle = color;
      c.lineWidth = width;
      c.lineJoin = 'round';
      c.lineCap = 'round';
      c.beginPath();
      for (let i = 0; i < xs.length; i++) {
        const X = this.sx(xs[i]), Y = this.sy(ys[i]);
        i ? c.lineTo(X, Y) : c.moveTo(X, Y);
      }
      c.stroke();
      c.restore();
    }
    on(evt, fn) { this.handlers[evt] = fn; return this; }
    _bindPointer() {
      const cv = this.canvas;
      const pos = (e) => {
        const r = cv.getBoundingClientRect();
        const px = e.clientX - r.left, py = e.clientY - r.top;
        const m = this.m;
        return {
          px, py, x: this.ix(px), y: this.iy(py), e,
          inside: px >= m.l && px <= m.l + this.iw && py >= m.t && py <= m.t + this.ih,
          dragging: this.dragging,
        };
      };
      // Touch has no hover: a tap (or a sideways scrub on a scrollable plot) stands in for it,
      // and the readout it shows stays pinned until the next tap elsewhere.
      cv.addEventListener('pointerdown', (e) => {
        this.touch = e.pointerType === 'touch';
        this.dragging = !!this.o.drag;
        if (this.o.drag) cv.setPointerCapture(e.pointerId);
        if (this.handlers.down) this.handlers.down(pos(e));
        if (this.touch) {
          this.pinned = true;
          if (this.handlers.move) this.handlers.move(Object.assign(pos(e), { dragging: false }));
        }
      });
      cv.addEventListener('pointermove', (e) => {
        this.touch = e.pointerType === 'touch';
        if (this.handlers.move) this.handlers.move(pos(e));
      });
      const end = (e) => {
        if (!this.dragging) return;
        this.dragging = false;
        if (this.handlers.up) this.handlers.up(pos(e));
      };
      cv.addEventListener('pointerup', end);
      cv.addEventListener('pointercancel', (e) => {
        end(e);
        // The browser took the gesture over for scrolling.
        if (e.pointerType === 'touch') this.unpin();
      });
      cv.addEventListener('pointerleave', (e) => {
        if (e.pointerType === 'touch') return;
        if (!this.o.drag) this.dragging = false;
        this.hideTip();
        if (this.handlers.leave) this.handlers.leave();
      });
    }
    unpin() {
      if (!this.pinned) return;
      this.pinned = false;
      this.hideTip();
      if (this.handlers.leave) this.handlers.leave();
    }
    showTip(px, py, html) {
      const t = this.tip;
      t.innerHTML = html;
      t.hidden = false;
      const tw = t.offsetWidth, th = t.offsetHeight;
      // Keep the tip clear of a fingertip.
      const gap = this.touch ? 36 : 10;
      let x = px + 14, y = py - th - gap;
      if (x + tw > this.w) x = px - tw - 14;
      if (x < 0) x = 0;
      if (y < 0) y = this.touch ? 0 : py + 16;
      t.style.transform = `translate(${Math.round(x)}px, ${Math.round(y)}px)`;
    }
    hideTip() { this.tip.hidden = true; }
    // Index of the item nearest the pointer within maxPx, or -1.
    nearest(items, p, getX, getY, maxPx = 14) {
      if (this.touch) maxPx = Math.max(maxPx, 22);
      let best = -1, bd = maxPx * maxPx;
      for (let i = 0; i < items.length; i++) {
        const dx = this.sx(getX(items[i])) - p.px, dy = this.sy(getY(items[i])) - p.py;
        const d = dx * dx + dy * dy;
        if (d < bd) { bd = d; best = i; }
      }
      return best;
    }
  }
  ML.Plot = Plot;

  // Sliders and checkboxes keep a pinned readout, so you can watch it change.
  document.addEventListener('pointerdown', (e) => {
    if (e.target.closest && e.target.closest('input')) return;
    for (const p of ML.plots) if (p.pinned && e.target !== p.canvas) p.unpin();
  });

  // Filled marker with a surface-colored ring, so overlapping dots stay legible.
  ML.SHAPES = ['circle', 'square', 'triangle', 'diamond', 'down', 'plus'];
  ML.marker = function (c, shape, x, y, r, fill, ring) {
    c.beginPath();
    switch (shape) {
      case 'square': c.rect(x - r * 0.86, y - r * 0.86, r * 1.72, r * 1.72); break;
      case 'triangle':
        c.moveTo(x, y - r * 1.15); c.lineTo(x + r * 1.05, y + r * 0.75); c.lineTo(x - r * 1.05, y + r * 0.75); c.closePath();
        break;
      case 'down':
        c.moveTo(x, y + r * 1.15); c.lineTo(x + r * 1.05, y - r * 0.75); c.lineTo(x - r * 1.05, y - r * 0.75); c.closePath();
        break;
      case 'diamond':
        c.moveTo(x, y - r * 1.2); c.lineTo(x + r * 1.2, y); c.lineTo(x, y + r * 1.2); c.lineTo(x - r * 1.2, y); c.closePath();
        break;
      case 'plus': {
        const a = r * 1.1, b = r * 0.42;
        c.moveTo(x - b, y - a); c.lineTo(x + b, y - a); c.lineTo(x + b, y - b); c.lineTo(x + a, y - b);
        c.lineTo(x + a, y + b); c.lineTo(x + b, y + b); c.lineTo(x + b, y + a); c.lineTo(x - b, y + a);
        c.lineTo(x - b, y + b); c.lineTo(x - a, y + b); c.lineTo(x - a, y - b); c.lineTo(x - b, y - b); c.closePath();
        break;
      }
      default: c.arc(x, y, r, 0, Math.PI * 2);
    }
    if (ring) { c.lineWidth = 3; c.lineJoin = 'round'; c.strokeStyle = ring; c.stroke(); }
    if (fill) { c.fillStyle = fill; c.fill(); }
  };

  // ---------- controls ----------

  ML.$ = (sel, root = document) => root.querySelector(sel);
  ML.$$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
  ML.id = (id) => document.getElementById(id);
  ML.text = (id, v) => { const el = ML.id(id); if (el) el.textContent = v; };

  // Range input with a live <output>. `map` turns the raw slider value into the model value.
  ML.range = function (id, onInput, fmt = (v) => v, map = (v) => v, unmap = (v) => v) {
    const input = ML.id(id);
    const out = ML.$(`output[for="${id}"]`);
    const api = {
      input,
      get value() { return map(parseFloat(input.value)); },
      set(v, fire) {
        input.value = unmap(v);
        if (out) out.textContent = fmt(api.value);
        if (fire) onInput(api.value);
      },
    };
    input.addEventListener('input', () => {
      if (out) out.textContent = fmt(api.value);
      onInput(api.value);
    });
    if (out) out.textContent = fmt(api.value);
    return api;
  };

  // Segmented control: a row of buttons, one pressed.
  ML.seg = function (id, onChange) {
    const root = ML.id(id);
    const btns = ML.$$('button', root);
    const api = {
      get value() { const b = btns.find((x) => x.getAttribute('aria-pressed') === 'true'); return b ? b.dataset.v : null; },
      set(v, fire) {
        for (const b of btns) b.setAttribute('aria-pressed', String(b.dataset.v === v));
        if (fire) onChange(v);
      },
    };
    for (const b of btns) b.addEventListener('click', () => api.set(b.dataset.v, true));
    return api;
  };

  ML.btn = (id, fn) => { const b = ML.id(id); b.addEventListener('click', fn); return b; };

  // ---------- routing ----------

  const hooks = {};
  ML.register = (id, h) => { hooks[id] = h; };
  let current = { view: null, step: 0 };

  function buildSteppers() {
    const chapters = ML.$$('section.chapter');
    chapters.forEach((sec, ci) => {
      const id = sec.dataset.view;
      const steps = ML.$$('.step', sec);
      const nav = ML.$('.stepper', sec);
      steps.forEach((st, i) => {
        const a = document.createElement('a');
        a.href = `#/${id}/${i + 1}`;
        a.className = 'pill';
        a.innerHTML = `<span class="num">${i + 1}</span><span class="lbl">${st.dataset.name}</span>`;
        nav.appendChild(a);
        const foot = document.createElement('div');
        foot.className = 'step-foot';
        const prev = i > 0 ? `<a class="btn ghost" href="#/${id}/${i}">← ${steps[i - 1].dataset.name}</a>` : '<span></span>';
        let next;
        if (i < steps.length - 1) next = `<a class="btn primary" href="#/${id}/${i + 2}">Next: ${steps[i + 1].dataset.name} →</a>`;
        else {
          const nc = sec.dataset.next;
          const nt = nc && ML.$(`section[data-view="${nc}"]`);
          next = nc ? `<a class="btn primary" href="#/${nc}${nt && nt.classList.contains('chapter') ? '/1' : ''}">Next chapter: ${nt.dataset.short} →</a>` : '';
        }
        foot.innerHTML = `${prev}<span class="count">Step ${i + 1} of ${steps.length}</span>${next}`;
        st.appendChild(foot);
      });
    });
  }

  function route() {
    const parts = location.hash.replace(/^#\/?/, '').split('/');
    let view = parts[0] || 'home';
    let sec = ML.$(`section.view[data-view="${view}"]`);
    if (!sec) { view = 'home'; sec = ML.$('section.view[data-view="home"]'); }
    const steps = ML.$$('.step', sec);
    let step = Math.max(1, Math.min(steps.length || 1, parseInt(parts[1], 10) || 1)) - 1;

    if (current.view && hooks[current.view] && hooks[current.view].leave &&
        (current.view !== view || current.step !== step)) {
      hooks[current.view].leave(current.step);
    }
    const changedView = current.view !== view;
    for (const s of ML.$$('section.view')) s.hidden = s !== sec;
    steps.forEach((st, i) => { st.hidden = i !== step; });
    ML.$$('.pill', sec).forEach((p, i) => {
      p.classList.toggle('active', i === step);
      if (i === step) p.setAttribute('aria-current', 'step'); else p.removeAttribute('aria-current');
    });
    for (const a of ML.$$('.topnav a')) {
      const on = a.dataset.view === view;
      a.classList.toggle('active', on);
      if (on) a.setAttribute('aria-current', 'page'); else a.removeAttribute('aria-current');
    }
    const title = sec.dataset.short;
    document.title = view === 'home' ? 'ML Showcase' : `${title}${steps.length ? ' · ' + steps[step].dataset.name : ''} · ML Showcase`;
    current = { view, step };
    if (hooks[view] && hooks[view].enter) hooks[view].enter(step);

    if (changedView) window.scrollTo(0, 0);
    else {
      const nav = ML.$('.stepper', sec);
      if (nav && nav.getBoundingClientRect().top < 0) nav.scrollIntoView({ block: 'start' });
    }
  }

  ML.start = function () {
    ML.readTheme();
    buildSteppers();
    window.addEventListener('hashchange', route);
  };
  ML.route = route;
})();
