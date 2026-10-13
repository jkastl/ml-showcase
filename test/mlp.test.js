const test = require('node:test');
const assert = require('node:assert/strict');
const ML = require('./load.js');

const M = ML.mlp;
const count = (data) => [0, 1, 2].map((c) => data.filter((e) => e.y === c).length);
const lossOn = (net, batch) => batch.reduce((s, ex) => s - Math.log(net.forward(ex.x).p[ex.y]), 0);

// The network the step 4 stress test falls back on: 8 hidden neurons, 60 epochs.
const trained = new M.MLP(8, 1000);
const untrainedVal = trained.evaluate(M.VAL);
{
  const r = ML.rng(5);
  for (let e = 0; e < 60; e++) trained.epoch(M.TRAIN, 0.003, r);
}

test('the beat counts match the page: 1,400 to train on and 270 to validate with', () => {
  assert.deepEqual(count(M.TRAIN), [620, 460, 320]);
  assert.deepEqual(count(M.VAL), [120, 90, 60]);
  assert.ok(M.TRAIN.every((e) => e.x.length === M.N && e.x.every(Number.isFinite)));
});

test('beats are reproducible from their seed', () => {
  const a = M.dataset(9, [4, 4, 4]), b = M.dataset(9, [4, 4, 4]);
  assert.deepEqual(a.map((e) => [e.y, Array.from(e.x)]), b.map((e) => [e.y, Array.from(e.x)]));
});

test('backpropagation matches finite differences of the loss', () => {
  const net = new M.MLP(5, 3);
  // Train a little so no weights are tiny and the softmax isn't uniform.
  const r = ML.rng(1);
  for (let e = 0; e < 2; e++) net.epoch(M.TRAIN.slice(0, 200), 0.01, r);
  const batch = M.TRAIN.slice(300, 306);
  const g = net.gradients(batch);
  const eps = 1e-6;
  net.params.forEach((P, pi) => {
    for (let i = 0; i < P.length; i += Math.max(1, Math.floor(P.length / 12))) {
      const keep = P[i];
      P[i] = keep + eps; const up = lossOn(net, batch);
      P[i] = keep - eps; const dn = lossOn(net, batch);
      P[i] = keep;
      const numeric = (up - dn) / (2 * eps);
      assert.ok(Math.abs(numeric - g[pi][i]) < 1e-6 * Math.max(1, Math.abs(numeric)), `param ${pi}[${i}]: ${numeric} vs ${g[pi][i]}`);
    }
  });
});

test('saliency is the gradient of log p(class) with respect to each input sample', () => {
  const x = Float64Array.from(M.VAL[0].x);
  const eps = 1e-6;
  for (const c of [0, 1, 2]) {
    const s = trained.saliency(x, c);
    for (let i = 0; i < M.N; i += 7) {
      const keep = x[i];
      x[i] = keep + eps; const up = Math.log(trained.forward(x).p[c]);
      x[i] = keep - eps; const dn = Math.log(trained.forward(x).p[c]);
      x[i] = keep;
      assert.ok(Math.abs((up - dn) / (2 * eps) - s[i]) < 1e-6, `class ${c}, sample ${i}`);
    }
  }
});

test('an untrained network is at chance, and training gets validation accuracy above 95%', () => {
  assert.ok(Math.abs(untrainedVal.loss - Math.log(3)) < 0.01, `untrained loss ${untrainedVal.loss}`);
  const v = trained.evaluate(M.VAL);
  assert.ok(v.acc > 0.95, `validation accuracy ${v.acc}`);
  assert.ok(v.loss < 0.2, `validation loss ${v.loss}`);
  assert.equal(v.cm.flat().reduce((a, b) => a + b, 0), M.VAL.length);
});

test('one hidden neuron underfits; 24 learn much more', () => {
  const acc = (h) => {
    const net = new M.MLP(h, 1), r = ML.rng(77);
    for (let e = 0; e < 30; e++) net.epoch(M.TRAIN, Math.pow(10, -2.7), r); // the page's default rate
    return net.evaluate(M.VAL).acc;
  };
  const one = acc(1), many = acc(24);
  assert.ok(many > one + 0.1, `1 neuron ${one}, 24 neurons ${many}`);
});

test("outputs are probabilities, even for a flat line it has never seen, so it can't say \"I don't know\"", () => {
  for (const x of [new Float64Array(M.N), M.VAL[5].x, Float64Array.from({ length: M.N }, (_, i) => (i % 2 ? 2 : -2))]) {
    const { p } = trained.forward(x);
    assert.ok(Math.abs(p[0] + p[1] + p[2] - 1) < 1e-12);
    assert.ok(p.every((v) => v >= 0 && v <= 1));
  }
});

test('the step 2 templates are zero-mean and peak at ±1; Blank is all zeros', () => {
  for (const kind of ['r', 'p', 'wide', 't']) {
    const w = Array.from(M.template(kind));
    assert.ok(Math.abs(ML.mean(w)) < 1e-12, kind);
    assert.ok(Math.abs(Math.max(...w.map(Math.abs)) - 1) < 1e-12, kind);
  }
  assert.ok(M.template('zero').every((v) => v === 0));
  assert.equal(M.act(M.VAL[0].x, M.template('zero')), 0);
});
