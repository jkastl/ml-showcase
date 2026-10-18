# ml-showcase

Machine learning, step by step, on simulated clinical problems. Four models are trained live in
the browser, and every step has something to drag, click or switch.

**[jkastl.github.io/ml-showcase](https://jkastl.github.io/ml-showcase/)**

[![Tests](https://github.com/jkastl/ml-showcase/actions/workflows/tests.yml/badge.svg)](https://github.com/jkastl/ml-showcase/actions/workflows/tests.yml)

| Chapter | Technique | Clinical question | Steps |
|---|---|---|---|
| 1 · Risk | Logistic regression | Who will develop type 2 diabetes within 5 years? | Meet the data, fit it by hand, gradient descent, pick a threshold (ROC), a new hospital (calibration and dataset shift) |
| 2 · Triage | Decision tree (CART) | Which emergency patients with suspected infection will deteriorate? | One split (Gini), grow a tree, overfitting, follow one patient (compared with qSOFA) |
| 3 · Subgroups | k-means clustering | Are there distinct subgroups of diabetes at diagnosis? | Place centers, assign/move iterations, choosing k (elbow, silhouette), feature scaling |
| 4 · ECG | Neural network (MLP) | Is this heartbeat normal, a PVC, or artifact? | A beat as numbers, one neuron as a template, training with backprop, stress test with saliency |

A wrap-up page ties each chapter to what it takes to deploy a clinical model responsibly.

All patients and heartbeats are generated in the page by seeded formulas, so every visit sees the
same data. Nothing is fetched, and none of it is sent anywhere; the page only records an anonymous
visit count ([GoatCounter](https://www.goatcounter.com/), no cookies). It's a teaching tool, not a clinical one.

## Running it

No build step and no dependencies. Open `index.html` in a browser, or serve the folder:

```sh
python3 -m http.server
```

GitHub Pages serves the repo root from `main`. Commit and push changes directly to `main`; there are
no feature branches or pull requests.

## Tests

The models live in `js/engine/`, apart from the page, so Node can run them. Tests use Node's built-in
runner (Node 22, as in CI) and take about a second:

```sh
node --test 'test/*.test.js'
```

They check the math against its definition: the logistic regression and network gradients against
finite differences, saliency against a numerical derivative, AUC against counting every
case/non-case pair, and each Gini cut against counting both sides directly. They also check what the
page tells you will happen:

- **Risk:** the fitted model gets under the 0.50 log loss the page sets as a target, and each
  threshold preset meets its target. At the new hospital, AUC barely moves while calibration slips.
  Recalibrating matches the 200 local patients exactly and shrinks the gap on everyone else.
- **Triage:** training AUC only rises with depth, while held-out AUC peaks and then falls. No leaf
  is smaller than the minimum. Growing one split at a time gives the same tree as building it
  depth-first. The depth-3 tree finds qSOFA's respiratory-rate cutoff on its own.
- **Subgroups:** neither the assign step nor the move step ever raises inertia, the bad start gets
  stuck, and silhouette peaks at the four simulated groups. Standardized units recover those groups
  far better than raw units.
- **ECG:** an untrained network is at chance, and training gets validation accuracy above 95%. One
  hidden neuron underfits, and the outputs are probabilities even for a flat line.

A GitHub Action runs them on every push to `main` and on pull requests.

## Layout

```
index.html            all chapter text and page structure; hash routes like #/risk/3
style.css             dark theme (the app is dark-only)
js/engine/            the data and models; no DOM, loads in the browser and in Node
  stats.js            seeded RNG, AUC, ROC, confusion matrix, axis ticks, number formats
  logistic.js         chapter 1: simulated patients, logistic regression, thresholds, recalibration
  tree.js             chapter 2: simulated patients, CART
  kmeans.js           chapter 3: simulated patients, k-means, k-means++, silhouette
  mlp.js              chapter 4: simulated heartbeats, the network (backprop, Adam), saliency
js/core.js            canvas plot helper, controls, router
js/risk.js            chapter 1 page
js/triage.js          chapter 2 page
js/subgroups.js       chapter 3 page
js/ecg.js             chapter 4 page
test/                 node:test suites, one per engine file
```

The engine files are classic scripts that attach to one global, `ML`, so the page works from
`file://` without a bundler and Node can `require()` the same files. Each chapter page script builds
its plots when the page loads and registers `enter`/`leave` hooks with the router, so training loops
stop when you navigate away.

## Versioning

The version and date in the footer of `index.html` are **updated by hand**. Nothing bumps them
automatically. Change both in the same commit as the change they describe, following
[semver](https://semver.org/):

- **Patch** (`1.2.0` → `1.2.1`): fixing a typo or tweaking the wording.
- **Minor** (`1.2.0` → `1.3.0`): adding, removing, or changing a step or chapter, or a small
  visual change.
- **Major** (`1.2.0` → `2.0.0`): a redesign or restructure of the app.

The date is the day of the change, in `YYYY-MM-DD` format.
