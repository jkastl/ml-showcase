# ml-showcase

Machine learning, step by step, on simulated clinical problems. Four models are trained live in
the browser, and every step has something to drag, click or switch.

**[jkastl.github.io/ml-showcase](https://jkastl.github.io/ml-showcase/)**

| Chapter | Technique | Clinical question | Steps |
|---|---|---|---|
| 1 · Risk | Logistic regression | Who will develop type 2 diabetes within 5 years? | Meet the data, fit it by hand, gradient descent, pick a threshold (ROC), a new hospital (calibration and dataset shift) |
| 2 · Triage | Decision tree (CART) | Which emergency patients with suspected infection will deteriorate? | One split (Gini), grow a tree, overfitting, follow one patient (compared with qSOFA) |
| 3 · Subgroups | k-means clustering | Are there distinct subgroups of diabetes at diagnosis? | Place centers, assign/move iterations, choosing k (elbow, silhouette), feature scaling |
| 4 · ECG | Neural network (MLP) | Is this heartbeat normal, a PVC, or artifact? | A beat as numbers, one neuron as a template, training with backprop, stress test with saliency |

A wrap-up page ties each chapter to what it takes to deploy a clinical model responsibly.

All patients and heartbeats are generated in the page by seeded formulas, so every visit sees the
same data. Nothing is fetched or sent anywhere. It's a teaching tool, not a clinical one.

## Running it

No build step and no dependencies. Open `index.html` in a browser, or serve the folder:

```sh
python3 -m http.server
```

GitHub Pages serves the repo root from `main`.

## Layout

```
index.html        all chapter text and page structure; hash routes like #/risk/3
style.css         light and dark themes (follows the OS setting)
js/core.js        seeded RNG, stats (AUC, ROC, confusion), canvas plot helper, controls, router
js/risk.js        chapter 1: logistic regression
js/triage.js      chapter 2: decision tree
js/subgroups.js   chapter 3: k-means
js/ecg.js         chapter 4: neural network
```

Each chapter script builds its plots when the page loads and registers `enter`/`leave` hooks with
the router, so training loops stop when you navigate away.

## Versioning

The version and date in the footer of `index.html` are updated by hand, following
[semver](https://semver.org/): patch for wording fixes, minor for new steps or visual changes,
major for a restructure.
