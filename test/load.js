// Loads the engine the same way the page does: each file attaches to globalThis.ML.
const path = require('node:path');
for (const f of ['stats', 'logistic', 'tree', 'kmeans', 'mlp']) {
  require(path.join(__dirname, '..', 'js', 'engine', f + '.js'));
}
module.exports = globalThis.ML;
