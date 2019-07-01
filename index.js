const fs = require('fs');
const tf = require('@tensorflow/tfjs-node');

// eda
const records = fs
  .readFileSync('cars.csv')
  .toString()
  .split('\n')
  .slice(1)
  .map(line => {
    const [mpg, cyl, , , , acc, , ,] = line.split(',').map(s => parseFloat(s));
    return [mpg, cyl, acc];
  })
  .filter(record => !record.some(n => isNaN(n)));

// converty arrays to tensors
const y = tf.tensor1d(records.map(record => record[0]));
const X = tf.tensor2d(records.map(record => record.slice(1)));

// display out the first 5 entries
console.log(y.slice(0, 5).arraySync());
console.log(X.slice(0, 5).arraySync());

// optimizer
const learningRate = 0.001;
const optimizer = tf.train.sgd(learningRate);

// model
const model = tf.sequential();
model.add(tf.layers.dense({ units: 1, inputShape: [2], activation: 'linear' }));
model.compile({ optimizer, loss: 'meanSquaredError' });
model.summary();

// training
(async () => {
  await model.fit(X, y, { epochs: 10 });
})();
