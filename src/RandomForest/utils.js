/* eslint-disable @typescript-eslint/naming-convention */
import Matrix from 'ml-matrix';
import * as Random from 'random-js';

export function checkFloat(n) {
  return n > 0.0 && n <= 1.0;
}

function getRandomInt(min, max) {
  return Math.round(Math.random() * (max - min) + min);
}

/**
 * Select n with replacement elements on the training set and
 * values, where n is the size of the training set.
 * @ignore
 * @param {Matrix} trainingSet
 * @param {Array} trainingValue
 * @param {number} seed - seed for the random selection, must be a
 * 32-bit integer.
 * @param {number} nSamples - samples.
 * @return {object} with new X and y.
 */
export function examplesBaggingWithReplacement(
  trainingSet,
  trainingValue,
  seed,
  nSamples,
) {
  const Xr = new Array(nSamples);
  const yr = new Array(nSamples);

  // const oob = new Array(trainingSet.rows).fill(0);
  // let oobN = trainingSet.rows;

  for (let i = 0; i < nSamples; ++i) {
    const index = getRandomInt(0, nSamples - 1);
    Xr[i] = trainingSet.getRow(index);
    yr[i] = trainingValue[index];

    // if (oob[index]++ === 0) {
    //   oobN--;
    // }
  }

  // const Xoob = new Array(oobN);
  // const ioob = new Array(oobN);

  // run backwards to have ioob filled in increasing order
  // for (let i = trainingSet.rows - 1; i >= 0 && oobN > 0; --i) {
  //   if (oob[i] === 0) {
  //     Xoob[--oobN] = trainingSet.getRow(i);
  //     ioob[oobN] = i;
  //   }
  // }

  return {
    X: new Matrix(Xr),
    y: yr,
    // Xoob: new Matrix(Xoob),
    // ioob,
    // seed: engine.next(),
  };
}

/**
 * selects n features from the training set with or without
 * replacement, returns the new training set and the indexes used.
 * @ignore
 * @param {Matrix} trainingSet
 * @param {number} nFeatures - features.
 * @param {boolean} replacement
 * @param {number} seed - seed for the random selection, must be a
 * 32-bit integer.
 * @return {object}
 */
export function featureBagging(trainingSet, nFeatures, replacement, seed) {
  if (trainingSet.columns < nFeatures) {
    throw new RangeError(
      'N should be less or equal to the number of columns of X',
    );
  }

  // Returned matrix
  const toRet = new Matrix(trainingSet.rows, nFeatures);
  let usedIndex;
  let index;

  if (replacement) {
    usedIndex = new Array(nFeatures);
    for (let i = 0; i < nFeatures; ++i) {
      // Select a random feature
      index = getRandomInt(0, trainingSet.columns - 1);
      usedIndex[i] = index;
      toRet.setColumn(i, trainingSet.getColumn(index));
    }
  } else {
    usedIndex = new Set();
    index = getRandomInt(0, trainingSet.columns - 1);
    for (let i = 0; i < nFeatures; ++i) {
      // make sure the next selected feature is different
      while (usedIndex.has(index)) {
        index = getRandomInt(0, trainingSet.columns - 1);
      }
      toRet.setColumn(i, trainingSet.getColumn(index));
      usedIndex.add(index);
    }
    usedIndex = Array.from(usedIndex);
  }

  return {
    X: toRet,
    usedIndex: usedIndex,
    // seed: engine.next(),
  };
}
