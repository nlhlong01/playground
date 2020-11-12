/* eslint-disable @typescript-eslint/naming-convention */
import arrayMean from 'ml-array-mean';
import arrayMode from 'ml-array-mode';
import { RandomForestClassifier } from 'ml-random-forest';
import { DecisionTreeClassifier as DTClassifier } from 'ml-cart';
import { Matrix } from 'ml-matrix';
import * as Utils from './utils';

const selectionMethods = {
  mean: arrayMean,
  mode: arrayMode
};

/**
 * The classifier now has the percentage of samples for each estimator to train
 * instead of always using 100% data.
 * @extends RandomForestClassifier
 */
export class CustomRandomForestClassifier extends RandomForestClassifier {
  /**
   * @param {number} [options.maxSamples] - the number of samples used on each
   * estimator, only used when useSampleBagging=0
   *        * if is an integer it selects maxFeatures elements over the sample
   * features.
   *        * if is a float between (0, 1), it takes the percentage of features.
   * @param {number|String} [options.maxFeatures] - the number of features used
   * on each estimator.
   *        * if is an integer it selects maxFeatures elements over the sample
   * features.
   *        * if is a float between (0, 1), it takes the percentage of features.
   */
  constructor(options, model) {
    super(options, model);
    if (options === true) {
      this.maxSamples = model.maxSamples;
      this.nSamples = model.nSamples;
      this.maxFeatures = model.maxFeatures;
      this.nFeatures = model.nFeatures;
      this.selectionMethod = model.selectionMethod;
    } else {
      this.maxSamples = options.maxSamples;
      this.maxFeatures = options.maxFeatures;

      if (
        !(
          options.selectionMethod === 'mean' ||
          options.selectionMethod === 'mode'
        )
      ) {
        throw new RangeError(
          `Unsupported selection method ${options.selectionMethod}`,
        );
      }
      this.selectionMethod = options.selectionMethod;
    }
  }

  selection(values) {
    return selectionMethods[this.selectionMethod](values);
  }

  train(trainingSet, trainingValues) {
    let currentSeed = this.seed;

    trainingSet = Matrix.checkMatrix(trainingSet);

    this.maxFeatures = this.maxFeatures || trainingSet.columns;

    if (Utils.checkFloat(this.maxFeatures)) {
      this.nFeatures = Math.floor(trainingSet.columns * this.maxFeatures);
    } else if (Number.isInteger(this.maxFeatures)) {
      if (this.maxFeatures > trainingSet.columns) {
        throw new RangeError(
          `The maxFeatures parameter should be less than ${trainingSet.columns}`
        );
      } else {
        this.nFeatures = this.maxFeatures;
      }
    } else {
      throw new RangeError(
        `Cannot process the maxFeatures parameter ${this.maxFeatures}`,
      );
    }

    if (Utils.checkFloat(this.maxSamples)) {
      this.nSamples = Math.floor(trainingSet.rows * this.maxSamples);
    } else if (Number.isInteger(this.maxSamples)) {
      if (this.maxSamples > trainingSet.rows) {
        throw new RangeError(
          `The maxSamples parameter should be less than ${trainingSet.rows}`,
        );
      } else {
        this.nSamples = this.maxSamples;
      }
    } else {
      throw new RangeError(
        `Cannot process the maxSamples parameter ${this.maxSamples}`,
      );
    }

    this.estimators = new Array(this.nEstimators);
    this.indexes = new Array(this.nEstimators);

    for (let i = 0; i < this.nEstimators; ++i) {
      let res = this.useSampleBagging
        ? Utils.examplesBaggingWithReplacement(
          trainingSet,
          trainingValues,
          currentSeed,
          this.nSamples,
        )
        : {
          X: trainingSet,
          y: trainingValues,
          seed: currentSeed,
          // Xoob: undefined,
          // yoob: [],
          // ioob: [],
        };
      let X = res.X;
      let y = res.y;
      currentSeed = res.seed;
      // let { Xoob, ioob } = res;

      res = Utils.featureBagging(
        X,
        this.nFeatures,
        this.replacement,
        currentSeed,
      );
      X = res.X;
      currentSeed = res.seed;

      this.indexes[i] = res.usedIndex;
      this.estimators[i] = new DTClassifier(this.treeOptions);
      this.estimators[i].train(X, y);
    }
  }

  predict(toPredict) {
    const predictionValues = this.predictionValues(toPredict);
    let predictions = new Array(predictionValues.rows);
    for (let i = 0; i < predictionValues.rows; ++i) {
      predictions[i] = this.selection(predictionValues.getRow(i));
    }

    return {
      predictions: predictions,
      predictionValues: predictionValues.to2DArray()
    };
  }

  toJSON() {
    return {
      ...super.toJSON(),
      nSamples: this.nSamples,
      nFeatures: this.nFeatures,
      maxSamples: this.maxSamples,
      maxFeatures: this.maxFeatures,
      selectionMethod: this.selectionMethod,
      name: 'RFClassifier',
    };
  }

  static load(model) {
    if (model.name !== 'RFClassifier') {
      throw new RangeError(`Invalid model: ${model.name}`);
    }

    return new CustomRandomForestClassifier(true, model);
  }
}