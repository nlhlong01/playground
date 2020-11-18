/* eslint-disable @typescript-eslint/naming-convention */
import * as d3 from 'd3';
import { HeatMap } from './heatmap';
import {
  State,
  datasets,
  regDatasets1D,
  regDatasets2D,
  problems,
  getKeyFromValue,
  Problem
} from './state';
import { DataGenerator, Example2D, Point, shuffle, isValid } from './dataset';
import 'seedrandom';
import {
  RandomForestClassifier as RFClassifier,
  RandomForestRegression as RFRegressor
} from './RandomForest/index';
import { ClassifierOptions } from './RandomForest/ml-random-forest';
import './styles.css';
import 'material-design-lite';
import 'material-design-lite/dist/material.indigo-blue.min.css';
import Worker from 'worker-loader!./train.worker';

const NUM_SAMPLES_CLASSIFY = 400;
const NUM_SAMPLES_REGRESS_1D = 200;
const NUM_SAMPLES_REGRESS_2D = 800;
// Size of the heatmaps.
const SIDE_LENGTH = 300;
// # of points per direction.
const DENSITY = 50;
const NUM_VISIBLE_EST = 16;

const state = State.deserializeState();

let mainBoundary: number[][];
let estimatorBoundaries: number[][][];

const xDomain: [number, number] = [-6, 6];

// Plot the main heatmap.
const mainHeatMap = new HeatMap(
  SIDE_LENGTH,
  DENSITY,
  xDomain,
  xDomain,
  d3.select('#main-heatmap'),
  { showAxes: true }
);

// Plot the tree heatmaps.
const estimatorHeatMaps: HeatMap[] = [];
for (let i = 0; i < NUM_VISIBLE_EST; i++) {
  const estimatorHeatMapsContainer = d3
    .select('.estimator-heatmaps-container')
    .append('div')
    .attr('id', `estimator-heatmap-${i}`)
    .attr('class' , 'mdl-cell mdl-cell--3-col');
  estimatorHeatMaps.push(new HeatMap(
    SIDE_LENGTH / 6,
    DENSITY,
    xDomain,
    xDomain,
    estimatorHeatMapsContainer,
    { noSvg: true },
  ));
}

const colorScale = d3.scale
  .linear<string, number>()
  .domain([-1, 0, 1])
  .range(['#f59322', '#e8eaeb', '#0877bd'])
  .clamp(true);

let trainWorker: Worker;
let options: ClassifierOptions;
let Method;
let rf: RFClassifier | RFRegressor;
let isClassifier: boolean;
let data: Example2D[] | Point[] = [];
let uploadedData: Example2D[] = [];
let trainData: Example2D[] = [];
let testData: Example2D[] = [];
let lossTrain: number;
let lossTest: number;
let accuracy: number;
let precision: number;
let recall: number;

/**
 * Prepares the UI on startup.
 */
function makeGUI() {
  d3.select('#train-button').on('click', () => {
    isLoading(true);
    const labelScale = d3.scale
      .linear()
      .domain([-1, 1])
      .range([0, 1]);

    trainWorker.terminate();
    trainWorker = new Worker();

    trainWorker.postMessage({
      options: options,
      trainingSet: trainData.map((d) => [d.x, d.y]),
      labels: trainData.map((d) => labelScale(d.label)),
      isClassifier: isClassifier
    });

    trainWorker.onmessage = (msg: MessageEvent) => {
      rf = Method.load(msg.data);

      // // Final predictions of RF and predictions of estimators.
      // let predictions;
      // let predictionValues;
      // ({ predictions, predictionValues } = rf.predict(
      //   data.map((d) => [d.x, d.y])
      // ));
      // // Rescale and discretize all predictions.
      // const labelRescale = d3.scale
      //   .quantize()
      //   .domain([0, 1])
      //   .range([-1, 1]);
      // predictions = predictions.map(labelRescale);
      // predictionValues = predictionValues.map(
      //   (est: number[]) => est.map(labelRescale)
      // );

      // if (isClassifier) {
      //   // Count the votes for each class.
      //   const voteCounts: number[][] = predictionValues
      //     .map((est: number[]) => {
      //       // # trees voting for class -1.
      //       const nNeg = est.filter((pred) => pred === -1).length;
      //       return [nNeg, est.length - nNeg];
      //     });
      //   data.forEach((d, i) => {
      //     d.voteCounts = voteCounts[i];
      //   });

      //   [trainData, testData] = splitTrainTest(data);
      //   const [trainPredictions, testPredictions] = splitTrainTest(predictions);
      //   const labels = data.map((d) => d.label);
      //   const [trainLabels, testLabels] = splitTrainTest(labels);

      //   lossTrain = getLoss(trainPredictions, trainLabels);
      //   lossTest = getLoss(testPredictions, testLabels);
      //   ({ accuracy, precision, recall } = score(testPredictions, testLabels));
      // }

      updatePoints();
      updateUI();
      isLoading(false);
    };
  });

  /* Data column */
  d3.select('#data-regen-button').on('click', () => {
    generateData();
    reset();
  });
  const dataThumbnails = d3.selectAll('canvas[data-dataset]');
  dataThumbnails.on('click', function () {
    const newDataset = datasets[this.dataset.dataset];
    if (newDataset === state.dataset) {
      return; // No-op.
    }
    state.dataset = newDataset;
    dataThumbnails.classed('selected', false);
    d3.select(this).classed('selected', true);
    generateData();
    reset();
  });
  const datasetKey = getKeyFromValue(datasets, state.dataset);
  // Select the dataset according to the current state.
  d3.select(`canvas[data-dataset=${datasetKey}]`)
    .classed('selected', true);

  const regData1DThumbnails = d3.selectAll('canvas[data-regDataset1D]');
  regData1DThumbnails.on('click', function () {
    const newDataset = regDatasets1D[this.dataset.regdataset1d];
    if (newDataset === state.regDataset1D) {
      return; // No-op.
    }
    state.regDataset1D = newDataset;
    regData1DThumbnails.classed('selected', false);
    d3.select(this).classed('selected', true);
    generateData();
    reset();
  });
  const regDataset1DKey = getKeyFromValue(regDatasets1D, state.regDataset1D);
  // Select the dataset according to the current state.
  d3.select(`canvas[data-regDataset1D=${regDataset1DKey}]`)
    .classed('selected', true);

  const regData2DThumbnails = d3.selectAll('canvas[data-regDataset2D]');
  regData2DThumbnails.on('click', function () {
    const newDataset = regDatasets2D[this.dataset.regdataset2d];
    if (newDataset === state.regDataset2D) {
      return; // No-op.
    }
    state.regDataset2D = newDataset;
    regData2DThumbnails.classed('selected', false);
    d3.select(this).classed('selected', true);
    generateData();
    reset();
  });
  const regDataset2DKey = getKeyFromValue(regDatasets2D, state.regDataset2D);
  // Select the dataset according to the current state.
  d3.select(`canvas[data-regDataset2D=${regDataset2DKey}]`)
    .classed('selected', true);

  d3.select('#file-input')
    .on('input', async function() {
      const file = this.files[0];
      if (file.type !== 'application/json') {
        this.value = '';
        alert('The uploaded file is not a JSON file.');
        return;
      }
      try {
        uploadedData = JSON.parse(await file.text());
        if (!isValid(uploadedData)) {
          this.value = '';
          uploadedData = [];
          throw Error('The uploaded file does not have a valid format');
        }
        d3.select('#file-name').text(file.name);
      } catch (err) {
        alert('The uploaded file does not have a valid format.');
      }
    });

  d3.select('#file-select')
    .on('click', () => {
      if (uploadedData.length === 0) return;
      data = uploadedData;
      [trainData, testData] = splitTrainTest(data);
      updatePoints();
      reset();
    });

  /* Main Column */
  // Heat maps of the esimators.
  for (let i = 0; i < NUM_VISIBLE_EST; i++) {
    d3.select(`#estimator-heatmap-${i} canvas`)
      .style('border', '2px solid black')
      .on('mouseenter', () => {
        mainHeatMap.updateBackground(
          estimatorBoundaries[i],
          state.discretize
        );
      })
      .on('mouseleave', () => {
        mainHeatMap.updateBackground(mainBoundary, state.discretize);
      });
  }

  /* Output Column */
  // Configure the number of trees
  const nTrees = d3.select('#nTrees').on('input', function () {
    state.nTrees = +this.value;
    d3.select("label[for='nTrees'] .value")
      .text(this.value);
    reset();
  });
  nTrees.property('value', state.nTrees);
  d3.select("label[for='nTrees'] .value")
    .text(state.nTrees);

  // Configure the max depth of each tree.
  const maxDepth = d3.select('#maxDepth').on('input', function () {
    state.maxDepth = +this.value;
    d3.select("label[for='maxDepth'] .value")
      .text(this.value);
    reset();
  });
  maxDepth.property('value', state.maxDepth);
  d3.select("label[for='maxDepth'] .value")
    .text(state.maxDepth);

  // Configure the number of samples to train each tree.
  const percSamples = d3.select('#percSamples').on('input', function () {
    state.percSamples = +this.value;
    d3.select("label[for='percSamples'] .value")
      .text(this.value);
    reset();
  });
  percSamples.property('value', state.percSamples);
  d3.select("label[for='percSamples'] .value")
    .text(state.percSamples);

  // Configure the maximum bagged feature.
  const maxFeatures = d3.select('#maxFeatures').on('change', function () {
    state.maxFeatures = +this.value;
    state.serialize();
    reset();
  });
  maxFeatures.property('value', state.maxFeatures);

  const problem = d3.select('#problem').on('change', function () {
    state.problem = problems[this.value];
    generateData();
    drawDatasetThumbnails();
    reset();
  });
  problem.property('value', getKeyFromValue(problems, state.problem));
  const showTestData = d3.select('#show-test-data').on('change', function () {
    state.showTestData = this.checked;
    state.serialize();
    mainHeatMap.updateTestPoints(state.showTestData ? testData : []);
  });

  // Check/uncheck the checkbox according to the current state.
  showTestData.property('checked', state.showTestData);

  const discretize = d3.select('#discretize').on('change', function () {
    state.discretize = this.checked;
    state.serialize();
    mainHeatMap.updateBackground(mainBoundary, state.discretize);
    estimatorHeatMaps.forEach((map: HeatMap, idx: number) => {
      map.updateBackground(estimatorBoundaries[idx], state.discretize);
    });
  });

  // Check/uncheck the checbox according to the current state.
  discretize.property('checked', state.discretize);

  /* Data configurations */
  // Configure the ratio of training data to test data.
  const percTrain = d3.select('#percTrainData').on('input', function () {
    state.percTrainData = this.value;
    d3.select("label[for='percTrainData'] .value")
      .text(this.value);
    // generateData();
    reset();
  });
  percTrain.property('value', state.percTrainData);
  d3.select("label[for='percTrainData'] .value")
    .text(state.percTrainData);

  // Configure the level of noise.
  const noise = d3.select('#noise').on('input', function () {
    state.noise = this.value;
    d3.select("label[for='noise'] .value")
      .text(this.value);
    generateData();
    reset();
  });
  const currentMax = parseInt(noise.property('max'));
  if (state.noise > currentMax) {
    if (state.noise <= 80) noise.property('max', state.noise);
    else state.noise = 50;
  } else if (state.noise < 0) state.noise = 0;
  noise.property('value', state.noise);
  d3.select("label[for='noise'] .value")
    .text(state.noise);

  /* Color map */
  // Add scale to the gradient color map.
  const x = d3.scale
    .linear()
    .domain([-1, 1])
    .range([0, 144]);
  const xAxis = d3.svg
    .axis()
    .scale(x)
    .orient('bottom')
    .tickValues([-1, 0, 1])
    .tickFormat(d3.format('d'));
  d3.select('#colormap g.core')
    .append('g')
    .attr('class', 'x axis')
    .attr('transform', 'translate(0,10)')
    .call(xAxis);
}

function updateDecisionBoundary(): void {
  let estIdx: number;
  let i: number;
  let j: number;

  // 1 for points inside, and 0 for points outside the circle.
  const xScale = d3.scale
    .linear()
    .domain([0, DENSITY - 1])
    .range(xDomain);
  const yScale = d3.scale
    .linear()
    .domain([DENSITY - 1, 0])
    .range(xDomain);

  for (estIdx = 0; estIdx < NUM_VISIBLE_EST; estIdx++) {
    estimatorBoundaries[estIdx] = new Array(DENSITY);
    for (i = 0; i < DENSITY; i++) {
      estimatorBoundaries[estIdx][i] = new Array(DENSITY);
    }
  }

  for (i = 0; i < DENSITY; i++) {
    mainBoundary[i] = new Array(DENSITY);
    for (j = 0; j < DENSITY; j++) {
      const x = xScale(i);
      const y = yScale(j);
      // Predict each point in the heatmap.
      const { predictions, predictionValues } = rf.predict([[x, y]]);

      // Update prediction of that point in all boundaries.
      mainBoundary[i][j] = predictions[0];
      for (estIdx = 0; estIdx < NUM_VISIBLE_EST; estIdx++) {
        estimatorBoundaries[estIdx][i][j] = predictionValues[0][estIdx];
      }
    }
  }
}

/**
 * Computes the average mean square error of the predicted values.
 * @param predClass Correct target value.
 * @param trueClass Estimated target value.
 */
function getLoss(predClass: number[], trueClass: number[]): number {
  if (predClass.length !== trueClass.length) {
    throw Error('Length of predictions must equal length of labels');
  }
  let loss = 0;
  for (let i = 0; i < predClass.length; i++) {
    loss += 0.5 * Math.pow(predClass[i] - trueClass[i], 2);
  }
  return loss / predClass.length;
}

/**
 * Compute classification metrics.
 * @param predClass Correct target value.
 * @param trueClass Estimated target value.
 */
function score(predClass: number[], trueClass: number[]) {
  if (predClass.length !== trueClass.length) {
    throw Error('Length of predictions must equal length of labels');
  }

  // 4 items of a confusion matrix.
  let tp = 0;
  let tn = 0;
  let fp = 0;
  let fn = 0;

  for (let i = 0; i < predClass.length; i++) {
    const pred = predClass[i];
    const label = trueClass[i];

    if (pred === -1 && label === -1) tn++;
    else if (pred === -1 && label === 1) fn++;
    else if (pred === 1 && label === -1) fp++;
    else if (pred === 1 && label === 1) tp++;
    else throw Error('Predicted or true class value is invalid');
  }

  return {
    accuracy: (tp + tn) / (tp + tn + fp + fn),
    precision: tp / (tp + fp),
    recall: tp / (tp + fn)
  };
}

/**
 * Update all heat maps and metrics.
 * @param reset True when called in reset()
 */
function updateUI(reset = false) {
  if (!reset) updateDecisionBoundary();
  mainHeatMap.updateBackground(mainBoundary, state.discretize);
  estimatorHeatMaps.forEach((map: HeatMap, idx: number) => {
    map.updateBackground(estimatorBoundaries[idx], state.discretize);
  });

  const updateMetric = (selector: string, value: number): void => {
    d3.select(selector).text(value.toFixed(3));
  };

  updateMetric('#loss-train', lossTrain);
  updateMetric('#loss-test', lossTest);
  updateMetric('#accuracy', accuracy);
  updateMetric('#precision', precision);
  updateMetric('#recall', recall);
}

/**
 * @param reset True when called on startup.
 */
function reset(onStartup = false) {
  if (!onStartup) {
    trainWorker.terminate();
    isLoading(false);
  }
  trainWorker = new Worker();
  options = {
    nEstimators: state.nTrees,
    maxSamples: state.percSamples / 100,
    maxFeatures: state.maxFeatures / 2,
    treeOptions: { maxDepth: state.maxDepth },
    useSampleBagging: true,
    replacement: false
  };
  isClassifier = state.problem === Problem.CLASSIFICATION;
  Method = isClassifier ? RFClassifier : RFRegressor;
  rf = null;
  lossTest = 0;
  lossTrain = 0;
  accuracy = 0;
  precision = 0;
  recall = 0;
  mainBoundary = [];
  estimatorBoundaries = new Array(NUM_VISIBLE_EST).fill([]);
  // TODO: Clean this
  // mainBoundary = new Array(DENSITY);
  // estimatorBoundaries = new Array(NUM_VISIBLE_EST);
  // for (let estIdx = 0; estIdx < NUM_VISIBLE_EST; estIdx++) {
  //   estimatorBoundaries[estIdx] = new Array(DENSITY);
  //   for (let i = 0; i < DENSITY; i++) {
  //     estimatorBoundaries[estIdx][i] = new Array(DENSITY);
  //     mainBoundary[i] = new Array(DENSITY);
  //   }
  // }
  data.forEach((d) => {
    delete d.voteCounts;
  });
  [trainData, testData] = splitTrainTest(data);

  state.serialize();
  updatePoints();
  updateUI(true);
}

function drawDatasetThumbnails() {
  const renderThumbnail = (
    canvas,
    dataGenerator: DataGenerator,
    noise = 0
  ) => {
    const w = 100;
    const h = 100;
    canvas.setAttribute('width', w);
    canvas.setAttribute('height', h);
    const context = canvas.getContext('2d');
    const data = dataGenerator(200, noise);
    data.forEach((d: Example2D) => {
      context.fillStyle = colorScale(d.label);
      context.fillRect((w * (d.x + 6)) / 12, (h * (d.y + 6)) / 12, 4, 4);
    });
    d3.select(canvas.parentNode).style('display', null);
  };
  d3.selectAll('.dataset').style('display', 'none');

  if (state.problem === Problem.CLASSIFICATION) {
    for (const dataset in datasets) {
      const canvas: any = document.querySelector(
        `canvas[data-dataset=${dataset}]`
      );
      const dataGenerator = datasets[dataset];
      renderThumbnail(canvas, dataGenerator);
    }
  }
  if (state.problem === Problem.REGRESSION_1D) {
    for (const dataset in regDatasets1D) {
      const canvas: any = document.querySelector(
        `canvas[data-regDataset1D=${dataset}]`
      );
      const dataGenerator = regDatasets1D[dataset];
      renderThumbnail(canvas, dataGenerator, 0.2);
    }
  }
  if (state.problem === Problem.REGRESSION_2D) {
    for (const dataset in regDatasets2D) {
      const canvas: any = document.querySelector(
        `canvas[data-regDataset2D=${dataset}]`
      );
      const dataGenerator = regDatasets2D[dataset];
      renderThumbnail(canvas, dataGenerator);
    }
  }
}

function generateData(firstTime = false) {
  if (!firstTime) {
    // Change the seed.
    state.seed = Math.random().toFixed(5);
    state.serialize();
  }

  Math.seedrandom(state.seed);
  let numSamples: number;
  let generator: DataGenerator;

  if (state.problem === Problem.CLASSIFICATION) {
    numSamples = NUM_SAMPLES_CLASSIFY;
    generator = state.dataset;
  } else if (state.problem === Problem.REGRESSION_1D) {
    numSamples = NUM_SAMPLES_REGRESS_1D;
    generator = state.regDataset1D;
  } else {
    numSamples = NUM_SAMPLES_REGRESS_2D;
    generator = state.regDataset2D;
  }

  data = generator(numSamples, state.noise / 100);
  // Shuffle the data in-place.
  shuffle(data);
  [trainData, testData] = splitTrainTest(data);
  updatePoints();
}

/**
 * Split the input array into 2 chunks by an index determined by the selected
 * percentage of train data.
 * @param arr
 */
function splitTrainTest(arr: any[]): any[][] {
  const splitIndex = Math.floor((arr.length * state.percTrainData) / 100);
  return [arr.slice(0, splitIndex) , arr.slice(splitIndex)];
}

/**
 * Redraw data points on the main heat map.
 */
function updatePoints() {
  mainHeatMap.updatePoints(trainData);
  mainHeatMap.updateTestPoints(state.showTestData ? testData : []);
}

/**
 * Shows busy indicators in the UI as something is running in the background.
 * They include making all heatmaps opaque and showing a progress indicator next
 * to the cursor.
 * @param {boolean} loading True if something is running in the background
 */
function isLoading(loading: boolean) {
  d3.select('#main-heatmap canvas')
    .style('opacity', loading ? 0.2 : 1);
  d3.select('#main-heatmap svg')
    .style('opacity', loading ? 0.2 : 1);
  d3.selectAll('.estimator-heatmaps-container canvas')
    .style('opacity', loading ? 0.2 : 1);
  d3.selectAll('*')
    .style('cursor', loading ? 'progress' : null);
}

drawDatasetThumbnails();
makeGUI();
generateData(true);
reset(true);
