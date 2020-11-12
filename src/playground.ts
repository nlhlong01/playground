import * as d3 from 'd3';
import { HeatMap } from './heatmap';
import {
  State,
  datasets,
  regDatasets,
  problems,
  getKeyFromValue,
  Problem
} from './state';
import { DataGenerator, Example2D, shuffle, isValid } from './dataset';
import 'seedrandom';
import {
  CustomRandomForestClassifier as RFClassifier
} from './RandomForest/classifier';
import { ClassifierOptions } from './RandomForest/ml-random-forest';
import './styles.css';
import 'material-design-lite';
import 'material-design-lite/dist/material.indigo-blue.min.css';
import Worker from 'worker-loader!./train.worker';

const NUM_SAMPLES_CLASSIFY = 400;
const NUM_SAMPLES_REGRESS = 1200;
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
let classifier: RFClassifier;
let data: Example2D[] = [];
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

    trainWorker.terminate();
    trainWorker = new Worker();

    trainWorker.postMessage({
      options: options,
      trainingSet: trainData.map((d) => [d.x, d.y]),
      // RF implementation does not accept negative labels
      labels: trainData.map((d) => d.label === -1 ? 0 : 1)
    });

    trainWorker.onmessage = (msg: MessageEvent) => {
      classifier = RFClassifier.load(msg.data);

      // Final predictions of RF and predictions of estimators.
      let predictions;
      let predictionValues;
      ({ predictions, predictionValues } = classifier.predict(
        data.map((d: Example2D) => [d.x, d.y])
      ));
      // Rescale and discretize all predictions.
      const labelScale = d3.scale
        .quantize()
        .domain([0, 1])
        .range([-1, 1]);
      predictions = predictions.map(labelScale);
      predictionValues = predictionValues.map(
        (est: number[]) => est.map(labelScale)
      );
      const voteCounts: number[][] = predictionValues
        .map((est: number[]) => {
          const nNeg = est.filter((pred) => pred === -1).length;
          return [nNeg, est.length - nNeg];
        });
      data.forEach((d, i) => {
        d.voteCounts = voteCounts[i];
      });

      [trainData, testData] = splitTrainTest(data);
      const [trainPredictions, testPredictions] = splitTrainTest(predictions);
      const labels = data.map((d) => d.label);
      const [trainLabels, testLabels] = splitTrainTest(labels);

      lossTrain = getLoss(trainPredictions, trainLabels);
      lossTest = getLoss(testPredictions, testLabels);
      ({ accuracy, precision, recall } = score(testPredictions, testLabels));

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
    .classed("selected", true);

  const regDataThumbnails = d3.selectAll('canvas[data-regDataset]');
  regDataThumbnails.on('click', function () {
    const newDataset = regDatasets[this.dataset.regdataset];
    if (newDataset === state.regDataset) {
      return; // No-op.
    }
    state.regDataset = newDataset;
    regDataThumbnails.classed('selected', false);
    d3.select(this).classed('selected', true);
    generateData();
    reset();
  });

  const regDatasetKey = getKeyFromValue(regDatasets, state.regDataset);
  // Select the dataset according to the current state.
  d3.select(`canvas[data-regDataset=${regDatasetKey}]`)
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
      const {
        predictions,
        predictionValues
      } = classifier.predict([[x, y]]);

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
    replacement: false,
    selectionMethod: 'mean'
  };
  classifier = new RFClassifier(options);
  lossTest = 0;
  lossTrain = 0;
  accuracy = 0;
  precision = 0;
  recall = 0;
  mainBoundary = [];
  estimatorBoundaries = new Array(NUM_VISIBLE_EST).fill([]);
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
  function renderThumbnail(canvas, dataGenerator: DataGenerator) {
    const w = 100;
    const h = 100;
    canvas.setAttribute('width', w);
    canvas.setAttribute('height', h);
    const context = canvas.getContext('2d');
    const data = dataGenerator(200, 0);
    data.forEach((d: Example2D) => {
      context.fillStyle = colorScale(d.label);
      context.fillRect((w * (d.x + 6)) / 12, (h * (d.y + 6)) / 12, 4, 4);
    });
    d3.select(canvas.parentNode).style('display', null);
  }
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
  if (state.problem === Problem.REGRESSION) {
    for (const regDataset in regDatasets) {
      const canvas: any = document.querySelector(
        `canvas[data-regDataset=${regDataset}]`
      );
      const dataGenerator = regDatasets[regDataset];
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
  const numSamples =
    state.problem === Problem.REGRESSION
      ? NUM_SAMPLES_REGRESS
      : NUM_SAMPLES_CLASSIFY;
  const generator =
    state.problem === Problem.CLASSIFICATION
      ? state.dataset
      : state.regDataset;

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
