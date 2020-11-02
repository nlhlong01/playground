import * as d3 from 'd3';
import { HeatMap, reduceMatrix } from './heatmap';
import {
  State,
  datasets,
  regDatasets,
  problems,
  getKeyFromValue,
  Problem
} from './state';
import { DataGenerator, Example2D, shuffle } from './dataset';
import 'seedrandom';
// TODO: Bring this file to the same folder.
import {
  RandomForestClassifier as RFClassifier
} from '../../random-forest/src/RandomForestClassifier';
import { ClassifierOptions } from './ml-random-forest';
import './styles.css';
import 'material-design-lite';
import 'material-design-lite/dist/material.indigo-blue.min.css';
import Worker from 'worker-loader!./train.worker';

const NUM_SAMPLES_CLASSIFY = 400;
const NUM_SAMPLES_REGRESS = 1200;
// # of points per direction.
const DENSITY = 50;
const NUM_VISIBLE_TREES = 20;

const state = State.deserializeState();

let mainBoundary: number[][];
let estimatorBoundaries: number[][][];

// Plot the main heatmap.
const xDomain: [number, number] = [-6, 6];
const mainHeatMap = new HeatMap(
  300,
  DENSITY,
  xDomain,
  xDomain,
  d3.select('#main-heatmap'),
  { showAxes: true }
);

// Plot the tree heatmaps.
const estimatorHeatMaps: HeatMap[] = [];
for (let i = 0; i < NUM_VISIBLE_TREES; i++) {
  const estimatorHeatMapsContainer = d3
    .select('.estimator-heatmaps-container')
    .append('div')
    .attr('id', `estimator-heatmap-${i}`)
    .classed('mdl-cell mdl-cell--3-col', true);
  estimatorHeatMaps.push(new HeatMap(
    40,
    DENSITY,
    xDomain,
    xDomain,
    estimatorHeatMapsContainer,
    { showAxes: false, noSvg: true, pointer: true },
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
let trainData: Example2D[] = [];
let testData: Example2D[] = [];
let lossTrain: number;
let lossTest: number;

function makeGUI() {
  d3.select('#train-button').on('click', () => {
    isLoading(true);

    trainWorker.terminate();
    trainWorker = new Worker();

    const trainingSet = trainData.map((d) => [d.x, d.y]);
    // RF implementation does not accept negative labels
    const labels = trainData.map((d) => d.label === -1 ? 0 : d.label);
    trainWorker.postMessage({
      options: options,
      trainingSet: trainingSet,
      labels: labels
    });
    trainWorker.onmessage = (evt: MessageEvent) => {
      classifier = RFClassifier.load(evt.data);
      const predictionValues: number[][] = classifier
        .predict(data.map((d) => [d.x, d.y]))
        .predictionValues;
      const voteCounts = predictionValues.map((val: number[]) => {
        val = val.map((i) => i === 0 ? -1 : 1);
        return [
          val.filter((i) => i === -1).length,
          val.filter((i) => i === 1).length
        ];
      });

      data.forEach((d, i) => {
        d.voteCounts = voteCounts[i];
      });
      splitTrainTest();
      updatePoints();

      lossTrain = getLoss(trainData);
      lossTest = getLoss(testData);

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

    if (newDataset === state.dataset) return;

    state.dataset = newDataset;
    dataThumbnails.classed('selected', false);
    d3.select(this).classed('selected', true);
    generateData();
    reset();
  });

  const datasetKey = getKeyFromValue(datasets, state.dataset);
  // Select the dataset according to the current state.
  d3.select(`canvas[data-dataset=${datasetKey}]`).classed('selected', true);

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

  /* Main Column */
  // Heat maps of the esimators.
  estimatorHeatMaps.forEach((map, idx) => {
    d3.select(`#estimator-heatmap-${idx} canvas`)
      .style('border', '2px solid black')
      .on('mouseenter', () => {
        mainHeatMap.updateBackground(
          estimatorBoundaries[idx],
          state.discretize
        );
      })
      .on('mouseleave', () => {
        mainHeatMap.updateBackground(mainBoundary, state.discretize);
      });
  });

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
    generateData();
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
  let i: number;
  let j: number;
  let k: number;

  // 1 for points inside, and 0 for points outside the circle.
  const xScale = d3.scale
    .linear()
    .domain([0, DENSITY - 1])
    .range(xDomain);
  const yScale = d3.scale
    .linear()
    .domain([DENSITY - 1, 0])
    .range(xDomain);

  for (k = 0; k < NUM_VISIBLE_TREES; k++) {
    estimatorBoundaries[k] = new Array(DENSITY);
    for (i = 0; i < DENSITY; i++) {
      estimatorBoundaries[k][i] = new Array(DENSITY);
    }
  }

  for (i = 0; i < DENSITY; i++) {
    mainBoundary[i] = new Array(DENSITY);
    for (j = 0; j < DENSITY; j++) {
      const x = xScale(i);
      const y = yScale(j);
      const { predictions, predictionValues } = classifier.predict([[x, y]]);
      mainBoundary[i][j] = predictions[0];
      for (k = 0; k < NUM_VISIBLE_TREES; k++) {
        estimatorBoundaries[k][i][j] = predictionValues[0][k];
      }
    }
  }
}

function getLoss(dataPoints: Example2D[]): number {
  let loss = 0;
  for (let i = 0; i < dataPoints.length; i++) {
    const dataPoint = dataPoints[i];
    const x = dataPoint.x;
    const y = dataPoint.y;

    // TODO: Choose less confusing var names.
    let prediction = classifier.predict([[x, y]]).predictions[0];

    // TODO: Get rid of label scales.
    const scale = d3.scale
      .linear()
      .domain([0, 1])
      .range([-1, 1]);
    prediction = scale(prediction);

    loss += 0.5 * Math.pow(prediction - dataPoint.label, 2);
  }
  return loss / dataPoints.length;
}

function updateUI(reset = false) {
  if (!reset) updateDecisionBoundary();
  mainHeatMap.updateBackground(mainBoundary, state.discretize);
  estimatorHeatMaps.forEach((map: HeatMap, idx: number) => {
    map.updateBackground(estimatorBoundaries[idx], state.discretize);
  });

  function humanReadable(n: number): string {
    return n.toFixed(3);
  }

  // Update loss and iteration number.
  d3.select('#loss-train').text(humanReadable(lossTrain));
  d3.select('#loss-test').text(humanReadable(lossTest));
}

/* Reset the learning progress */
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
    selectionMethod: 'mean',
  };
  classifier = new RFClassifier(options);
  lossTest = 0;
  lossTrain = 0;
  mainBoundary = [];
  estimatorBoundaries = new Array(NUM_VISIBLE_TREES).fill([]);
  data.forEach((d) => {
    delete d.voteCounts;
  });
  splitTrainTest();
  updatePoints();

  state.serialize();

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

/* Generate data and display in the heatmap. */
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
  splitTrainTest();

  updatePoints();
}

function updatePoints() {
  mainHeatMap.updatePoints(trainData);
  mainHeatMap.updateTestPoints(state.showTestData ? testData : []);
}

function splitTrainTest() {
  const splitIndex = Math.floor((data.length * state.percTrainData) / 100);
  trainData = data.slice(0, splitIndex);
  testData = data.slice(splitIndex);
}

function isLoading(loading: boolean) {
  d3.selectAll('*')
    .style('cursor', loading ? 'progress' : null);
  d3.select('#main-heatmap canvas')
    .style('opacity', loading ? 0.2 : 1);
  d3.select('#main-heatmap svg')
    .style('opacity', loading ? 0.2 : 1);
  d3.selectAll('.estimator-heatmaps-container canvas')
    .style('opacity', loading ? 0.2 : 1);
}

drawDatasetThumbnails();
makeGUI();
generateData(true);
reset(true);
