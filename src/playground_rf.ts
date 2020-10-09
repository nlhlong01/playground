/* eslint-disable */
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
import { Example2D, shuffle } from './dataset';
import 'seedrandom';
import {
  RandomForestClassifier
} from '../../random-forest/src/RandomForestClassifier';
import './styles.css';
import 'material-design-lite';
import 'material-design-lite/dist/material.indigo-blue.min.css';

const NUM_SAMPLES_CLASSIFY = 500;
const NUM_SAMPLES_REGRESS = 1200;
// # of points per direction.
const DENSITY = 40;

interface InputFeature {
  f: (x: number, y: number) => number;
  label?: string;
}

// Input feature functions.
const INPUTS: { [name: string]: InputFeature } = {
  x: { f: (x, y) => x, label: 'X_1' },
  y: { f: (x, y) => y, label: 'X_2' },
  xSquared: { f: (x, y) => x * x, label: 'X_1^2' },
  ySquared: { f: (x, y) => y * y, label: 'X_2^2' },
  xTimesY: { f: (x, y) => x * y, label: 'X_1X_2' },
  sinX: { f: (x, y) => Math.sin(x), label: 'sin(X_1)' },
  sinY: { f: (x, y) => Math.sin(y), label: 'sin(X_2)' }
};

const state = State.deserializeState();

// TODO: Doublecheck this comment
// All points in the heatmap background
let boundary: number[][] = [];

// Plot the heatmap.
const xDomain: [number, number] = [-6, 6];
const heatMap = new HeatMap(
  300,
  DENSITY,
  xDomain,
  xDomain,
  d3.select('#heatmap'),
  { showAxes: true }
);

const treeHeatMaps = [];

for (let i = 0; i < 20; ++i) {
  const heatmapContainer = d3.select('.trees-container')
    .append('div')
    .classed('mdl-cell mdl-cell--3-col', true);
  
  const treeHeatMap = new HeatMap(
    40,
    DENSITY,
    xDomain,
    xDomain,
    heatmapContainer,
    { showAxes: false }
  );

  treeHeatMaps.push(treeHeatMap);
}

const colorScale = d3.scale
  .linear<string, number>()
  .domain([-1, 0, 1])
  .range(['#f59322', '#e8eaeb', '#0877bd'])
  .clamp(true);

let classifier: RandomForestClassifier = null;
let trainData: Example2D[] = [];
let testData: Example2D[] = [];
let lossTrain = 0;
let lossTest = 0;

function train() {
  // RF implementation does not accept negative label
  const labelScale = d3.scale
    .linear()
    .domain([-1, 1])
    .range([0, 1]);

  const trainingSet = trainData.map((d) => [d.x, d.y]);
  const labels = trainData.map((d) => labelScale(d.label));

  classifier = new RandomForestClassifier({
    nSamples: state.nSamples / NUM_SAMPLES_CLASSIFY,
    nEstimators: state.nTrees,
    maxFeatures: state.maxFeatures / 2,
    treeOptions: { maxDepth: state.maxDepth },
    selectionMethod: 'mean',
    useSampleBagging: true,
    replacement: false
  });
  classifier.train(trainingSet, labels);

  lossTrain = getLoss(trainData);
  lossTest = getLoss(testData);

  updateUI();
}

function makeGUI() {
  d3.select('#train-button').on('click', () => {
    userHasInteracted();
    train();
    updateUI();
    console.log(classifier.toJSON());
  });

  /* Data column */
  d3.select('#data-regen-button').on('click', () => {
    generateData();
    parametersChanged = true;
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
    parametersChanged = true;
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
    parametersChanged = true;
    reset();
  });

  const regDatasetKey = getKeyFromValue(regDatasets, state.regDataset);
  // Select the dataset according to the current state.
  d3.select(`canvas[data-regDataset=${regDatasetKey}]`)
    .classed('selected',true);

  /* Output Column */
  const showTestData = d3.select('#show-test-data').on('change', function () {
    state.showTestData = this.checked;
    state.serialize();
    userHasInteracted();
    heatMap.updateTestPoints(state.showTestData ? testData : []);
  });

  // Check/uncheck the checkbox according to the current state.
  showTestData.property('checked', state.showTestData);

  const discretize = d3.select('#discretize').on('change', function () {
    state.discretize = this.checked;
    state.serialize();
    userHasInteracted();
    updateUI();
  });

  // Check/uncheck the checbox according to the current state.
  discretize.property('checked', state.discretize);

  /* Network configurations */
  // Configure the ratio of training data to test data.
  const percTrain = d3.select('#percTrainData').on('input', function () {
    state.percTrainData = this.value;
    d3.select("label[for='percTrainData'] .value")
      .text(this.value);
    generateData();
    parametersChanged = true;
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
    parametersChanged = true;
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

  // Configure the number of trees
  const nTrees = d3.select('#nTrees').on('input', function () {
    state.nTrees = +this.value;
    d3.select("label[for='nTrees'] .value")
      .text(this.value);
    parametersChanged = true;
    reset();
  });
  nTrees.property('value', state.nTrees);
  d3.select("label[for='nTrees'] .value")
    .text(state.nTrees);

  // Configure the max depth of each tree
  const maxDepth = d3.select('#maxDepth').on('input', function () {
    state.maxDepth = +this.value;
    d3.select("label[for='maxDepth'] .value")
      .text(this.value);
    parametersChanged = true;
    reset();
  });
  maxDepth.property('value', state.maxDepth);
  d3.select("label[for='maxDepth'] .value")
    .text(state.maxDepth);

  // Configure the number of samples to train each tree
  const nSamples = d3.select('#nSamples').on('input', function () {
    state.nSamples = +this.value;
    d3.select("label[for='nSamples'] .value")
      .text(this.value);
    parametersChanged = true;
    reset();
  });
  nSamples.property('value', state.nSamples);
  d3.select("label[for='nSamples'] .value")
    .text(state.nSamples);

  const maxFeatures = d3.select('#maxFeatures').on('change', function () {
    state.maxFeatures = +this.value;
    state.serialize();
    userHasInteracted();
    parametersChanged = true;
    reset();
  });
  maxFeatures.property('value', state.maxFeatures);

  const problem = d3.select('#problem').on('change', function () {
    state.problem = problems[this.value];
    generateData();
    drawDatasetThumbnails();
    parametersChanged = true;
    reset();
  });
  problem.property('value', getKeyFromValue(problems, state.problem));

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

function updateHoverCard(coordinates?: [number, number]) {
  const hovercard = d3.select('#hovercard');

  d3.select('#svg').on('click', () => {
    hovercard.select('.value').style('display', 'none');
  });
  hovercard.style({
    left: `${coordinates[0] + 20}px`,
    top: `${coordinates[1]}px`,
    display: 'block'
  });
}

/**
 * Given a neural network, it asks the network for the output (prediction)
 * of every node in the network using inputs sampled on a square grid.
 * It returns a map where each key is the node ID and the value is a square
 * matrix of the outputs of the network for each input in the grid respectively.
 */
function updateDecisionBoundary(): void {
  if (!classifier) return;

  let i: number;
  let j: number;

  for (i = 0; i < DENSITY; i++) {
    boundary[i] = new Array(DENSITY);
    for (j = 0; j < DENSITY; j++) {
      // 1 for points inside, and 0 for points outside the circle.
      const xScale = d3.scale
        .linear()
        .domain([0, DENSITY - 1])
        .range(xDomain);
      const yScale = d3.scale
        .linear()
        .domain([DENSITY - 1, 0])
        .range(xDomain);

      boundary[i][j] = classifier.predict([[xScale(i), yScale(j)]])[0];
    }
  }
}

function getLoss(dataPoints: Example2D[]): number {
  let loss = 0;
  for (let i = 0; i < dataPoints.length; i++) {
    const dataPoint = dataPoints[i];
    let output = classifier.predict([[dataPoint.x, dataPoint.y]])[0];

    const scale = d3.scale
      .linear()
      .domain([0, 1])
      .range([-1, 1]);
    
    output = scale(output);

    loss += 0.5 * Math.pow(output - dataPoint.label, 2);
  }
  return loss / dataPoints.length;
}

function updateUI() {
  // Get the decision boundary of the network.
  updateDecisionBoundary();
  heatMap.updateBackground(boundary, state.discretize);
  treeHeatMaps.forEach((heatMap) => {
    heatMap.updateBackground(boundary, state.discretize);
  });
  // miniHeatmap.updateBackground(boundary, state.discretize);

  function humanReadable(n: number): string {
    return n.toFixed(3);
  }

  // Update loss and iteration number.
  d3.select('#loss-train').text(humanReadable(lossTrain));
  d3.select('#loss-test').text(humanReadable(lossTest));
}

// function constructInputIds(): string[] {
//   const result: string[] = [];
//   for (const inputName in INPUTS) {
//     if (state[inputName]) {
//       result.push(inputName);
//     }
//   }
//   return result;
// }

// // Create selected inputs.
// function constructInput(x: number, y: number): number[] {
//   const input: number[] = [];
//   for (const inputName in INPUTS) {
//     if (state[inputName]) input.push(INPUTS[inputName].f(x, y));
//   }
//   return input;
// }

/* Reset the learning progress */
function reset(onStartup = false) {
  lossTest = 0;
  lossTrain = 0;
  state.serialize();
  if (!onStartup) {
    userHasInteracted();
  }
  classifier = null;
  boundary = [];

  // lossTrain = getLoss(trainData);
  // lossTest = getLoss(testData);
  // heatMap.clearBackground();
  updateUI();
}

function drawDatasetThumbnails() {
  function renderThumbnail(canvas, dataGenerator) {
    const w = 100;
    const h = 100;
    canvas.setAttribute('width', w);
    canvas.setAttribute('height', h);
    const context = canvas.getContext('2d');
    const data = dataGenerator(200, 0);
    data.forEach((d) => {
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
    // userHasInteracted();
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
  const data = generator(numSamples, state.noise / 100);
  // Shuffle the data in-place.
  shuffle(data);
  // Split into train and test data.
  const splitIndex = Math.floor((data.length * state.percTrainData) / 100);

  trainData = data.slice(0, splitIndex);
  testData = data.slice(splitIndex);

  heatMap.updatePoints(trainData);
  heatMap.updateTestPoints(state.showTestData ? testData : []);
}

let firstInteraction = true;
let parametersChanged = false;

function userHasInteracted() {
  if (!firstInteraction) {
    return;
  }
  firstInteraction = false;
  let page = 'index';
  if (state.tutorial != null && state.tutorial !== '') {
    page = `/v/tutorials/${state.tutorial}`;
  }
}

drawDatasetThumbnails();
makeGUI();
generateData(true);
