/* eslint-disable */
import * as d3 from 'd3';
import * as nn from './nn';
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
// import { AppendingLineChart } from './linechart';
import 'seedrandom';
import { RandomForestClassifier } from 'ml-random-forest';
import './styles.css';
import 'material-design-lite';
import 'material-design-lite/dist/material.indigo-blue.min.css';
import './analytics';

let mainWidth;

const NUM_SAMPLES_CLASSIFY = 500;
const NUM_SAMPLES_REGRESS = 1200;
// # of points per direction.
const DENSITY = 100;

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
let boundary: { [id: string]: number[][] } = {};

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

const colorScale = d3.scale
  .linear<string, number>()
  .domain([-1, 0, 1])
  .range(['#f59322', '#e8eaeb', '#0877bd'])
  .clamp(true);

let trainData: Example2D[] = [];
let testData: Example2D[] = [];
// let lossTrain = 0;
// let lossTest = 0;
// Line chart showing train & test errors
// const lineChart = new AppendingLineChart(
//   d3.select("#linechart"),
//   ["#777", "black"]
// );

function makeGUI() {
  /* Top controls */
  d3.select('#reset-button').on('click', () => {
    // reset();
    // userHasInteracted();
    d3.select('#play-pause-button');
  });

  d3.select('#play-pause-button').on('click', () => {
    // // Change the button's content.
    // userHasInteracted();
    // player.playOrPause();
  });

  /* Data column */
  d3.select('#data-regen-button').on('click', () => {
    generateData();
    // parametersChanged = true;
  });
  const dataThumbnails = d3.selectAll('canvas[data-dataset]');
  dataThumbnails.on('click', function () {
    const newDataset = datasets[this.dataset.dataset];

    if (newDataset === state.dataset) return;

    state.dataset = newDataset;
    dataThumbnails.classed('selected', false);
    d3.select(this).classed('selected', true);
    generateData();
    // parametersChanged = true;
    // reset();
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
    // parametersChanged = true;
    // reset();
  });

  const regDatasetKey = getKeyFromValue(regDatasets, state.regDataset);
  // Select the dataset according to the current state.
  d3.select(`canvas[data-regDataset=${regDatasetKey}]`)
    .classed("selected", true);

  /* Output Column */
  const showTestData = d3.select('#show-test-data').on('change', function () {
    state.showTestData = this.checked;
    state.serialize();
    // userHasInteracted();
    heatMap.updateTestPoints(state.showTestData ? testData : []);
  });

  // Check/uncheck the checkbox according to the current state.
  showTestData.property('checked', state.showTestData);

  const discretize = d3.select('#discretize').on('change', function () {
    // state.discretize = this.checked;
    // state.serialize();
    // userHasInteracted();
    // updateUI();
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
    // parametersChanged = true;
    // reset();
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
    // parametersChanged = true;
    // reset();
  });
  const currentMax = parseInt(noise.property('max'));
  if (state.noise > currentMax) {
    if (state.noise <= 80) noise.property('max', state.noise);
    else state.noise = 50;
  } else if (state.noise < 0) state.noise = 0;
  noise.property('value', state.noise);
  d3.select("label[for='noise'] .value")
    .text(state.noise);

  const batchSize = d3.select('#batchSize').on('input', function () {
    state.batchSize = this.value;
    d3.select("label[for='batchSize'] .value")
      .text(this.value);
    // parametersChanged = true;
    // reset();
  });
  batchSize.property('value', state.batchSize);
  d3.select("label[for='batchSize'] .value")
    .text(state.batchSize);

  const problem = d3.select('#problem').on('change', function () {
    state.problem = problems[this.value];
    generateData();
    drawDatasetThumbnails();
    // parametersChanged = true;
    // reset();
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

  // Listen for css-responsive changes and redraw the svg network.
  window.addEventListener('resize', () => {
    const newWidth = document
      .querySelector('#main-part')
      .getBoundingClientRect().width;
    if (newWidth !== mainWidth) {
      mainWidth = newWidth;
      // drawNetwork(network);
      updateUI(true);
    }
  });
}

/**
 * Given a neural network, it asks the network for the output (prediction)
 * of every node in the network using inputs sampled on a square grid.
 * It returns a map where each key is the node ID and the value is a square
 * matrix of the outputs of the network for each input in the grid respectively.
 */
function updateDecisionBoundary(network: nn.Node[][], firstTime: boolean) {
  if (firstTime) {
    boundary = {};
    // Go through all neurons and output, igoring the inputs.
    nn.forEachNode(network, true, (node) => {
      boundary[node.id] = new Array(DENSITY);
    });
    // Go through all predefined inputs.
    for (const nodeId in INPUTS) {
      boundary[nodeId] = new Array(DENSITY);
    }
  }
  const xScale = d3.scale
    .linear()
    .domain([0, DENSITY - 1])
    .range(xDomain);
  const yScale = d3.scale
    .linear()
    .domain([DENSITY - 1, 0])
    .range(xDomain);

  let i = 0;
  let j = 0;
  for (i = 0; i < DENSITY; i++) {
    if (firstTime) {
      nn.forEachNode(network, true, (node) => {
        boundary[node.id][i] = new Array(DENSITY);
      });
      // Go through all predefined inputs.
      for (const nodeId in INPUTS) {
        boundary[nodeId][i] = new Array(DENSITY);
      }
    }
    for (j = 0; j < DENSITY; j++) {
      // 1 for points inside, and 0 for points outside the circle.
      const x = xScale(i);
      const y = yScale(j);
      const input = constructInput(x, y);
      // Get the results from the network.
      nn.forwardProp(network, input);
      nn.forEachNode(network, true, (node) => {
        boundary[node.id][i][j] = node.output;
      });
      if (firstTime) {
        // Go through all predefined inputs.
        for (const nodeId in INPUTS) {
          boundary[nodeId][i][j] = INPUTS[nodeId].f(x, y);
        }
      }
    }
  }
}

// function getLoss(network: nn.Node[][], dataPoints: Example2D[]): number {
//   let loss = 0;
//   for (let i = 0; i < dataPoints.length; i++) {
//     const dataPoint = dataPoints[i];
//     const input = constructInput(dataPoint.x, dataPoint.y);
//     const output = nn.forwardProp(network, input);
//     loss += nn.Errors.SQUARE.error(output, dataPoint.label);
//   }
//   return loss / dataPoints.length;
// }

function updateUI(firstStep = false) {
  // // Get the decision boundary of the network.
  // updateDecisionBoundary(network, firstStep);
  // const selectedId = selectedNodeId != null
  //   ? selectedNodeId
  //   : nn.getOutputNode(network).id;
  // heatMap.updateBackground(boundary[selectedId], state.discretize);

  // function zeroPad(n: number): string {
  //   const pad = '000000';
  //   return (pad + n).slice(-pad.length);
  // }

  // function addCommas(s: string): string {
  //   return s.replace(/\B(?=(\d{3})+(?!\d))/g, ',');
  // }

  // function humanReadable(n: number): string {
  //   return n.toFixed(3);
  // }

  // // Update loss and iteration number.
  // d3.select('#loss-train').text(humanReadable(lossTrain));
  // d3.select('#loss-test').text(humanReadable(lossTest));
  // d3.select('#iter-number').text(addCommas(zeroPad(iter)));
  // lineChart.addDataPoint([lossTrain, lossTest]);
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

// Create selected inputs.
function constructInput(x: number, y: number): number[] {
  const input: number[] = [];
  for (const inputName in INPUTS) {
    if (state[inputName]) input.push(INPUTS[inputName].f(x, y));
  }
  return input;
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

// let firstInteraction = true;
// let parametersChanged = false;

// function userHasInteracted() {
//   if (!firstInteraction) {
//     return;
//   }
//   firstInteraction = false;
//   let page = 'index';
//   if (state.tutorial != null && state.tutorial !== '') {
//     page = `/v/tutorials/${state.tutorial}`;
//   }
//   ga('set', 'page', page);
//   ga('send', 'pageview', { sessionControl: 'start' });
// }

drawDatasetThumbnails();
makeGUI();
generateData(true);

const trainingSet = trainData.map((d) => [d.x, d.y]);
const labels = trainData.map((d) => d.label < 0 ? 0 : d.label); 
const testSet = testData.map((d) => [d.x, d.y]);

const options = {
  seed: 15,
  maxFeatures: 1,
  nEstimators: 100,
  treeOptions: {
    maxDepth: 10
  }
};

const classifier = new RandomForestClassifier(options);
classifier.train(trainingSet, labels);

boundary = {};
boundary.output = new Array(DENSITY);
const xScale = d3.scale
  .linear()
  .domain([0, DENSITY - 1])
  .range(xDomain);
const yScale = d3.scale
  .linear()
  .domain([DENSITY - 1, 0])
  .range(xDomain);

let i = 0;
let j = 0;
for (i = 0; i < DENSITY; i++) {
  boundary.output[i] = new Array(DENSITY);
  for (j = 0; j < DENSITY; j++) {
    const x = xScale(i);
    const y = yScale(j);
    const output = classifier.predict([[x, y]]);
    boundary.output[i][j] = output[0];
  }
}
heatMap.updateBackground(boundary.output, state.discretize);
