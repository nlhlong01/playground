/* Copyright 2016 Google Inc. All Rights Reserved.
Modifications Copyright 2020 Long Nguyen.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

/* eslint-disable @typescript-eslint/naming-convention */
import * as d3 from 'd3';
import {
  RandomForestClassifier as RFClassifier,
  RandomForestRegression as RFRegressor
} from 'ml-random-forest';
import 'seedrandom';
import 'material-design-lite';
import 'material-design-lite/dist/material.indigo-blue.min.css';
import '../styles.css';
import Worker from 'worker-loader!./train.worker';
import { HeatMap } from '../heatmap';
import {
  State,
  datasets,
  regDatasets,
  problems,
  getKeyFromValue,
  Problem
} from './state';
import { DataGenerator, Example2D, shuffle, isValid } from '../dataset';
import * as Utils from '../utils';

const NUM_SAMPLES_CLASSIFY = 400;
const NUM_SAMPLES_REGRESS = 800;
// Size of the heatmaps.
const SIDE_LENGTH = 300;
// # of points per direction.
const DENSITY = 50;
const NUM_VISIBLE_TREES = 16;

const state = State.deserializeState();
const xDomain: [number, number] = [-6, 6];
// Label values must be scaled before and after training since RF impl does not
// accepts negative values.
const inputScale = d3
  .scaleLinear()
  .domain([-1, 1])
  .range([0, 1]);
const outputScale = d3
  .scaleLinear()
  .domain([0, 1])
  .range([-1, 1]);
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
const treeHeatMaps: HeatMap[] = new Array(NUM_VISIBLE_TREES);
for (let i = 0; i < NUM_VISIBLE_TREES; i++) {
  const container = d3
    .select('.tree-heatmaps-container')
    .append('div')
    .attr('id', `tree-heatmap-${i}`)
    .attr('class' , 'mdl-cell mdl-cell--3-col');
  treeHeatMaps[i] = new HeatMap(
    SIDE_LENGTH / 6,
    DENSITY,
    xDomain,
    xDomain,
    container,
    { noSvg: true },
  );
}
const colorScale = d3
  .scaleLinear<string, number>()
  .domain([-1, 0, 1])
  .range(['#f59322', '#e8eaeb', '#0877bd'])
  .clamp(true);

let trainWorker: Worker;
let options;
let Method;
let rf: RFClassifier | RFRegressor;
let data: Example2D[];
let uploadedData: Example2D[];
let trainData: Example2D[];
let testData: Example2D[];
let metricList = [];
let getMetrics: (yPred: number[], yTrue: number[]) => any;
let trainMetrics;
let testMetrics;
let mainBoundary: number[][];
let treeBoundaries: number[][][];

/**
 * Prepares the UI on startup.
 */
function makeGUI() {
  d3.select('#start-button').on('click', () => {
    isLoading(true);

    trainWorker.terminate();
    trainWorker = new Worker();

    trainWorker.postMessage({
      options: options,
      trainingSet: trainData.map((d) => [d.x, d.y]),
      // Scale the input label to avoid negative class labels.
      labels: trainData.map((d) => inputScale(d.label)),
      isClassifier: isClassification()
    });

    trainWorker.onmessage = (msg: MessageEvent) => {
      const model = msg.data;
      rf = Method.load(model);

      // Final predictions of RF and predictions of decision trees.
      const predictions: number[] = new Array(data.length);
      const treePredictions: number[][] = rf
        .predictionValues(data.map((d) => [d.x, d.y]))
        .to2DArray();

      for (let i = 0; i < treePredictions.length; i++) {
        for (let j = 0; j < treePredictions[0].length; j++) {
          // Rescale predictions back to {-1, 1}
          treePredictions[i][j] = outputScale(treePredictions[i][j]);
          // In case of classifier, add the vote counts for each point.
          if (isClassification()) {
            const pred = treePredictions[i][j];
            data[i]['voteCounts'] = data[i]['voteCounts'] || [0, 0];
            // Increment the vote count of each class.
            if (pred === -1) data[i]['voteCounts'][0]++;
            else if (pred === 1) data[i]['voteCounts'][1]++;
            else throw new Error('Vote is invalid');
          }
        }
        predictions[i] = rf.selection(treePredictions[i]);
      }

      [trainData, testData] = splitTrainTest(data);
      const [trainPredictions, testPredictions] = splitTrainTest(predictions);
      const labels = data.map((d) => d.label);
      const [trainLabels, testLabels] = splitTrainTest(labels);

      trainMetrics = getMetrics(trainPredictions, trainLabels);
      testMetrics = getMetrics(testPredictions, testLabels);

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
    const newDataset = datasets[(this as HTMLElement).dataset.dataset];
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

  const regDataThumbnails = d3.selectAll('canvas[data-regDataset]');
  regDataThumbnails.on('click', function () {
    const newDataset = regDatasets[(this as HTMLCanvasElement)
      .dataset
      .regdataset];
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
      const element = this as HTMLInputElement;
      const file = element.files[0];

      if (file.type !== 'application/json') {
        element.value = '';
        alert('The uploaded file is not a JSON file.');
        return;
      }

      try {
        uploadedData = JSON.parse(await file.text());
        if (!isValid(uploadedData)) {
          element.value = '';
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
  for (let i = 0; i < NUM_VISIBLE_TREES; i++) {
    d3.select(`#tree-heatmap-${i} canvas`)
      .style('border', '2px solid black')
      .on('mouseenter', () => {
        mainHeatMap.updateBackground(treeBoundaries[i], state.discretize);
      })
      .on('mouseleave', () => {
        mainHeatMap.updateBackground(mainBoundary, state.discretize);
      });
  }

  /* Output Column */
  // Configure the number of trees
  const nTrees = d3.select('#nTrees').on('input', function () {
    const element = this as HTMLInputElement;
    state.nTrees = +element.value;
    d3.select("label[for='nTrees'] .value")
      .text(element.value);
    reset();
  });
  nTrees.property('value', state.nTrees);
  d3.select("label[for='nTrees'] .value")
    .text(state.nTrees);

  // Configure the max depth of each tree.
  const maxDepth = d3.select('#maxDepth').on('input', function () {
    const element = this as HTMLInputElement;
    state.maxDepth = +element.value;
    d3.select("label[for='maxDepth'] .value")
      .text(element.value);
    reset();
  });
  maxDepth.property('value', state.maxDepth);
  d3.select("label[for='maxDepth'] .value")
    .text(state.maxDepth);

  // Configure the number of samples to train each tree.
  const percSamples = d3.select('#percSamples').on('input', function () {
    const element = this as HTMLInputElement;
    state.percSamples = +element.value;
    d3.select("label[for='percSamples'] .value")
      .text(element.value);
    reset();
  });
  percSamples.property('value', state.percSamples);
  d3.select("label[for='percSamples'] .value")
    .text(state.percSamples);

  const problem = d3.select('#problem').on('change', function () {
    state.problem = problems[(this as HTMLSelectElement).value];
    generateData();
    drawDatasetThumbnails();
    reset();
  });
  problem.property('value', getKeyFromValue(problems, state.problem));

  const showTestData = d3.select('#show-test-data').on('change', function () {
    state.showTestData = (this as HTMLInputElement).checked;
    state.serialize();
    mainHeatMap.updateTestPoints(state.showTestData ? testData : []);
  });
  // Check/uncheck the checkbox according to the current state.
  showTestData.property('checked', state.showTestData);

  const discretize = d3.select('#discretize').on('change', function () {
    state.discretize = (this as HTMLInputElement).checked;
    state.serialize();
    mainHeatMap.updateBackground(mainBoundary, state.discretize);
    treeHeatMaps.forEach((map: HeatMap, idx: number) => {
      map.updateBackground(treeBoundaries[idx], state.discretize);
    });
  });
  // Check/uncheck the checbox according to the current state.
  discretize.property('checked', state.discretize);

  /* Data configurations */
  // Configure the ratio of training data to test data.
  const percTrain = d3.select('#percTrainData').on('input', function () {
    const element = this as HTMLInputElement;
    state.percTrainData = +element.value;
    d3.select("label[for='percTrainData'] .value")
      .text(element.value);
    reset();
  });
  percTrain.property('value', state.percTrainData);
  d3.select("label[for='percTrainData'] .value")
    .text(state.percTrainData);

  // Configure the level of noise.
  const noise = d3.select('#noise').on('input', function () {
    const element = this as HTMLInputElement;
    state.noise = +element.value;
    d3.select("label[for='noise'] .value")
      .text(element.value);
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
  const x = d3
    .scaleLinear()
    .domain([-1, 1])
    .range([0, 144]);
  const xAxis = d3
    .axisBottom(x)
    .tickValues([-1, 0, 1])
    .tickFormat(d3.format('d'));
  d3.select('#colormap g.core')
    .append('g')
    .attr('class', 'x axis')
    .attr('transform', 'translate(0,10)')
    .call(xAxis);
}

function updateDecisionBoundary(): void {
  let treeIdx: number;
  let i: number;
  let j: number;

  const xScale = d3
    .scaleLinear()
    .domain([0, DENSITY - 1])
    .range(xDomain);
  const yScale = d3
    .scaleLinear()
    .domain([DENSITY - 1, 0])
    .range(xDomain);

  treeBoundaries = new Array(NUM_VISIBLE_TREES);
  for (treeIdx = 0; treeIdx < NUM_VISIBLE_TREES; treeIdx++) {
    treeBoundaries[treeIdx] = new Array(DENSITY);
    for (i = 0; i < DENSITY; i++) {
      treeBoundaries[treeIdx][i] = new Array(DENSITY);
    }
  }

  mainBoundary = new Array(DENSITY);
  for (i = 0; i < DENSITY; i++) {
    mainBoundary[i] = new Array(DENSITY);
    for (j = 0; j < DENSITY; j++) {
      const x = xScale(i);
      const y = yScale(j);

      // Predict each point in the heatmap.
      const treePredictions: number[] = rf
        .predictionValues([[x, y]])
        .to2DArray()[0];
      const prediction: number = outputScale(
        rf.selection(treePredictions)
      );

      // Adds predictions to boundaries.
      mainBoundary[i][j] = prediction;
      for (treeIdx = 0; treeIdx < NUM_VISIBLE_TREES; treeIdx++) {
        treeBoundaries[treeIdx][i][j] = outputScale(
          treePredictions[treeIdx]
        );
      }
    }
  }
}

/**
 * Update all heat maps and metrics.
 * @param reset True when called in reset()
 */
function updateUI(reset = false) {
  if (!reset) updateDecisionBoundary();
  mainHeatMap.updateBackground(mainBoundary, state.discretize);
  for (let i = 0; i < treeHeatMaps.length; i++) {
    treeHeatMaps[i].updateBackground(treeBoundaries[i], state.discretize);
  }

  // Metrics table
  d3.selectAll('.metrics tbody tr').remove();
  metricList.forEach((metric) => {
    const row = d3.select('.metrics tbody').append('tr');
    // First row contains metric name
    row.append('td')
      .attr('class', 'mdl-data-table__cell--non-numeric')
      .text(metric);
    // Next 2 rows contain train and test metric values
    row.append('td')
      .text(trainMetrics ? trainMetrics[metric].toFixed(3) : '0.000');
    row.append('td')
      .text(testMetrics ? testMetrics[metric].toFixed(3) : '0.000');
  });
}

/**
 * Reset the app to initial state.
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
    maxFeatures: 1.0,
    treeOptions: { maxDepth: state.maxDepth },
    seed: undefined,
    useSampleBagging: true,
    replacement: false
  };

  if (isClassification()) {
    Method = RFClassifier;
    metricList = ['Accuracy', 'Precision', 'Recall'];
    getMetrics = Utils.getClfMetrics;
  } else {
    Method = RFRegressor;
    metricList = ['R2 Score'];
    getMetrics = Utils.getRegrMetrics;
  }

  trainMetrics = null;
  testMetrics = null;

  rf = null;
  d3.select("#start-button .value")
    .text(isClassification() ? 'classify' : 'regress');

  mainBoundary = new Array(DENSITY);
  treeBoundaries = new Array(NUM_VISIBLE_TREES);
  for (let treeIdx = 0; treeIdx < NUM_VISIBLE_TREES; treeIdx++) {
    treeBoundaries[treeIdx] = new Array(DENSITY);
    for (let x = 0; x < DENSITY; x++) {
      mainBoundary[x] = new Array(DENSITY);
      treeBoundaries[treeIdx][x] = new Array(DENSITY);
    }
  }

  uploadedData = uploadedData || [];
  data.forEach((d) => {
    delete d.voteCounts;
  });
  [trainData, testData] = splitTrainTest(data);

  state.serialize();
  updatePoints();
  updateUI(true);
}

function drawDatasetThumbnails() {
  const renderThumbnail = (canvas, dataGenerator: DataGenerator) => {
    const w = 100;
    const h = 100;
    canvas.setAttribute('width', w);
    canvas.setAttribute('height', h);
    const context = canvas.getContext('2d');
    const data = dataGenerator(200, 0);
    data.forEach((d: Example2D) => {
      context.fillStyle = colorScale(d.label);
      context.fillRect((w * (d.x + 6)) / 12, (h * (-d.y + 6)) / 12, 4, 4);
    });
    d3.select(canvas.parentNode).style('display', null);
  };
  d3.selectAll('.dataset').style('display', 'none');

  if (isClassification()) {
    for (const dataset in datasets) {
      const canvas: any = document.querySelector(
        `canvas[data-dataset=${dataset}]`
      );
      const dataGenerator = datasets[dataset];
      renderThumbnail(canvas, dataGenerator);
    }
  } else {
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

  const numSamples = isClassification()
    ? NUM_SAMPLES_CLASSIFY
    : NUM_SAMPLES_REGRESS;
  const generator = isClassification()
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
  d3.selectAll('.tree-heatmaps-container canvas')
    .style('opacity', loading ? 0.2 : 1);
  d3.selectAll('*')
    .style('cursor', loading ? 'progress' : null);
}

function isClassification() {
  return state.problem === Problem.CLASSIFICATION;
}

drawDatasetThumbnails();
makeGUI();
generateData(true);
reset(true);
