/* Copyright 2016 Google Inc. All Rights Reserved.
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

import * as d3 from 'd3';
import { LineChart } from './linechart';
import {
  State,
  datasets,
  getKeyFromValue,
} from './state';
import { DataGenerator, Point, shuffle, isValid } from './dataset';
import 'seedrandom';
import {
  RandomForestRegression as RFRegressor
} from 'ml-random-forest';
import '../styles.css';
import 'material-design-lite';
import 'material-design-lite/dist/material.indigo-blue.min.css';

const NUM_SAMPLES = 200;
// Size of the linecharts.
const SIDE_LENGTH = 300;
// # of points along the x-axis.
const DENSITY = 100;
const NUM_VISIBLE_TREES = 16;

const state = State.deserializeState();

const xDomain: [number, number] = [-6, 6];

// Plot the main linechart.
const mainLineChart = new LineChart(
  SIDE_LENGTH,
  // DENSITY,
  xDomain,
  xDomain,
  d3.select('#main-linechart'),
  { showAxes: true }
);

// Plot the tree linecharts.
const treeLineCharts: LineChart[] = new Array(NUM_VISIBLE_TREES);
for (let i = 0; i < NUM_VISIBLE_TREES; i++) {
  const container = d3
    .select('.tree-linecharts-container')
    .append('div')
    .attr('id', `tree-linechart-${i}`)
    .attr('class' , 'mdl-cell mdl-cell--3-col');
  treeLineCharts[i] = new LineChart(
    SIDE_LENGTH / 6,
    xDomain,
    xDomain,
    container,
    { noPoint: true },
  );
}

let options;
let regressor: RFRegressor;
let curve: Point[];
let treeCurves: Point[][];
let data: Point[];
let uploadedData: Point[];

/**
 * Prepares the UI on startup.
 */
function makeGUI() {
  d3.select('#start-button').on('click', () => {
    let pointIdx: number;
    let treeIdx = 0;
    const xScale = d3.scale
      .linear()
      .domain([0, DENSITY - 1])
      .range(xDomain);

    regressor = new RFRegressor(options);
    regressor.train(data.map((d) => [d.x]), data.map((d) => d.y));

    treeCurves = new Array(NUM_VISIBLE_TREES);
    for (treeIdx = 0; treeIdx < treeCurves.length; treeIdx++) {
      treeCurves[treeIdx] = new Array(DENSITY);
    }

    curve = new Array(DENSITY);
    for (pointIdx = 0; pointIdx < curve.length; pointIdx++) {
      const x = xScale(pointIdx);
      const treePredictions: number[] = regressor
        .predictionValues([[x]])
        .to2DArray()[0];
      for (treeIdx = 0; treeIdx < treeCurves.length; treeIdx++) {
        const y = treePredictions[treeIdx];
        treeCurves[treeIdx][pointIdx] = { x, y };
      }
      // Get the final prediction based on estimators' predictions.
      const prediction: number = regressor.selection(treePredictions);
      const y = prediction;
      curve[pointIdx] = { x, y };
    }

    d3.selectAll('path.plot').remove();
    updateUI();
  });

  /* Data column */
  d3.select('#data-regen-button').on('click', () => {
    generateData();
    reset();
  });

  const dataThumbnails = d3.selectAll('canvas[data-dataset]');
  dataThumbnails.on('click', function () {
    const newDataset = datasets[this.dataset.dataset];
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
      mainLineChart.updatePoints(data);
      reset();
    });

  /* Main Column */
  // Heat maps of the esimators.
  for (let i = 0; i < NUM_VISIBLE_TREES; i++) {
    d3.select(`#tree-linechart-${i} div`)
      .style('border', '2px solid black')
      .on('mouseenter', () => {
        mainLineChart.updatePlot(treeCurves[i]);
      })
      .on('mouseleave', () => {
        mainLineChart.updatePlot(curve);
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

  /* Data configurations */
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
}

/**
 * Update all heat maps and metrics.
 * @param reset True when called in reset()
 */
function updateUI() {
  mainLineChart.updatePlot(curve);
  for (let i = 0; i < treeLineCharts.length; i++) {
    treeLineCharts[i].updatePlot(treeCurves[i]);
  }

  const updateMetric = (selector: string, value: number): void => {
    d3.select(selector).text(value.toFixed(3));
  };
}

/**
 * @param reset True when called on startup.
 */
function reset() {
  options = {
    nEstimators: state.nTrees,
    maxSamples: state.percSamples / 100,
    maxFeatures: 1.0,
    treeOptions: { maxDepth: state.maxDepth },
    seed: undefined,
    useSampleBagging: true,
    replacement: false
  };
  regressor = null;

  curve = new Array(NUM_SAMPLES);
  treeCurves = new Array(NUM_VISIBLE_TREES);
  for (let i = 0; i < treeCurves.length; i++) {
    treeCurves[i] = new Array(DENSITY);
  }

  state.serialize();
  mainLineChart.updatePoints(data);
  updateUI();
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
    data.forEach((d: Point) => {
      context.fillStyle = 'slateblue';
      context.fillRect((w * (d.x + 6)) / 12, (h * (-d.y + 6)) / 12, 4, 4);
    });
    d3.select(canvas.parentNode).style('display', null);
  };
  d3.selectAll('.dataset').style('display', 'none');

  for (const dataset in datasets) {
    const canvas: any = document.querySelector(
      `canvas[data-dataset=${dataset}]`
    );
    const dataGenerator = datasets[dataset];
    renderThumbnail(canvas, dataGenerator, 0.5);
  }
}

function generateData(firstTime = false) {
  if (!firstTime) {
    // Change the seed.
    state.seed = Math.random().toFixed(5);
    state.serialize();
  }

  Math.seedrandom(state.seed);
  const numSamples = NUM_SAMPLES;
  const generator = state.dataset;

  data = generator(numSamples, state.noise / 100);
  // Shuffle the data in-place.
  shuffle(data);
  mainLineChart.updatePoints(data);
}

drawDatasetThumbnails();
makeGUI();
generateData(true);
reset();
