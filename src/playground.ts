/* eslint-disable @typescript-eslint/no-unused-vars */
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
import * as nn from './nn';
import * as rf from './randomforest';
import { HeatMap, reduceMatrix } from './heatmap';
import {
  State,
  datasets,
  regDatasets,
  activations,
  problems,
  regularizations,
  getKeyFromValue,
  Problem
} from './state';
import { Example2D, shuffle } from './dataset';
import { AppendingLineChart } from './linechart';
import 'seedrandom';

let mainWidth;

// Scrolls for more info
d3.select('.more button').on('click', () => {
  const position = 800;

  d3.transition()
    .duration(1000)
    .tween("scroll", scrollTween(position));
});

function scrollTween(offset) {
  return function () {
    const i = d3.interpolateNumber(
      window.pageYOffset || document.documentElement.scrollTop,
      offset
    );
    return function (t) {
      scrollTo(0, i(t));
    };
  };
}

const RECT_SIZE = 30;
const BIAS_SIZE = 10;
const NUM_SAMPLES_CLASSIFY = 500;
const NUM_SAMPLES_REGRESS = 1200;
const DENSITY = 100;

enum HoverType {
  BIAS,
  WEIGHT
}

interface InputFeature {
  f: (x: number, y: number) => number;
  label?: string;
}

// Input function types
const INPUTS: { [name: string]: InputFeature } = {
  x: { f: (x, y) => x, label: 'X_1' },
  y: { f: (x, y) => y, label: 'X_2' },
  xSquared: { f: (x, y) => x * x, label: 'X_1^2' },
  ySquared: { f: (x, y) => y * y, label: 'X_2^2' },
  xTimesY: { f: (x, y) => x * y, label: 'X_1X_2' },
  sinX: { f: (x, y) => Math.sin(x), label: 'sin(X_1)' },
  sinY: { f: (x, y) => Math.sin(y), label: 'sin(X_2)' }
};

const HIDABLE_CONTROLS = [
  ['Show test data', 'showTestData'],
  ['Discretize output', 'discretize'],
  ['Play button', 'playButton'],
  ['Step button', 'stepButton'],
  ['Reset button', 'resetButton'],
  ['Learning rate', 'learningRate'],
  ['Activation', 'activation'],
  ['Regularization', 'regularization'],
  ['Regularization rate', 'regularizationRate'],
  ['Problem type', 'problem'],
  ['Which dataset', 'dataset'],
  ['Ratio train data', 'percTrainData'],
  ['Noise level', 'noise'],
  ['Batch size', 'batchSize'],
  ['# of hidden layers', 'numHiddenLayers']
];

class Player {
  private timerIndex = 0;

  private isPlaying = false;

  private callback: (isPlaying: boolean) => void = null;

  /** Plays/pauses the player. */
  playOrPause() {
    if (this.isPlaying) {
      this.isPlaying = false;
      this.pause();
    } else {
      this.isPlaying = true;
      if (iter === 0) {
        simulationStarted();
      }
      this.play();
    }
  }

  onPlayPause(callback: (isPlaying: boolean) => void) {
    this.callback = callback;
  }

  play() {
    this.pause();
    this.isPlaying = true;
    if (this.callback) {
      this.callback(this.isPlaying);
    }
    this.start(this.timerIndex);
  }

  pause() {
    this.timerIndex++;
    this.isPlaying = false;
    if (this.callback) {
      this.callback(this.isPlaying);
    }
  }

  private start(localTimerIndex: number) {
    d3.timer(() => {
      if (localTimerIndex < this.timerIndex) {
        return true; // Done.
      }
      oneStep();
      return false; // Not done.
    }, 0);
  }
}

const state = State.deserializeState();

// Filter out inputs that are hidden.
state.getHiddenProps().forEach((prop) => {
  if (prop in INPUTS) delete INPUTS[prop];
});

let boundary: { [id: string]: number[][] } = {};
let selectedNodeId: string = null;

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

const linkWidthScale = d3.scale
  .linear()
  .domain([0, 5])
  .range([1, 10])
  .clamp(true);

const colorScale = d3.scale
  .linear<string, number>()
  .domain([-1, 0, 1])
  .range(['#f59322', '#e8eaeb', '#0877bd'])
  .clamp(true);

let iter = 0;
let trainData: Example2D[] = [];
let testData: Example2D[] = [];
let network: nn.Node[][] = null;
let lossTrain = 0;
let lossTest = 0;
const player = new Player();
// Line chart showing train & test errors
const lineChart = new AppendingLineChart(
  d3.select("#linechart"),
  ["#777", "black"]
);

function makeGUI() {
  d3.select('#reset-button').on('click', () => {
    reset();
    userHasInteracted();
    d3.select('#play-pause-button');
  });

  d3.select('#play-pause-button').on('click', () => {
    // Change the button's content.
    userHasInteracted();
    player.playOrPause();
  });

  player.onPlayPause((isPlaying) => {
    d3.select('#play-pause-button').classed('playing', isPlaying);
  });

  d3.select('#next-step-button').on('click', () => {
    player.pause();
    userHasInteracted();
    if (iter === 0) simulationStarted();
    oneStep();
  });

  d3.select('#data-regen-button').on('click', () => {
    generateData();
    parametersChanged = true;
  });

  const dataThumbnails = d3.selectAll('canvas[data-dataset]');
  dataThumbnails.on('click', function () {
    const newDataset = datasets[this.dataset.dataset];

    if (newDataset === state.dataset) return; // Do nothing.

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
    .classed("selected", true);

  // TODO: Remove when switched to Random Forest
  d3.select('#add-layers').on('click', () => {
    if (state.numHiddenLayers >= 6) return;
    state.networkShape[state.numHiddenLayers] = 2;
    state.numHiddenLayers++;
    parametersChanged = true;
    reset();
  });

  d3.select('#remove-layers').on('click', () => {
    if (state.numHiddenLayers <= 0) return;
    state.numHiddenLayers--;
    state.networkShape.splice(state.numHiddenLayers);
    parametersChanged = true;
    reset();
  });

  const showTestData = d3.select('#show-test-data').on('change', () => {
    state.showTestData = this.checked;
    state.serialize();
    userHasInteracted();
    heatMap.updateTestPoints(state.showTestData ? testData : []);
  });

  // Check/uncheck the checkbox according to the current state.
  showTestData.property('checked', state.showTestData);

  const discretize = d3.select('#discretize').on('change', () => {
    state.discretize = this.checked;
    state.serialize();
    userHasInteracted();
    updateUI();
  });

  // Check/uncheck the checbox according to the current state.
  discretize.property('checked', state.discretize);

  // Configure the ratio of training data to test data.
  const percTrain = d3.select('#percTrainData').on('input', () => {
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

  const batchSize = d3.select('#batchSize').on('input', function () {
    state.batchSize = this.value;
    d3.select("label[for='batchSize'] .value")
      .text(this.value);
    parametersChanged = true;
    reset();
  });
  batchSize.property('value', state.batchSize);
  d3.select("label[for='batchSize'] .value")
    .text(state.batchSize);

  const activationDropdown = d3.select('#activations')
    .on('change', function () {
      state.activation = activations[this.value];
      parametersChanged = true;
      reset();
    });
  activationDropdown.property(
    'value',
    getKeyFromValue(activations, state.activation)
  );

  const learningRate = d3.select('#learningRate').on('change', function () {
    state.learningRate = +this.value;
    state.serialize();
    userHasInteracted();
    parametersChanged = true;
  });
  learningRate.property('value', state.learningRate);

  const regularDropdown = d3
    .select('#regularizations')
    .on('change', function () {
      state.regularization = regularizations[this.value];
      parametersChanged = true;
      reset();
    });
  regularDropdown.property(
    'value',
    getKeyFromValue(regularizations, state.regularization)
  );

  const regularRate = d3.select('#regularRate').on('change', function () {
    state.regularizationRate = +this.value;
    parametersChanged = true;
    reset();
  });
  regularRate.property('value', state.regularizationRate);

  const problem = d3.select('#problem').on('change', function () {
    state.problem = problems[this.value];
    generateData();
    drawDatasetThumbnails();
    parametersChanged = true;
    reset();
  });
  problem.property('value', getKeyFromValue(problems, state.problem));

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
      drawNetwork(network);
      updateUI(true);
    }
  });

  // Hide the text below the visualization depending on the URL.
  if (state.hideText) {
    d3.select('#article-text').style('display', 'none');
    d3.select('div.more').style('display', 'none');
    d3.select('header').style('display', 'none');
  }
}

function updateBiasesUI(network: nn.Node[][]) {
  nn.forEachNode(network, true, (node) => {
    d3.select(`rect#bias-${node.id}`)
      .style('fill', colorScale(node.bias));
  });
}

function updateWeightsUI(network: nn.Node[][], container) {
  for (let layerIdx = 1; layerIdx < network.length; layerIdx++) {
    const currentLayer = network[layerIdx];
    // Update all the nodes in this layer.
    for (let i = 0; i < currentLayer.length; i++) {
      const node = currentLayer[i];
      // Update all the links to this node.
      for (let j = 0; j < node.inputLinks.length; j++) {
        const link = node.inputLinks[j];
        container
          .select(`#link${link.source.id}-${link.dest.id}`)
          .style({
            'stroke-dashoffset': -iter / 3,
            'stroke-width': linkWidthScale(Math.abs(link.weight)),
            stroke: colorScale(link.weight)
          })
          .datum(link);
      }
    }
  }
}

function drawNode(
  cx: number,
  cy: number,
  nodeId: string,
  isInput: boolean,
  container,
  node?: nn.Node
) {
  const x = cx - RECT_SIZE / 2;
  const y = cy - RECT_SIZE / 2;


  const nodeGroup = container.append('g').attr({
    class: 'node',
    id: `node${nodeId}`,
    transform: `translate(${x},${y})`
  });

  // Draw the main rectangle.
  nodeGroup.append('rect').attr({
    x: 0,
    y: 0,
    width: RECT_SIZE,
    height: RECT_SIZE
  });
  const activeOrNotClass = state[nodeId] ? 'active' : 'inactive';
  if (isInput) {
    const label = INPUTS[nodeId].label != null ? INPUTS[nodeId].label : nodeId;
    // Draw the input label.
    const text = nodeGroup.append('text').attr({
      class: 'main-label',
      x: -10,
      y: RECT_SIZE / 2,
      'text-anchor': 'end'
    });
    if (/[_^]/.test(label)) {
      const myRe = /(.*?)([_^])(.)/g;
      let myArray;
      let lastIndex;
      while ((myArray = myRe.exec(label)) != null) {
        lastIndex = myRe.lastIndex;
        const prefix = myArray[1];
        const sep = myArray[2];
        const suffix = myArray[3];
        if (prefix) {
          text.append('tspan').text(prefix);
        }
        text
          .append('tspan')
          .attr('baseline-shift', sep === '_' ? 'sub' : 'super')
          .style('font-size', '9px')
          .text(suffix);
      }
      if (label.substring(lastIndex)) {
        text.append('tspan').text(label.substring(lastIndex));
      }
    } else {
      text.append('tspan').text(label);
    }
    nodeGroup.classed(activeOrNotClass, true);
  }
  if (!isInput) {
    // Draw the node's bias.
    nodeGroup
      .append('rect')
      .attr({
        id: `bias-${nodeId}`,
        x: - BIAS_SIZE - 2,
        y: RECT_SIZE - BIAS_SIZE + 3,
        width: BIAS_SIZE,
        height: BIAS_SIZE
      })
      .on('mouseenter', () => updateHoverCard(
        HoverType.BIAS,
        node,
        d3.mouse(container.node())
      ))
      .on('mouseleave', () => updateHoverCard(null));
  }

  // Draw the node's canvas.
  const div = d3
    .select('#network')
    .insert('div', ':first-child')
    .attr({
      id: `canvas-${nodeId}`,
      class: 'canvas'
    })
    .style({
      position: 'absolute',
      left: `${x + 3}px`,
      top: `${y + 3}px`
    })
    .on('mouseenter', () => {
      selectedNodeId = nodeId;
      div.classed('hovered', true);
      nodeGroup.classed('hovered', true);
      updateDecisionBoundary(network, false);
      // Change the background of the heat map accordingly.
      heatMap.updateBackground(boundary[nodeId], state.discretize);
    })
    .on('mouseleave', () => {
      selectedNodeId = null;
      div.classed('hovered', false);
      nodeGroup.classed('hovered', false);
      updateDecisionBoundary(network, false);
      heatMap.updateBackground(
        boundary[nn.getOutputNode(network).id],
        state.discretize
      );
    });
  if (isInput) {
    div.on('click', () => {
      state[nodeId] = !state[nodeId];
      parametersChanged = true;
      reset();
    });
    div.style('cursor', 'pointer');
  }
  if (isInput) {
    div.classed(activeOrNotClass, true);
  }

  const nodeHeatMap = new HeatMap(
    RECT_SIZE,
    DENSITY / 10,
    xDomain,
    xDomain,
    div,
    { noSvg: true }
  );

  div.datum({ heatmap: nodeHeatMap, id: nodeId });
}

// Draw network
function drawNetwork(network: nn.Node[][]): void {
  const svg = d3.select('#svg');
  // Remove all svg elements.
  svg.select('g.core').remove();
  // Remove all div elements.
  d3.select('#network').selectAll('div.canvas').remove();
  d3.select('#network').selectAll('div.plus-minus-neurons').remove();

  // Get the width of the svg container.
  const padding = 3;
  const co = d3.select('.column.output').node() as HTMLDivElement;
  const cf = d3.select('.column.features').node() as HTMLDivElement;
  const width = co.offsetLeft - cf.offsetLeft;
  svg.attr('width', width);

  // Map of all node coordinates.
  const node2coord: { [id: string]: { cx: number; cy: number } } = {};
  const container = svg
    .append('g')
    .classed('core', true)
    .attr('transform', `translate(${padding},${padding})`);

  // Draw the network layer by layer.
  const numLayers = network.length;
  const featureWidth = 118;
  const layerScale = d3.scale
    .ordinal<number, number>()
    .domain(d3.range(1, numLayers - 1))
    .rangePoints([featureWidth, width - RECT_SIZE], 0.7);
  const nodeIndexScale = (nodeIndex: number) => nodeIndex * (RECT_SIZE + 25);

  const calloutThumb = d3.select('.callout.thumbnail').style('display', 'none');
  const calloutWeights = d3.select('.callout.weights').style('display', 'none');
  let idWithCallout = null;
  let targetIdWithCallout = null;

  // Draw the input layer separately.
  let cx = RECT_SIZE / 2 + 50;
  const nodeIds = Object.keys(INPUTS);
  let maxY = nodeIndexScale(nodeIds.length);
  nodeIds.forEach((nodeId, i) => {
    const cy = nodeIndexScale(i) + RECT_SIZE / 2;
    node2coord[nodeId] = { cx, cy };
    drawNode(cx, cy, nodeId, true, container);
  });

  // Draw the intermediate layers.
  for (let layerIdx = 1; layerIdx < numLayers - 1; layerIdx++) {
    const numNodes = network[layerIdx].length;
    const cx = layerScale(layerIdx) + RECT_SIZE / 2;
    maxY = Math.max(maxY, nodeIndexScale(numNodes));
    addPlusMinusControl(layerScale(layerIdx), layerIdx);
    for (let i = 0; i < numNodes; i++) {
      const node = network[layerIdx][i];
      const cy = nodeIndexScale(i) + RECT_SIZE / 2;
      node2coord[node.id] = { cx, cy };
      drawNode(cx, cy, node.id, false, container, node);

      // Show callout to thumbnails.
      const numNodes = network[layerIdx].length;
      const nextNumNodes = network[layerIdx + 1].length;
      if (
        idWithCallout == null &&
        i === numNodes - 1 &&
        nextNumNodes <= numNodes
      ) {
        calloutThumb.style({
          display: null,
          top: `${20 + 3 + cy}px`,
          left: `${cx}px`
        });
        idWithCallout = node.id;
      }

      // Draw links.
      for (let j = 0; j < node.inputLinks.length; j++) {
        const link = node.inputLinks[j];
        const path: SVGPathElement = drawLink(
          link,
          node2coord,
          network,
          container,
          j === 0,
          j,
          node.inputLinks.length
        ).node() as any;
        // Show callout to weights.
        const prevLayer = network[layerIdx - 1];
        const lastNodePrevLayer = prevLayer[prevLayer.length - 1];
        if (
          targetIdWithCallout == null &&
          i === numNodes - 1 &&
          link.source.id === lastNodePrevLayer.id &&
          (link.source.id !== idWithCallout || numLayers <= 5) &&
          link.dest.id !== idWithCallout &&
          prevLayer.length >= numNodes
        ) {
          const midPoint = path.getPointAtLength(path.getTotalLength() * 0.7);
          calloutWeights.style({
            display: null,
            top: `${midPoint.y + 5}px`,
            left: `${midPoint.x + 3}px`
          });
          targetIdWithCallout = link.dest.id;
        }
      }
    }
  }

  // Draw the output node separately.
  cx = width + RECT_SIZE / 2;
  const node = network[numLayers - 1][0];
  const cy = nodeIndexScale(0) + RECT_SIZE / 2;
  node2coord[node.id] = { cx, cy };
  // Draw links.
  for (let i = 0; i < node.inputLinks.length; i++) {
    const link = node.inputLinks[i];
    drawLink(
      link,
      node2coord,
      network,
      container,
      i === 0,
      i,
      node.inputLinks.length
    );
  }
  // Adjust the height of the svg.
  svg.attr('height', maxY);

  // Adjust the height of the features column.
  const height = Math.max(
    getRelativeHeight(calloutThumb),
    getRelativeHeight(calloutWeights),
    getRelativeHeight(d3.select('#network'))
  );
  d3.select('.column.features').style('height', `${height}px`);
}

function getRelativeHeight(selection) {
  const node = selection.node() as HTMLAnchorElement;
  return node.offsetHeight + node.offsetTop;
}

function addPlusMinusControl(x: number, layerIdx: number) {
  const div = d3
    .select('#network')
    .append('div')
    .classed('plus-minus-neurons', true)
    .style('left', `${x - 10}px`);

  const i = layerIdx - 1;
  const firstRow = div.append('div').attr('class', `ui-numNodes${layerIdx}`);
  firstRow
    .append('button')
    .attr('class', 'mdl-button mdl-js-button mdl-button--icon')
    .on('click', () => {
      const numNeurons = state.networkShape[i];
      if (numNeurons >= 8) {
        return;
      }
      state.networkShape[i]++;
      parametersChanged = true;
      reset();
    })
    .append('i')
    .attr('class', 'material-icons')
    .text('add');

  firstRow
    .append('button')
    .attr('class', 'mdl-button mdl-js-button mdl-button--icon')
    .on('click', () => {
      const numNeurons = state.networkShape[i];
      if (numNeurons <= 1) {
        return;
      }
      state.networkShape[i]--;
      parametersChanged = true;
      reset();
    })
    .append('i')
    .attr('class', 'material-icons')
    .text('remove');

  const suffix = state.networkShape[i] > 1 ? 's' : '';
  div.append('div').text(`${state.networkShape[i]} neuron${suffix}`);
}

function updateHoverCard(
  type: HoverType,
  nodeOrLink?: nn.Node | nn.Link,
  coordinates?: [number, number]
) {
  const hovercard = d3.select('#hovercard');
  if (type == null) {
    hovercard.style('display', 'none');
    d3.select('#svg').on('click', null);
    return;
  }
  d3.select('#svg').on('click', () => {
    hovercard.select('.value').style('display', 'none');
    const input = hovercard.select('input');
    input.style('display', null);
    input.on('input', function () {
      if (this.value != null && this.value !== '') {
        if (type === HoverType.WEIGHT) {
          (nodeOrLink as nn.Link).weight = +this.value;
        } else {
          (nodeOrLink as nn.Node).bias = +this.value;
        }
        updateUI();
      }
    });
    input.on('keypress', () => {
      if ((d3.event as any).keyCode === 13) {
        updateHoverCard(type, nodeOrLink, coordinates);
      }
    });
    (input.node() as HTMLInputElement).focus();
  });
  const value =
    type === HoverType.WEIGHT
      ? (nodeOrLink as nn.Link).weight
      : (nodeOrLink as nn.Node).bias;
  const name = type === HoverType.WEIGHT ? 'Weight' : 'Bias';
  hovercard.style({
    left: `${coordinates[0] + 20}px`,
    top: `${coordinates[1]}px`,
    display: 'block'
  });
  hovercard.select('.type')
    .text(name);
  hovercard.select('.value')
    .style('display', null)
    .text(value.toPrecision(2));
  hovercard
    .select('input')
    .property('value', value.toPrecision(2))
    .style('display', 'none');
}

function drawLink(
  input: nn.Link,
  node2coord: { [id: string]: { cx: number; cy: number } },
  network: nn.Node[][],
  container,
  isFirst: boolean,
  index: number,
  length: number
) {
  const line = container.insert('path', ':first-child');
  const source = node2coord[input.source.id];
  const dest = node2coord[input.dest.id];
  const datum = {
    source: {
      y: source.cx + RECT_SIZE / 2 + 2,
      x: source.cy
    },
    target: {
      y: dest.cx - RECT_SIZE / 2,
      x: dest.cy + ((index - (length - 1) / 2) / length) * 12
    }
  };
  const diagonal = d3.svg.diagonal().projection((d) => [d.y, d.x]);
  line.attr({
    'marker-start': 'url(#markerArrow)',
    class: 'link',
    id: 'link' + input.source.id + '-' + input.dest.id,
    d: diagonal(datum, 0)
  });

  // Add an invisible thick link that will be used for
  // showing the weight value on hover.
  container
    .append('path')
    .attr('d', diagonal(datum, 0))
    .attr('class', 'link-hover')
    .on('mouseenter', function () {
      updateHoverCard(HoverType.WEIGHT, input, d3.mouse(this));
    })
    .on('mouseleave', () => {
      updateHoverCard(null);
    });
  return line;
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

function getLoss(network: nn.Node[][], dataPoints: Example2D[]): number {
  let loss = 0;
  for (let i = 0; i < dataPoints.length; i++) {
    const dataPoint = dataPoints[i];
    const input = constructInput(dataPoint.x, dataPoint.y);
    const output = nn.forwardProp(network, input);
    loss += nn.Errors.SQUARE.error(output, dataPoint.label);
  }
  return loss / dataPoints.length;
}

function updateUI(firstStep = false) {
  // Update the links visually.
  updateWeightsUI(network, d3.select('g.core'));
  // Update the bias values visually.
  updateBiasesUI(network);
  // Get the decision boundary of the network.
  updateDecisionBoundary(network, firstStep);
  const selectedId =
    selectedNodeId != null ? selectedNodeId : nn.getOutputNode(network).id;
  heatMap.updateBackground(boundary[selectedId], state.discretize);

  // Update all decision boundaries.
  d3.select('#network')
    .selectAll('div.canvas')
    .each((data: { heatmap: HeatMap; id: string }) => {
      data.heatmap.updateBackground(
        reduceMatrix(boundary[data.id], 10),
        state.discretize
      );
    });

  function zeroPad(n: number): string {
    const pad = '000000';
    return (pad + n).slice(-pad.length);
  }

  function addCommas(s: string): string {
    return s.replace(/\B(?=(\d{3})+(?!\d))/g, ',');
  }

  function humanReadable(n: number): string {
    return n.toFixed(3);
  }

  // Update loss and iteration number.
  d3.select('#loss-train').text(humanReadable(lossTrain));
  d3.select('#loss-test').text(humanReadable(lossTest));
  d3.select('#iter-number').text(addCommas(zeroPad(iter)));
  lineChart.addDataPoint([lossTrain, lossTest]);
}

function constructInputIds(): string[] {
  const result: string[] = [];
  for (const inputName in INPUTS) {
    if (state[inputName]) {
      result.push(inputName);
    }
  }
  return result;
}

function constructInput(x: number, y: number): number[] {
  const input: number[] = [];
  for (const inputName in INPUTS) {
    if (state[inputName]) {
      input.push(INPUTS[inputName].f(x, y));
    }
  }
  return input;
}

function oneStep(): void {
  iter++;
  trainData.forEach((point, i) => {
    const input = constructInput(point.x, point.y);
    nn.forwardProp(network, input);
    nn.backProp(network, point.label, nn.Errors.SQUARE);
    if ((i + 1) % state.batchSize === 0) {
      nn.updateWeights(network, state.learningRate, state.regularizationRate);
    }
  });
  // Compute the loss.
  lossTrain = getLoss(network, trainData);
  lossTest = getLoss(network, testData);
  updateUI();
}

function getOutputWeights(network: nn.Node[][]): number[] {
  const weights: number[] = [];
  for (let layerIdx = 0; layerIdx < network.length - 1; layerIdx++) {
    const currentLayer = network[layerIdx];
    for (let i = 0; i < currentLayer.length; i++) {
      const node = currentLayer[i];
      for (let j = 0; j < node.outputs.length; j++) {
        const output = node.outputs[j];
        weights.push(output.weight);
      }
    }
  }
  return weights;
}

function reset(onStartup = false) {
  lineChart.reset();
  state.serialize();
  if (!onStartup) {
    userHasInteracted();
  }
  player.pause();

  const suffix = state.numHiddenLayers !== 1 ? 's' : '';
  d3.select('#layers-label').text('Hidden layer' + suffix);
  d3.select('#num-layers').text(state.numHiddenLayers);

  // Make a simple network.
  iter = 0;
  const numInputs = constructInput(0, 0).length;
  const shape = [numInputs].concat(state.networkShape).concat([1]);
  const outputActivation =
    state.problem === Problem.REGRESSION
      ? nn.Activations.LINEAR
      : nn.Activations.TANH;
  network = nn.buildNetwork(
    shape,
    state.activation,
    outputActivation,
    state.regularization,
    constructInputIds(),
    state.initZero
  );
  lossTrain = getLoss(network, trainData);
  lossTest = getLoss(network, testData);
  drawNetwork(network);
  updateUI(true);
}

function initTutorial() {
  if (state.tutorial == null || state.tutorial === '' || state.hideText) {
    return;
  }
  // Remove all other text.
  d3.selectAll('article div.l--body').remove();
  const tutorial = d3
    .select('article')
    .append('div')
    .attr('class', 'l--body');
  // Insert tutorial text.
  d3.html(`tutorials/${state.tutorial}.html`, (err, htmlFragment) => {
    if (err) {
      throw err;
    }
    tutorial.node().appendChild(htmlFragment);
    // If the tutorial has a <title> tag, set the page title to that.
    const title = tutorial.select('title');
    if (title.size()) {
      d3.select('header h1')
        .style({
          'margin-top': '20px',
          'margin-bottom': '20px'
        })
        .text(title.text());
      document.title = title.text();
    }
  });
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

function hideControls() {
  // Set display:none to all the UI elements that are hidden.
  const hiddenProps = state.getHiddenProps();
  hiddenProps.forEach((prop) => {
    const controls = d3.selectAll(`.ui-${prop}`);
    if (controls.size() === 0) {
      console.warn(`0 html elements found with class .ui-${prop}`);
    }
    controls.style('display', 'none');
  });

  // Also add checkbox for each hidable control in the "use it in classrom"
  // section.
  const hideControls = d3.select('.hide-controls');
  HIDABLE_CONTROLS.forEach(([text, id]) => {
    const label = hideControls
      .append('label')
      .attr('class', 'mdl-checkbox mdl-js-checkbox mdl-js-ripple-effect');
    const input = label.append('input').attr({
      type: 'checkbox',
      class: 'mdl-checkbox__input'
    });
    if (hiddenProps.indexOf(id) === -1) {
      input.attr('checked', 'true');
    }
    input.on('change', function () {
      state.setHideProperty(id, !this.checked);
      state.serialize();
      userHasInteracted();
      d3.select('.hide-controls-link').attr('href', window.location.href);
    });
    label.append('span')
      .attr('class', 'mdl-checkbox__label label')
      .text(text);
  });
  d3.select('.hide-controls-link').attr('href', window.location.href);
}

function generateData(firstTime = false) {
  if (!firstTime) {
    // Change the seed.
    state.seed = Math.random().toFixed(5);
    state.serialize();
    userHasInteracted();
  }
  Math.seedrandom(state.seed);
  const numSamples =
    state.problem === Problem.REGRESSION
      ? NUM_SAMPLES_REGRESS
      : NUM_SAMPLES_CLASSIFY;
  const generator =
    state.problem === Problem.CLASSIFICATION ? state.dataset : state.regDataset;
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
  ga('set', 'page', page);
  ga('send', 'pageview', { sessionControl: 'start' });
}

function simulationStarted() {
  ga('send', {
    hitType: 'event',
    eventCategory: 'Starting Simulation',
    eventAction: parametersChanged ? 'changed' : 'unchanged',
    eventLabel: state.tutorial == null ? '' : state.tutorial
  });
  parametersChanged = false;
}

export default function main() {
  drawDatasetThumbnails();
  initTutorial();
  makeGUI();
  generateData(true);
  reset(true);
  hideControls();
}
