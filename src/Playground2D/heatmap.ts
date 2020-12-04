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

import * as d3 from 'd3';
import { Example2D } from './dataset';

export interface HeatMapSettings {
  [key: string]: any;
  showAxes?: boolean;
  noSvg?: boolean;
}

/** Number of different shades (colors) when drawing a gradient heatmap */
const NUM_SHADES = 30;

/**
 * Draws a heatmap using canvas. Used for showing the learned decision
 * boundary of the classification algorithm. Can also draw data points
 * using an svg overlayed on top of the canvas heatmap.
 */
export class HeatMap {
  // Default settings.
  private settings: HeatMapSettings = {
    showAxes: false,
    noSvg: false
  };
  private xScale: d3.ScaleLinear<number, number>;
  private yScale: d3.ScaleLinear<number, number>;
  private numSamples: number;
  private color;
  private canvas;
  private svg;

  constructor(
    width: number,
    numSamples: number,
    xDomain: [number, number],
    yDomain: [number, number],
    container,
    userSettings?: HeatMapSettings
  ) {
    this.numSamples = numSamples;
    const height = width;
    const padding = userSettings.showAxes ? 20 : 0;

    if (userSettings != null) {
      // overwrite the defaults with the user-specified settings.
      for (const prop in userSettings) {
        this.settings[prop] = userSettings[prop];
      }
    }

    this.xScale = d3
      .scaleLinear()
      .domain(xDomain)
      .range([0, width - 2 * padding]);

    this.yScale = d3
      .scaleLinear()
      .domain(yDomain)
      .range([height - 2 * padding, 0]);

    // Get a range of colors.
    const tmpScale = d3
      .scaleLinear<string, number>()
      .domain([0, 0.5, 1])
      .range(['#f59322', '#e8eaeb', '#0877bd'])
      .clamp(true);

    // Due to numerical error, we need to specify
    // d3.range(0, end + small_epsilon, step)
    // in order to guarantee that we will have end/step entries with
    // the last element being equal to end.
    const colors = d3
      .range(0, 1 + 1e-9, 1 / NUM_SHADES)
      .map((a) => tmpScale(a));

    this.color = d3
      .scaleQuantize()
      .domain([-1, 1])
      .range(colors);

    container = container.append('div').style({
      width: `${width}px`,
      height: `${height}px`,
      position: 'relative',
      top: `-${padding}px`,
      left: `-${padding}px`
    });

    this.canvas = container
      .append('canvas')
      .attr('width', numSamples)
      .attr('height', numSamples)
      .style('width', `${width - 2 * padding}px`)
      .style('height', `${height - 2 * padding}px`)
      .style('position', 'absolute')
      .style('top', `${padding}px`)
      .style('left', `${padding}px`);

    if (!this.settings.noSvg) {
      this.svg = container
        .append('svg')
        .attr({
          width: width,
          height: height
        })
        .style({
          // Overlay the svg on top of the canvas.
          position: 'absolute',
          left: '0',
          top: '0'
        })
        .append('g')
        .attr('transform', `translate(${padding}, ${padding})`);

      this.svg.append('g').attr('class', 'train');
      this.svg.append('g').attr('class', 'test');
    }

    if (this.settings.showAxes) {
      const xAxis = d3.axisBottom(this.xScale);
      const yAxis = d3.axisRight(this.yScale);

      this.svg
        .append('g')
        .attr('class', 'x axis')
        .attr('transform', `translate(0,${height - 2 * padding})`)
        .call(xAxis);

      this.svg
        .append('g')
        .attr('class', 'y axis')
        .attr('transform', 'translate(' + (width - 2 * padding) + ',0)')
        .call(yAxis);
    }
  }

  updateTestPoints(points: Example2D[]): void {
    if (this.settings.noSvg) {
      throw Error("Can't add points since noSvg=true");
    }
    this.updateCircles(this.svg.select('g.test'), points);
  }

  updatePoints(points: Example2D[]): void {
    if (this.settings.noSvg) {
      throw Error("Can't add points since noSvg=true");
    }
    this.updateCircles(this.svg.select('g.train'), points);
  }

  clearBackground(): void {
    const canvas = this.canvas.node();
    const context = canvas.getContext('2d');
    context.clearRect(0, 0, canvas.width, canvas.height);
  }

  updateBackground(data: number[][], discretize: boolean): void {
    if (!data.length || !data[0].length) throw new Error(
      'Boundary is invalid'
    );

    const dx = data[0].length;
    const dy = data.length;

    if (dx !== this.numSamples || dy !== this.numSamples) {
      throw new Error(
        'The provided data matrix must be of size ' + 'numSamples X numSamples'
      );
    }

    // Compute the pixel colors; scaled by CSS.
    const context = (this.canvas.node() as HTMLCanvasElement).getContext('2d');
    const image = context.createImageData(dx, dy);

    for (let y = 0, p = -1; y < dy; ++y) {
      for (let x = 0; x < dx; ++x) {
        let value = data[x][y];
        if (value === undefined) {
          this.clearBackground();
          return;
        }
        if (discretize) value = (value >= 0 ? 1 : -1);
        const c = d3.rgb(this.color(value));
        image.data[++p] = c.r;
        image.data[++p] = c.g;
        image.data[++p] = c.b;
        image.data[++p] = 160;
      }
    }
    context.putImageData(image, 0, 0);
  }

  private updateCircles(container, points: Example2D[]) {
    // Keep only points that are inside the bounds.
    const xDomain = this.xScale.domain();
    const yDomain = this.yScale.domain();
    points = points.filter(
      (p) =>
        p.x >= xDomain[0] &&
        p.x <= xDomain[1] &&
        p.y >= yDomain[0] &&
        p.y <= yDomain[1]
    );

    // Attach data to initially empty selection.
    const selection = container.selectAll('circle').data(points);

    // Insert elements to match length of points array.
    selection
      .enter()
      .append('circle')
      .attr('r', 3);

    // Update points to be in the correct position.
    selection
      .attr({
        cx: (d: Example2D) => this.xScale(d.x),
        cy: (d: Example2D) => this.yScale(d.y)
      })
      .style('fill', (d) => this.color(d.label));

    // Remove points if the length has gone down.
    selection.exit().remove();

    // Update hover cards.
    const hoverCard = d3.select('#hovercard');
    selection
      .on('mouseenter', (d: Example2D) => {
        if (d.voteCounts === undefined) return;
        if (d.voteCounts.length !== 2) throw new Error(
          'Vote counts are not valid'
        );

        const container = d3.select('#main-heatmap canvas');
        const coordinates = d3.pointer(container.node());

        hoverCard
          .style('left', `${coordinates[0] + 20}px`)
          .style('top', `${coordinates[1]}px`)
          .style('display', 'block');

        d3.select('#hovercard.ui-nvotes #first-class.value')
          .text(d.voteCounts[0]);
        d3.select('#hovercard.ui-nvotes #second-class.value')
          .text(d.voteCounts[1]);
      })
      .on('mouseleave', () => {
        hoverCard.style('display', 'none');
      });
  }
} // Close class HeatMap.
