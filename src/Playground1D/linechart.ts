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
import { Point } from './dataset';

export interface PlotSettings {
  [key: string]: any;
  showAxes?: boolean;
  noPoint?: boolean;
}

/**
 * Draws a heatmap using canvas. Used for showing the learned decision
 * boundary of the classification algorithm. Can also draw data points
 * using an svg overlayed on top of the canvas heatmap.
 */
export class LineChart {
  // Default settings.
  private settings: PlotSettings = {
    showAxes: false,
    noPoint: false
  };
  private xScale: d3.scale.Linear<number, number>;
  private yScale: d3.scale.Linear<number, number>;
  private numSamples: number;
  private svg;

  constructor(
    width: number,
    // numSamples: number,
    xDomain: [number, number],
    yDomain: [number, number],
    container,
    userSettings?: PlotSettings
  ) {
    // this.numSamples = numSamples;
    const height = width;
    const padding = userSettings.showAxes ? 20 : 0;

    if (userSettings != null) {
      // overwrite the defaults with the user-specified settings.
      for (const prop in userSettings) {
        this.settings[prop] = userSettings[prop];
      }
    }

    this.xScale = d3.scale
      .linear()
      .domain(xDomain)
      .range([0, width - 2 * padding]);

    this.yScale = d3.scale
      .linear()
      .domain(yDomain)
      .range([height - 2 * padding, 0]);

    container = container.append('div').style({
      width: `${width}px`,
      height: `${height}px`,
      position: 'relative',
      top: `-${padding}px`,
      left: `-${padding}px`
    });

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

    if (!this.settings.noPoint) {
      this.svg.append('g').attr('class', 'train');
      this.svg.append('g').attr('class', 'test');
    }

    if (this.settings.showAxes) {
      const xAxis = d3.svg
        .axis()
        .scale(this.xScale)
        .orient('bottom');

      const yAxis = d3.svg
        .axis()
        .scale(this.yScale)
        .orient('right');

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

  updateTestPoints(points: Point[]): void {
    if (this.settings.noPoint) {
      throw Error("Can't add points since noPoint=true");
    }
    this.updateCircles(this.svg.select('g.test'), points);
  }

  updatePoints(points: Point[]): void {
    if (this.settings.noPoint) {
      throw Error("Can't add points since noPoint=true");
    }
    this.updateCircles(this.svg.select('g.train'), points);
  }

  updatePlot(data: Point[]) {
    this.svg.select('path').remove();

    // Keep only points that are inside the bounds.
    const xDomain = this.xScale.domain();
    const yDomain = this.yScale.domain();
    data = data.filter(
      (p) => (
        p.x >= xDomain[0] &&
        p.x <= xDomain[1] &&
        p.y >= yDomain[0] &&
        p.y <= yDomain[1]
      )
    );

    const line = d3.svg
      .line<{ x: number; y: number }>()
      .x((d) => this.xScale(d.x))
      .y((d) => this.yScale(d.y));

    this.svg
      .append('path')
      .datum(data)
      .attr('class', 'plot')
      .attr('fill', 'none')
      .attr('stroke', 'cornflowerblue')
      .attr('stroke-width', 3)
      .attr('stroke-linejoin', 'round')
      .attr('stroke-linecap', 'round')
      .attr('d', line);
  }

  private updateCircles(container, points: Point[]) {
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
        cx: (d) => this.xScale(d.x),
        cy: (d) => this.yScale(d.y)
      })
      .style('fill', () => 'darkorange');

    // Remove points if the length has gone down.
    selection.exit().remove();
  }
} // Close class Plot.
