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
  private xScale: d3.ScaleLinear<number, number>;
  private yScale: d3.ScaleLinear<number, number>;
  private svg;

  constructor(
    width: number,
    xDomain: [number, number],
    yDomain: [number, number],
    container,
    userSettings?: PlotSettings
  ) {
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

    container = container
      .append('div')
      .style('width', `${width}px`)
      .style('height', `${height}px`)
      .style('position', 'relative')
      .style('top', `-${padding}px`)
      .style('left', `-${padding}px`);

    this.svg = container
      .append('svg')
      .attr('width', width)
      .attr('height', height)
      // Overlay the svg on top of the canvas.
      .style('position', 'absolute')
      .style('left', '0')
      .style('top', '0')
      .append('g')
      .attr('transform', `translate(${padding}, ${padding})`);

    if (!this.settings.noPoint) {
      this.svg.append('g').attr('class', 'point');
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
        .attr('transform', `translate(${width - 2 * padding},0)`)
        .call(yAxis);
    }
  }

  updatePoints(points: Point[]): void {
    if (this.settings.noPoint) {
      throw Error("Can't add points since noPoint=true");
    }
    this.updateCircles(this.svg.select('g.point'), points);
  }

  updatePlot(points: Point[]) {
    this.svg.select('path.plot').remove();

    // Keep only line segments that are inside the bounds.
    const xDomain = this.xScale.domain();
    const yDomain = this.yScale.domain();
    points = points.filter(
      (d) => (
        d.x >= xDomain[0] &&
        d.x <= xDomain[1] &&
        d.y >= yDomain[0] &&
        d.y <= yDomain[1]
      )
    );

    const line = d3
      .line<{ x: number; y: number }>()
      .x((d) => this.xScale(d.x))
      .y((d) => this.yScale(d.y));

    this.svg
      .append('path')
      .datum(points)
      .attr('class', 'plot')
      .attr('fill', 'none')
      .attr('stroke', 'coral')
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
      (p) => (
        p.x >= xDomain[0] &&
        p.x <= xDomain[1] &&
        p.y >= yDomain[0] &&
        p.y <= yDomain[1]
      )
    );

    // Attach data to initially empty selection.
    const selection = container.selectAll('circle').data(points);

    // Insert elements to match length of points array.
    selection
      .enter()
      .append('circle')
      .attr('r', 2.5);

    // Update points to be in the correct position.
    selection
      .attr('cx', (d) => this.xScale(d.x))
      .attr('cy', (d) => this.yScale(d.y))
      .style('fill', () => 'slateblue');

    // Remove points if the length has gone down.
    selection.exit().remove();
  }
} // Close class Plot.
