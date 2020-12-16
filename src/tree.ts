import * as d3 from 'd3';
import { HierarchyPointNode } from 'd3';
import { color } from './utils';

export class Tree {
  private container;
  private svg;
  private root;
  private node;

  constructor(
    nodeSize: [number, number],
    textBoxSize: [number, number],
    container
  ) {
    this.container = container;
    this.node = {
      dx: nodeSize[0],
      dy: nodeSize[1],
      width: textBoxSize[0],
      height: textBoxSize[1]
    };
    this.svg = container.append('svg');
  }

  draw(data) {
    this.root = d3.hierarchy(
      data,
      (d) => {
        const children = [];
        if (d.left) children.push(d.left);
        if (d.right) children.push(d.right);
        return children;
      }
    );
    this.root = d3
      .tree()
      .separation(() => 1)
      .nodeSize([this.node.dx, this.node.dy])(this.root);

    let xMin = Infinity;
    let xMax = -Infinity;
    let yMin = Infinity;
    let yMax = -Infinity;
    (this.root as HierarchyPointNode<any>).each(d => {
      // Get coordinates of extreme nodes.
      if (d.x > xMax) xMax = d.x;
      if (d.x < xMin) xMin = d.x;
      if (d.y > yMax) yMax = d.y;
      if (d.y < yMin) yMin = d.y;
    });

    this.svg
      .attr('width', xMax - xMin + 2 * this.node.dx)
      .attr('height', this.node.dy * (this.root.height + 2));

    // Tree group
    const g = this.svg
      .append('g')
      .attr('font-size', 12)
      .attr(
        'transform',
        `translate(${-xMin + this.node.dx},${this.node.dy})`
      );

    // Link
    g
      .append('g')
      .classed('links', true)
      .attr('fill', 'none')
      .attr('stroke', '#555')
      .attr('stroke-opacity', 0.4)
      .attr('stroke-width', 1.5)
      .selectAll('path')
      .data(this.root.links())
      .join('path')
      .attr(
        'd',
        d3.linkVertical()
          .source((d) => [
            d.source['x'],
            d.source['y'] + this.node.height / 2
          ])
          .target((d) => [
            d.target['x'],
            d.target['y'] - this.node.height / 2
          ])
      );

    // Node
    const nodeGroup = g
      .append('g')
      .classed('nodes', true)
      .attr('stroke-linejoin', 'round')
      .attr('stroke-width', 3)
      .selectAll('g')
      .data(this.root.descendants())
      .join('g')
      .attr('transform', (d) => `translate(${d.x},${d.y})`);

    // Node text
    nodeGroup
      .append('foreignObject')
      .attr('x', -this.node.width / 2)
      .attr('y', -this.node.height / 2)
      .attr('width', this.node.width)
      .attr('height', this.node.height)
      .style('border', 'thin solid')
      .append('xhtml')
      .append('div')
      .style('background-color', (d) => {
        const colorScale = d3
          .scaleLinear()
          .domain([0, 1])
          .range([1, -1]);
        return color(colorScale(d.data.distribution[0][0]));
      })
      .classed('text', true)
      .style('text-align', 'center')
      .each(function(d) {
        const { gain, splitColumn, splitValue, samples, distribution } = d.data;
        const text = d3.select(this);

        text
          .append('b')
          .html(`gain = ${gain ? gain.toFixed(3) : 0}`);

        if (splitColumn !== undefined) {
          const feature = splitColumn === 0 ? 'x' : 'y';
          text
            .append('b')
            .html(`${feature} < ${splitValue.toFixed(3)}`);
        }

        text
          .append('b')
          .html(`samples = ${samples}`);

        const x = Math.round((distribution[0][0] || 0) * samples);
        const y = samples - x;
        text
          .append('b')
          .html(`dist = [${x}, ${y}]`);
      });
  }
}