import * as d3 from 'd3';
import { HierarchyPointNode } from 'd3';

export class Tree {
  private svg;

  constructor(width: number, container, data) {
    const padding = 10;

    let root = d3.hierarchy(
      data,
      (d) => {
        const children = [];
        if (d.left) children.push(d.left);
        if (d.right) children.push(d.right);
        return children;
      }
    );
    // Set the size of root.
    root['dx'] = 60;
    root['dy'] = 60;

    root = d3
      .tree()
      .nodeSize([root['dx'], root['dy']])(root);

    let xMin = Infinity;
    let xMax = -Infinity;
    let yMin = Infinity;
    let yMax = -Infinity;

    (root as HierarchyPointNode<any>).each(d => {
      if (d.x > xMax) xMax = d.x;
      if (d.x < xMin) xMin = d.x;
      if (d.y > yMax) yMax = d.y;
      if (d.y < yMin) yMin = d.y;
    });

    this.svg = container
      .append('svg')
      // .attr('width', width)
      .attr('width', xMax - xMin + 2 * root['dx'])
      .attr('height', root['dy'] * (root.height + 2));
      // .attr('viewBox', [0, 0, width, root['dy'] * (root.height + 1)]);

    // Tree group
    const g = this.svg
      .append('g')
      .attr('font-size', 12)
      .attr(
        'transform',
        // `translate(${-xMin + (width - (xMax - xMin)) / 2},${root['dy']})`
        `translate(${-xMin + root['dx']},${root['dy']})`
      );

    // Link
    g
      .append('g')
      .attr('fill', 'none')
      .attr('stroke', '#555')
      .attr('stroke-opacity', 0.4)
      .attr('stroke-width', 1.5)
      .selectAll('path')
      .data(root.links())
      .join('path')
      .attr(
        'd',
        d3.linkVertical()
          .x((d) => d['x'])
          .y((d) => d['y'])
      );

    // Node group
    const node = g
      .append('g')
      .attr('stroke-linejoin', 'round')
      .attr('stroke-width', 3)
      .selectAll('g')
      .data(root.descendants())
      .join('g')
      .attr('transform', (d) => `translate(${d.x},${d.y})`);

    // Text
    const text = node.append('text');
      // .attr('dy', '0.31em')
      // .attr('x', (d) => d.children ? 6 : -6)
      // .attr('text-anchor', (d) => d.children ? 'start' : 'end')

    node
      .append('tspan')
      .text((d) => (
        `${!d.data.splitColumn ? 'x' : 'y'} < ${d.data.splitValue}`
      ));
    node
      .append('tspan')
      .text((d) => `gain = ${parseInt(d.data.gain).toFixed(3)}`);
  }
}