// import * as d3 from 'd3';
// import { Example2D } from './dataset';

// export class Tree {
//   private svg;

//   constructor(width: number, container, data) {
//     const tree = data => {
//       const root = d3.hierarchy(data);
//       root['dx'] = 10;
//       root['dy'] = width / (root.height + 1);
//       return d3.tree().nodeSize([root['dx'], root['dy']])(root);
//     };

//     const root = tree(data);

//     let x0 = Infinity;
//     let x1 = -x0;

//     const isLeaf = (d) => !d.left && !d.right;

//     root.each(d => {
//       if (d.x > x1) x1 = d.x;
//       if (d.x < x0) x0 = d.x;
//     });

//     this.svg = container
//       .attr("viewBox", [0, 0, width, x1 - x0 + root.dx * 2]);

//     const g = this.svg
//       .append("g")
//       .attr("font-family", "sans-serif")
//       .attr("font-size", 10)
//       .attr("transform", `translate(${root['dy'] / 3},${root['dx'] - x0})`);

//     g
//       .append('g')
//       .attr('fill', 'none')
//       .attr('stroke', '#555')
//       .attr('stroke-opacity', 0.4)
//       .attr('stroke-width', 1.5)
//       .selectAll('path')
//       .data(root.links())
//       .join('path')
//       .attr(
//         'd',
//         d3.linkHorizontal()
//           .x((d) => d.y)
//           .y((d) => d.x)
//       );

//     const node = g
//       .append('g')
//       .attr('stroke-linejoin', 'round')
//       .attr('stroke-width', 3)
//       .selectAll('g')
//       .data(root.descendants())
//       .join('g')
//       .attr('transform', (d) => `translate(${d.y},${d.x})`);

//     node
//       .append('circle')
//       .attr('fill', (d) => (isLeaf(d) ? '#999' : '#555'))
//       .attr('r', 2.5);

//     node
//       .append('text')
//       .attr('dy', '0.31em')
//       .attr('x', (d) => (isLeaf(d) ? 6 : -6))
//       .attr('text-anchor', (d) => (isLeaf(d) ? 'start' : 'end'))
//       .text((d) => 'node')
//       .clone(true)
//       .lower()
//       .attr('stroke', 'white');
//   }
// }