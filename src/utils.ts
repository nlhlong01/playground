import * as d3 from 'd3';

const NUM_SHADES = 30;

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

export const color = d3
  .scaleQuantize()
  .domain([-1, 1])
  .range(colors);
