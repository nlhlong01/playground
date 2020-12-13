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

import { Schema, Validator } from 'jsonschema';

/**
 * A two dimensional example: x and y coordinates with the label.
 */
export type Point = {
  x: number;
  y: number;
};

const schema: Schema = {
  type: 'array',
  maxItems: 500,
  items: {
    type: 'object',
    required: ['x', 'y'],
    properties: {
      x: {
        type: 'number',
        minimum: -6,
        maximum: 6
      },
      y: {
        type: 'number',
        minimum: -6,
        maximum: 6
      }
    }
  }
};

/**
 * Check if the JSON data is in valid format based on the predefined schema.
 * @param data input data.
 * @returns {boolean} true if the data is valid.
 */
export function isValid(data: any): boolean {
  return new Validator().validate(data, schema).valid;
}

/**
 * Shuffles the array using Fisher-Yates algorithm. Uses the seedrandom
 * library as the random generator.
 */
export function shuffle(array: any[]): void {
  let counter = array.length;
  let temp = 0;
  let index = 0;

  // While there are elements in the array
  while (counter > 0) {
    // Pick a random index
    index = Math.floor(Math.random() * counter);
    // Decrease counter by 1
    counter--;
    // And swap the last element with it
    temp = array[counter];
    array[counter] = array[index];
    array[index] = temp;
  }
}

export type DataGenerator = (
  numSamples: number,
  noise: number
) => Point[];

function regressFunc(f: (x: number) => number) {
  return (numSamples: number, noise: number): Point[] => {
    const points: Point[] = [];
    for (let i = 0; i < numSamples; i++) {
      let x: number;
      let y: number;
      let noiseY: number;

      do {
        x = randUniform(-6, 6);
        noiseY = normalRandom() * noise;
        y = f(x) + noiseY;
      } while (y < -6 || y > 6);

      points.push({ x, y });
    }
    return points;
  };
}

export const regressLinear = regressFunc((x) => x);
export const regressQuadr = regressFunc((x) => (6 / 40) * x ** 2 - 3);
export const regressQuadrShift = regressFunc(
  (x) => (3 / 40) * x ** 2 - x / 2 - 1
);
export const regressSine = regressFunc((x) => 2 * Math.sin(x));
export const regressSigmoid = regressFunc((x) => 5 - 5 * (1 + Math.tanh(x)));
export const regressStep = regressFunc((x) => x > 2 ? 3 : -3);

/**
 * Returns a sample from a uniform [a, b] distribution.
 * Uses the seedrandom library as the random generator.
 */
function randUniform(a: number, b: number) {
  return Math.random() * (b - a) + a;
}

/**
 * Samples from a normal distribution. Uses the seedrandom library as the
 * random generator.
 *
 * @param mean The mean. Default is 0.
 * @param variance The variance. Default is 1.
 */
function normalRandom(mean = 0, variance = 1): number {
  let v1: number;
  let v2: number;
  let s: number;

  do {
    v1 = 2 * Math.random() - 1;
    v2 = 2 * Math.random() - 1;
    s = v1 ** 2 + v2 ** 2;
  } while (s > 1);

  const result = Math.sqrt(-2 * Math.log(s) / s) * v1;
  return mean + Math.sqrt(variance) * result;
}
