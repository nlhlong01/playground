/* eslint-disable @typescript-eslint/naming-convention */
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

import * as nn from './nn';
import * as dataset from './dataset';
import 'seedrandom';

/** A map between names and activation functions. */
export const activations: { [key: string]: nn.ActivationFunction } = {
  relu: nn.Activations.RELU,
  tanh: nn.Activations.TANH,
  sigmoid: nn.Activations.SIGMOID,
  linear: nn.Activations.LINEAR
};

/** A map between names and regularization functions. */
export const regularizations: { [key: string]: nn.RegularizationFunction } = {
  none: null,
  L1: nn.RegularizationFunction.L1,
  L2: nn.RegularizationFunction.L2
};

/** A map between dataset names and functions generating classification data. */
export const datasets: { [key: string]: dataset.DataGenerator } = {
  circle: dataset.classifyCircleData,
  xor: dataset.classifyXORData,
  gauss: dataset.classifyTwoGaussData,
  spiral: dataset.classifySpiralData
};

/** A map between dataset names and functions that generate regression data. */
export const regDatasets1D: { [key: string]: dataset.DataGenerator } = {
  'reg1D-linear': dataset.regressLinear,
  'reg1D-quadr': dataset.regressQuadr,
  'reg1D-quadrShift': dataset.regressQuadrShift,
  'reg1D-sine': dataset.regressSine,
  'reg1D-sigmoid': dataset.regressSigmoid,
  'reg1D-step': dataset.regressStep
};

/** A map between dataset names and functions that generate regression data. */
export const regDatasets2D: { [key: string]: dataset.DataGenerator } = {
  'reg2D-plane': dataset.regressPlane,
  'reg2D-gauss': dataset.regressGaussian
};

export function getKeyFromValue(obj: any, value: any): string {
  for (const key in obj) if (obj[key] === value) return key;

  return undefined;
}

/**
 * The data type of a state variable. Used for determining the
 * (de)serialization method.
 */
export enum Type {
  STRING,
  NUMBER,
  ARRAY_NUMBER,
  ARRAY_STRING,
  BOOLEAN,
  OBJECT
}

export enum Problem {
  CLASSIFICATION,
  REGRESSION_1D,
  REGRESSION_2D
}

export const problems = {
  classification: Problem.CLASSIFICATION,
  regression1D: Problem.REGRESSION_1D,
  regression2D: Problem.REGRESSION_2D
};

export interface Property {
  name: string;
  type: Type;
  keyMap?: { [key: string]: any };
}

// Add the GUI state.
export class State {
  private static PROPS: Property[] = [
    { name: 'dataset', type: Type.OBJECT, keyMap: datasets },
    { name: 'regDataset1D', type: Type.OBJECT, keyMap: regDatasets1D },
    { name: 'regDataset2D', type: Type.OBJECT, keyMap: regDatasets2D },
    { name: 'noise', type: Type.NUMBER },
    { name: 'seed', type: Type.STRING },
    { name: 'showTestData', type: Type.BOOLEAN },
    { name: 'discretize', type: Type.BOOLEAN },
    { name: 'percTrainData', type: Type.NUMBER },
    { name: 'x', type: Type.BOOLEAN },
    { name: 'y', type: Type.BOOLEAN },
    { name: 'tutorial', type: Type.STRING },
    { name: 'problem', type: Type.OBJECT, keyMap: problems },
    { name: 'initZero', type: Type.BOOLEAN },
    { name: 'percSamples', type: Type.NUMBER },
    { name: 'nTrees', type: Type.NUMBER },
    { name: 'maxDepth', type: Type.NUMBER },
    { name: 'maxFeatures', type: Type.NUMBER }
  ];

  [key: string]: any;
  showTestData = false;
  noise = 0;
  discretize = false;
  tutorial: string = null;
  percTrainData = 70;
  problem = Problem.CLASSIFICATION;
  initZero = false;
  x = true;
  y = true;
  dataset: dataset.DataGenerator = dataset.classifyCircleData;
  regDataset1D: dataset.DataGenerator = dataset.regressLinear;
  regDataset2D: dataset.DataGenerator = dataset.regressPlane;
  seed: string;
  percSamples = 80;
  nTrees = 100;
  maxDepth = 5;
  maxFeatures = 2;

  /**
   * Deserializes the state from the url hash.
   */
  static deserializeState(): State {
    const map: { [key: string]: string } = {};
    for (const keyvalue of window.location.hash.slice(1).split('&')) {
      const [name, value] = keyvalue.split('=');
      map[name] = value;
    }
    const state = new State();

    function hasKey(name: string): boolean {
      return name in map && map[name] != null && map[name].trim() !== '';
    }

    function parseArray(value: string): string[] {
      return value.trim() === '' ? [] : value.split(',');
    }

    // Deserialize regular properties.
    State.PROPS.forEach(({ name, type, keyMap }) => {
      switch (type) {
        case Type.OBJECT:
          if (keyMap == null) {
            throw Error(
              // eslint-disable-next-line max-len
              'A key-value map must be provided for state variables of type Object'
            );
          }
          if (hasKey(name) && map[name] in keyMap) {
            state[name] = keyMap[map[name]];
          }
          break;

        case Type.NUMBER:
          if (hasKey(name)) {
            // "+" operator is for converting a string to a number.
            state[name] = +map[name];
          }
          break;

        case Type.STRING:
          if (hasKey(name)) {
            state[name] = map[name];
          }
          break;

        case Type.BOOLEAN:
          if (hasKey(name)) {
            state[name] = map[name] !== 'false';
          }
          break;

        case Type.ARRAY_NUMBER:
          if (name in map) {
            state[name] = parseArray(map[name]).map(Number);
          }
          break;

        case Type.ARRAY_STRING:
          if (name in map) {
            state[name] = parseArray(map[name]);
          }
          break;

        default:
          throw Error('Encountered an unknown type for a state variable');
      }
    });

    // Deserialize state properties that correspond to hiding UI controls.
    if (state.seed == null) {
      state.seed = Math.random().toFixed(5);
    }
    Math.seedrandom(state.seed);

    return state;
  }

  /**
   * Serializes the state into the url hash.
   */
  serialize() {
    // Serialize regular properties.
    const props: string[] = [];
    State.PROPS.forEach(({ name, type, keyMap }) => {
      let value = this[name];
      // Don't serialize missing values.
      if (value == null) {
        return;
      }
      if (type === Type.OBJECT) {
        value = getKeyFromValue(keyMap, value);
      } else if (type === Type.ARRAY_NUMBER || type === Type.ARRAY_STRING) {
        value = value.join(',');
      }
      props.push(`${name}=${value}`);
    });
    window.location.hash = props.join('&');
  }
}
