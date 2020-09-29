// interface Options {
//   seed: number;
//   maxFeatures: number | string;
//   replacement: boolean;
//   nEstimators: number;
// }

export declare class RandomForestClassifier {
  constructor(options);
  train: (trainingset: number[][], trainingValues: number[]) => void;
  predict: (toPredict: number[][]) => number[];
}
