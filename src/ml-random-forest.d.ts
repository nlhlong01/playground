export declare class RandomForestClassifier {
  constructor(options);
  train: (trainingset: number[][], trainingValues: number[]) => void;
  predict: (toPredict: number[][]) => number[];
}
