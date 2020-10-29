export type ClassifierOptions = {
  replacement?: boolean;
  maxSamples?: number;
  maxFeatures?: number;
  nEstimators?: number;
  treeOptions?: any;
  isClassifier?: boolean;
  seed?: number;
  useSampleBagging?: true;
  selectionMethod?: 'mean' | 'median' | 'mode';
};

export declare class RandomForestClassifier {
  constructor(options: ClassifierOptions);
  train: (trainingset: number[][], trainingValues: number[]) => void;
  predict: (toPredict: number[][]) => number[];
}
