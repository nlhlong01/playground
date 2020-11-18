// import { RFClassifier } from './RandomForest/classifier';
import {
  RandomForestClassifier as RFClassifier,
  RandomForestRegression as RFRegressor
} from './RandomForest/index';

const ctx: Worker = self as any;
let rf;

ctx.onmessage = function(msg: MessageEvent) {
  const { options, trainingSet, labels, isClassifier } = msg.data;
  rf =  isClassifier ? new RFClassifier(options) : new RFRegressor(options);
  rf.train(trainingSet, labels);
  const model = JSON.parse(JSON.stringify(rf));
  postMessage(model);
};
