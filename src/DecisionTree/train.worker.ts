// import { RFClassifier } from './RandomForest/classifier';
import {
  DecisionTreeClassifier as DTClassifier,
  DecisionTreeRegression as DTRegressor
} from 'ml-cart';

const ctx: Worker = self as any;
let rf;

ctx.onmessage = function(msg: MessageEvent) {
  const { options, trainingSet, labels, isClassifier } = msg.data;
  rf =  isClassifier ? new DTClassifier(options) : new DTRegressor(options);
  rf.train(trainingSet, labels);
  const model = JSON.parse(JSON.stringify(rf));
  postMessage(model);
};
