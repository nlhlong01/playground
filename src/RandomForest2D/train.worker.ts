import {
  RandomForestClassifier as RFClassifier,
  RandomForestRegression as RFRegressor
} from 'ml-random-forest';

const ctx: Worker = self as any;
let rf;

ctx.onmessage = function(msg: MessageEvent) {
  const { options, trainingSet, labels, isClassifier } = msg.data;
  rf =  isClassifier ? new RFClassifier(options) : new RFRegressor(options);
  rf.train(trainingSet, labels);
  const model = JSON.parse(JSON.stringify(rf));
  postMessage(model);
};
