// import { RFClassifier } from './RandomForest/classifier';
import {
  RandomForestClassifier as RFClassifier
} from './RandomForest/RandomForestClassifier';

const ctx: Worker = self as any;
let classifier: RFClassifier;

ctx.onmessage = function(msg: MessageEvent) {
  const { options, trainingSet, labels } = msg.data;
  classifier = new RFClassifier(options);
  classifier.train(trainingSet, labels);
  const model = JSON.parse(JSON.stringify(classifier));

  postMessage(model);
};
