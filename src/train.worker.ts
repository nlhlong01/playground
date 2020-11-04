import {
  RandomForestClassifier as RFClassifier
} from '../../random-forest/src/RandomForestClassifier';

const ctx: Worker = self as any;
let classifier: RFClassifier;

ctx.onmessage = function(msg: MessageEvent) {
  const { options, trainingSet, labels } = msg.data;
  classifier = new RFClassifier(options);
  classifier.train(trainingSet, labels);
  const model = JSON.parse(JSON.stringify(classifier));

  postMessage(model);
};
