import {
  RandomForestClassifier as RFClassifier
} from '../../random-forest/src/RandomForestClassifier';

const ctx: Worker = self as any;
let classifier: RFClassifier;

ctx.onmessage = function(evt: MessageEvent) {
  const { options, trainingSet, labels } = evt.data;
  classifier = new RFClassifier(options);
  classifier.train(trainingSet, labels);
  const model = JSON.parse(JSON.stringify(classifier));

  postMessage(model);
};
