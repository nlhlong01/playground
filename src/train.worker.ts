import {
  RandomForestClassifier as RFClassifier
} from '../../random-forest/src/RandomForestClassifier';

const ctx: Worker = self as any;

ctx.onmessage = function(e) {
  const { options, trainingSet, labels } = e.data;
  const classifier = new RFClassifier(options);
  classifier.train(trainingSet, labels);
  const model = JSON.parse(JSON.stringify(classifier));

  postMessage(model);
};
