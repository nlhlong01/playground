onmessage = function({ classifier, trainingSet, labels }) {
  console.log('Worker: Message received from main script');
  classifier.train(trainingSet, labels);
  console.log('Worker: Posting message back to main script');
  postMessage(classifier);
};
