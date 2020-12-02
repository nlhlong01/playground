# Machine Learning playground

## Description

Machine Learning playground is an interactive visualization of 3 ML algorithms:
Random Forest, Decision Tree and Support Vector Machine (SVM).

Tech stack: TypeScript, D3, Webpack, Cypress, Material Design.

This project is built on the [Tensorflow's Deep Machine Learning](https://playground.tensorflow.org).
The UI of the Random Forest Playground is inspired by [Alex Rogozhnikov's Gradient Boosting Interactive Playground](https://arogozhnikov.github.io/)

## Deployment

- Random Forest:
  - [1-dimensional](https://nlhlong01.github.io/playground/rf1d.html).
  - [2-dimensional](https://nlhlong01.github.io/playground).

## Development

To run the visualization locally, run:
- `npm i` to install dependencies
- `npm run build` to compile the app and place it in the `dist/` directory

For a fast edit-refresh cycle when developing run `npm start`.
This will start an http server and automatically re-compile the TypeScript,
HTML and CSS files whenever they change.
