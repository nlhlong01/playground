/* eslint-disable */
/// <reference types="cypress" />
/**
 * @type {Cypress.PluginConfig}
 */
module.exports = (on, config) => {
  const options = {
    webpackOptions: require('../../webpack.dev')
  };
  on('file:preprocessor', require('@cypress/webpack-preprocessor')(options));

  require('@cypress/code-coverage/task')(on, config);
  return config;
};
