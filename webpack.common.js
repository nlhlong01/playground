/* eslint-disable no-undef */
/* eslint-disable @typescript-eslint/naming-convention */
const path = require('path');
const HtmlWebpackPlugin = require('html-webpack-plugin');
const MiniCssExtractPlugin = require('mini-css-extract-plugin');
const { CleanWebpackPlugin } = require('clean-webpack-plugin');

module.exports = {
  context: path.resolve(__dirname, 'src'),
  entry: {
    RandomForest1D: './RandomForest1D/playground.ts',
    RandomForest2D: './RandomForest2D/playground.ts',
    DecisionTree: './DecisionTree/playground.ts'
  },
  plugins: [
    new HtmlWebpackPlugin({
      template: './RandomForest2D/index.html',
      filename: 'index.html',
      inject: false
    }),
    new HtmlWebpackPlugin({
      template: './RandomForest1D/index.html',
      filename: 'randomforest1d.html',
      inject: false
    }),
    new HtmlWebpackPlugin({
      template: './DecisionTree/index.html',
      filename: 'decisiontree.html',
      inject: false
    }),
    new MiniCssExtractPlugin({
      filename: 'bundle.css'
    }),
    new CleanWebpackPlugin()
  ],
  module: {
    rules: [
      {
        test: /\.css$/i,
        use: [MiniCssExtractPlugin.loader, 'css-loader']
      },
      {
        test: /\.worker\.js$/,
        loader: 'worker-loader'
      }
    ]
  },
  resolve: {
    extensions: ['.ts', '.js']
  },
  output: {
    filename: '[name].bundle.js',
    path: path.resolve(__dirname, 'dist')
  }
};
