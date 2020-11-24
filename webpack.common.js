/* eslint-disable no-undef */
/* eslint-disable @typescript-eslint/naming-convention */
const path = require('path');
const HtmlWebpackPlugin = require('html-webpack-plugin');
const MiniCssExtractPlugin = require('mini-css-extract-plugin');
const { CleanWebpackPlugin } = require('clean-webpack-plugin');

module.exports = {
  context: path.resolve(__dirname, 'src'),
  entry: {
    Playground2D: './playground.ts',
    Playground1D: './Playground1D/playground.ts'
  },
  plugins: [
    new HtmlWebpackPlugin({
      template: './randomforest.html',
      filename: 'randomforest.html',
      inject: false
    }),
    new HtmlWebpackPlugin({
      template: './Playground1D/index.html',
      filename: 'index.html',
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
        use: { loader: 'worker-loader' }
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
