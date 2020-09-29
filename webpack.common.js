/* eslint-disable no-undef */
/* eslint-disable @typescript-eslint/naming-convention */
const path = require('path');
const HtmlWebpackPlugin = require('html-webpack-plugin');
const MiniCssExtractPlugin = require('mini-css-extract-plugin');
const { CleanWebpackPlugin } = require('clean-webpack-plugin');

module.exports = {
  context: path.resolve(__dirname, 'src'),
  entry: './playground_rf.ts',
  plugins: [
    new HtmlWebpackPlugin({
      template: './index_rf.html'
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
      }
    ]
  },
  resolve: {
    extensions: ['.tsx', '.ts', '.js']
  },
  output: {
    path: path.resolve(__dirname, 'dist'),
    filename: 'bundle.js'
  }
};
