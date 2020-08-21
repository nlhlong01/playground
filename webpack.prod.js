/* eslint-disable no-undef */
const path = require("path");

module.exports = {
  context: path.resolve(__dirname, "src"),
  entry: "./app.ts",
  mode: "production",
  module: {
    rules: [
      {
        test: /\.tsx?$/,
        use: "ts-loader",
        exclude: /node_modules/
      }
    ]
  },
  resolve: {
    extensions: [".tsx", ".ts", ".js"]
  },
  output: {
    path: path.resolve(__dirname, "dist"),
    filename: "bundle.js"
  }
};
