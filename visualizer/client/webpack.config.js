const HtmlWebPackPlugin = require("html-webpack-plugin");
const CopyPlugin = require('copy-webpack-plugin');  // For copying entire scene directory


module.exports = {
    entry: {
        sysa: './src/index.js',
        test: './src/test.js',
        data_label: './src/data_label.js',
        sysb: './src/baseline.js'
    },
    module: {
        rules: [
            {
                test:  /\.js$/,
                exclude: /node_modules/,
                use: [
                    {
                        loader: "babel-loader"
                    }
                ]
            },
            {
                test:  /\.html$/,
                use: [
                    {
                        loader: "html-loader",
                        options: { minimize: true }
                    }
                ]
            },
            {
                test:  /\.(png|svg|jpg|gif|obj)$/,
                use: [
                    {
                        loader: "file-loader"
                    }
                ]
            }
        ]
    }, 
    devtool: "eval-source-map",
    plugins: [
        new HtmlWebPackPlugin({
            template: "./src/index.html",
            filename: "./sysa.html",
            chunks: ['sysa']
        }),
        new HtmlWebPackPlugin({
            template: "./src/test.html",
            filename: "./test.html",
            chunks: ['test']
        }),
        new HtmlWebPackPlugin({
            template: "./src/data_label.html",
            filename: "./data_label.html",
            chunks: ['data_label']
        }),
        new HtmlWebPackPlugin({
            template: "./src/baseline.html",
            filename: "./sysb.html",
            chunks: ['sysb']
        }),
        new CopyPlugin({
            patterns: [
              { from: 'static'},
            ],
          }),
    ]
};