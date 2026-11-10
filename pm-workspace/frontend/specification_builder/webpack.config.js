/**
 * Copyright (c) 2025 Antmicro <www.antmicro.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

// Designed to be run from the pipeline_manager/frontend directory

import path from 'path';
import webpack from 'webpack';

export default (_, argv) =>
/** @type {import("webpack").Configuration} */({
        target: 'web',
        mode: argv.mode || 'development',
        devtool: false,
        entry: path.resolve('./specification_builder/specification_builder_source.js'),
        output: {
            path: path.resolve('.'),
            filename: './specification_builder/specification_builder.js',
            chunkFormat: 'module',
            library: 'SpecLibrary',
        },
        externalsPresets: {
            node: true,
        },
        externals: {
            './TextAreaInterface.vue': 'import("TextAreaInterface.vue")',
            './SliderInterface.vue': 'import("SliderInterface.vue")',
            './NumberInterface.vue': 'import("NumberInterface.vue")',
            './ListInterface.vue': 'import("ListInterface.vue")',
            './IntegerInterface.vue': 'import("IntegerInterface.vue")',
            './HexInterface.vue': 'import("HexInterface.vue")',
            './CheckboxInterface.vue': 'import("CheckboxInterface.vue")',
            './SelectInterface.vue': 'import("SelectInterface.vue")',
            './InputInterface.vue': 'import("InputInterface.vue")',
        },
        module: {
            rules: [
                {
                    test: /\.py$/,
                    use: 'raw-loader',
                    include: [
                        path.resolve('../dataflow_builder/data_structures.py'),
                        path.resolve('../specification_builder.py'),
                        path.resolve('./specification_builder/utils/retrieve_python_methods_info.py'),
                        path.resolve('./specification_builder/utils/argument_preparation_utils.py'),
                    ],
                },
                {
                    test: /\.(ts|js)x?$/,
                    exclude: [
                        /node_modules/,

                    ],
                    use: {
                        loader: 'babel-loader',
                        options: {
                            presets: [
                                '@babel/preset-env',
                                '@babel/preset-typescript',
                            ],
                        },
                    },
                },
            ],
        },
        resolve: {
            enforceExtension: false,
            descriptionFiles: ['package.json'],
        },
        plugins: [
            new webpack.DefinePlugin({
                'window.isWebpack': JSON.stringify(true),
            }),
        ],
    });

export { default as webpack } from 'webpack';
