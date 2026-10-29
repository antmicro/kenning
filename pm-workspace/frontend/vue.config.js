/*
 * Copyright (c) 2022-2023 Antmicro <www.antmicro.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

import webpack from 'webpack';
import { defineConfig } from '@vue/cli-service';
import { fileURLToPath } from 'url';
import path, { dirname } from 'path';

const __filename = fileURLToPath(import.meta.url); // eslint-disable-line no-underscore-dangle

export default defineConfig({
    publicPath: '',
    css: {
        extract: !process.env.VUE_APP_SINGLEHTML_BUILD,
        loaderOptions: {
            scss: {
                additionalData: `
          @import "./styles/_variables.scss";
        `,
            },
        },
    },
    /* eslint-disable no-param-reassign */
    configureWebpack: (config) => {
        config.plugins = config.plugins || [];
        config.plugins.push(
            new webpack.DefinePlugin({
                __VUE_OPTIONS_API__: 'true',
                __VUE_PROD_DEVTOOLS__: 'false',
                __VUE_PROD_HYDRATION_MISMATCH_DETAILS__: 'false',
            }),
        );

        if (process.env.NODE_ENV === 'development') {
            config.devtool = 'eval-source-map';
        } else if (process.env.NODE_ENV === 'production') {
            config.optimization = {
                splitChunks: process.env.VUE_APP_SINGLEHTML_BUILD ? false : {
                    maxSize: 250000, // This value is arbitrary, can be adjusted if needed
                    chunks: 'all',
                },
            };
        }
        config.resolve = {
            alias: {
                '@baklavajs': path.resolve(dirname(__filename), 'node_modules/@baklavajs/'),
            },
            fallback: {
                fs: false,
                path: false,
                system: false,
                file: false,
            },
            extensions: ['.ts', '.js', '.json'],
        };

        config.module.rules = config.module.rules.filter(
            (rule) => !rule.test.toString().match(/(svg|png|jpg|jpeg|gif|webp|avif)/),
        );
        config.module.rules.push({
            test: /\.(svg|png|jpg|jpeg|gif|webp|avif)(\?.*)?$/,
            type: 'asset',
            generator: { filename: '[path][name][ext]' },
        });
    },
    pages: {
        index: {
            entry: 'src/main.js',
            template: 'index.html',
            filename: 'index.html',
            title: 'Pipeline Manager',
        },
    },
});
