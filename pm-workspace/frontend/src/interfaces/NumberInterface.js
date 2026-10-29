/*
 * Copyright (c) 2022-2025 Antmicro <www.antmicro.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

import { NumberInterface as BaklavaNumberInterface } from '@baklavajs/renderer-vue';
import DefaultComponentMixin from './DefaultComponentMixin.js';

export default class NumberInterface extends BaklavaNumberInterface {
    // eslint-disable-next-line class-methods-use-this
    lazyComponent = () => import(/* webpackMode: "eager" */ './NumberInterface.vue');

    componentName = 'NumberInterface';

    constructor(name, value, min, max, readonly = false) {
        super(name, value, min, max);
        this.readonly = readonly;
        this.defaultComponent = true;
        this.setPort(false);
    }
}

Object.assign(NumberInterface.prototype, DefaultComponentMixin);
