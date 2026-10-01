/*
 * Copyright (c) 2022-2023 Antmicro <www.antmicro.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

import { NodeInterface } from '@baklavajs/core';
import DefaultComponentMixin from './DefaultComponentMixin.js';

export default class HexInterface extends NodeInterface {
    // eslint-disable-next-line class-methods-use-this
    lazyComponent = () => import(/* webpackMode: "eager" */ './HexInterface.vue');

    componentName = 'HexInterface';

    constructor(name, value, min, max, readonly = false) {
        super(name, value);
        this.min = min;
        this.max = max;
        this.readonly = readonly;
        this.setPort(false);
    }
}

Object.assign(HexInterface.prototype, DefaultComponentMixin);
