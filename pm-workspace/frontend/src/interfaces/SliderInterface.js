/*
 * Copyright (c) 2022-2024 Antmicro <www.antmicro.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

import { NodeInterface } from '@baklavajs/core';
import DefaultComponentMixin from './DefaultComponentMixin.js';

export default class SliderInterface extends NodeInterface {
    // eslint-disable-next-line class-methods-use-this
    lazyComponent = () => import(/* webpackMode: "eager" */ './SliderInterface.vue');

    componentName = 'SliderInterface';

    constructor(name, value, min, max, step, readonly = false) {
        super(name, value);
        this.min = min;
        this.max = max;
        this.step = step;
        this.readonly = readonly;
        this.defaultComponent = true;
        this.setPort(false);
    }
}

Object.assign(SliderInterface.prototype, DefaultComponentMixin);
