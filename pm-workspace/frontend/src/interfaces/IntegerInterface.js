/*
 * Copyright (c) 2022-2025 Antmicro <www.antmicro.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

import { IntegerInterface as BaklavaIntegerInterface } from '@baklavajs/renderer-vue';
import DefaultComponentMixin from './DefaultComponentMixin.js';

export default class IntegerInterface extends BaklavaIntegerInterface {
    // eslint-disable-next-line class-methods-use-this
    lazyComponent = () => import(/* webpackMode: "eager" */ './IntegerInterface.vue');

    componentName = 'IntegerInterface';

    constructor(name, value, min, max, readonly = false, dynamicCounter = false) {
        super(name, value, min, max);
        this.readonly = readonly;
        this.dynamicCounter = dynamicCounter;
        this.defaultComponent = true;
        this.setPort(false);
    }
}

Object.assign(IntegerInterface.prototype, DefaultComponentMixin);
