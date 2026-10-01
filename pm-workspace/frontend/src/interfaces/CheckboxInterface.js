/*
 * Copyright (c) 2022-2025 Antmicro <www.antmicro.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

import { CheckboxInterface as BaklavaCheckboxInterface } from '@baklavajs/renderer-vue';
import DefaultComponentMixin from './DefaultComponentMixin.js';

export default class CheckboxInterface extends BaklavaCheckboxInterface {
    // eslint-disable-next-line class-methods-use-this
    lazyComponent = () => import(/* webpackMode: "eager" */ './CheckboxInterface.vue');

    componentName = 'CheckboxInterface';

    constructor(name, value, readonly = false) {
        super(name, value);
        this.setPort(false);
        this.defaultComponent = true;
        this.readonly = readonly;
    }
}

Object.assign(CheckboxInterface.prototype, DefaultComponentMixin);
