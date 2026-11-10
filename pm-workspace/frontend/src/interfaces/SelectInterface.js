/*
 * Copyright (c) 2022-2025 Antmicro <www.antmicro.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

import { SelectInterface as BaklavaSelectInterface } from '@baklavajs/renderer-vue';
import DefaultComponentMixin from './DefaultComponentMixin.js';

export default class SelectInterface extends BaklavaSelectInterface {
    // eslint-disable-next-line class-methods-use-this
    lazyComponent = () => import(/* webpackMode: "eager" */ './SelectInterface.vue');

    componentName = 'SelectInterface';

    constructor(name, value, items, readonly = false) {
        super(name, value, items);
        this.readonly = readonly;
        this.defaultComponent = true;
        this.setPort(false);
    }
}

Object.assign(SelectInterface.prototype, DefaultComponentMixin);
