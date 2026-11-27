/*
 * Copyright (c) 2022-2023 Antmicro <www.antmicro.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

import { NodeInterface } from '@baklavajs/core';
import DefaultComponentMixin from './DefaultComponentMixin.js';

export default class InputInterface extends NodeInterface {
    // eslint-disable-next-line class-methods-use-this
    lazyComponent = () => import(/* webpackMode: "eager" */ './InputInterface.vue');

    componentName = 'InputInterface';

    constructor(name, value, readonly = false) {
        super(name, value);
        this.readonly = readonly;
        this.setPort(false);
    }
}

Object.assign(InputInterface.prototype, DefaultComponentMixin);
