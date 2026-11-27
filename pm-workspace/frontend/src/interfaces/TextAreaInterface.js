/*
 * Copyright (c) 2022-2025 Antmicro <www.antmicro.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

import { NodeInterface } from '@baklavajs/core';
import DefaultComponentMixin from './DefaultComponentMixin.js';

export default class TextAreaInterface extends NodeInterface {
    // eslint-disable-next-line class-methods-use-this
    lazyComponent = () => import(/* webpackMode: "eager" */ './TextAreaInterface.vue');

    componentName = 'TextAreaInterface';

    constructor(name, value, readonly = false) {
        super(name, value);
        this.readonly = readonly;
        this.setPort(false);
    }
}

Object.assign(TextAreaInterface.prototype, DefaultComponentMixin);
