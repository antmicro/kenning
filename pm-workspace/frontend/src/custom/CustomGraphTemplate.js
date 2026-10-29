/*
 * Copyright (c) 2022-2025 Antmicro <www.antmicro.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

import { GraphTemplate as BaklavaGraphTemplate } from '@baklavajs/core';

export default class GraphTemplate extends BaklavaGraphTemplate {
    _name = '';

    constructor(state, editor) {
        super(state, editor);
        if (state.name) this.name = state.name;
    }
}
