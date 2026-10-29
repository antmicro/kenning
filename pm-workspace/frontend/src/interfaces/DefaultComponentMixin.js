/*
 * Copyright (c) 2022-2025 Antmicro <www.antmicro.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

import { markRaw } from 'vue';

const DefaultComponentMixin = {
    setDefaultComponent() {
        delete this.defaultComponent;
        this.lazyComponent().then((module) => {
            this.setComponent(markRaw(module.default));
        });
    },
};

export default DefaultComponentMixin;
