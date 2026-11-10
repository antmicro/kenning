/*
 * Copyright (c) 2025 Antmicro <www.antmicro.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 */

import { computed, ComputedRef, markRaw } from 'vue';
import InputInterface from '../../../interfaces/InputInterface.js';
import InputInterfaceComponent from '../../../interfaces/InputInterface.vue';

function newInputInterface<T>(text: string, value: string, vModelName: string): ComputedRef<T> {
    return computed(() => {
        const option: any = new InputInterface(
            text,
            value,
        ).setPort(false);

        option.componentName = 'InputInterface';
        option.configurationVModel = vModelName;
        option.setComponent(markRaw(InputInterfaceComponent));
        return option as T;
    });
}

export default newInputInterface;
