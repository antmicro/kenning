<!--
Copyright (c) 2025 Antmicro <www.antmicro.com>

SPDX-License-Identifier: Apache-2.0
-->

<!--
Popup menu for choosing export options.
-->

<!-- eslint-disable vue/no-mutating-props -->
<template>
    <div class="create-menu">
        <div v-for="[option, label] in options" :key="option.id" class="option">
            <component
                :is="option.component"
                :intf="option"
                v-model="exportGraph[label]"
            />
        </div>
    </div>
</template>

<script>
import {
    defineComponent, computed, markRaw,
} from 'vue';

import IntegerInterface from '../../interfaces/IntegerInterface.js';
import ButtonInterface from '../../interfaces/ButtonInterface.js';
import InputInterface from '../../interfaces/InputInterface.js';
import InputInterfaceComponent from '../../interfaces/InputInterface.vue';

export default defineComponent({
    props: {
        modelValue: {
            type: Boolean,
            default: false,
        },
        exportGraph: {
            required: true,
            type: Object,
        },
    },
    setup(props, { emit }) {
        const close = () => {
            if (props.modelValue) {
                emit('update:modelValue', false);
            }
        };

        const createIntegerOption = (text, label) => [computed(() => {
            if (props.exportGraph[label] === undefined) return undefined;

            const option = new IntegerInterface(text, props.exportGraph[label]);
            option.events.setValue.subscribe(this, (v) => {
                props.exportGraph[label] = v; // eslint-disable-line vue/no-mutating-props,max-len,no-param-reassign
            });
            option.setDefaultComponent();
            return option;
        }), label];

        const width = createIntegerOption('Width', 'width');
        const height = createIntegerOption('Height', 'height');

        const fileName = [computed(() => {
            const component = new InputInterface('File name');
            component.componentName = 'InputInterface';
            component.setComponent(markRaw(InputInterfaceComponent));
            return component;
        }), 'saveName'];

        const save = [computed(() => new ButtonInterface('Export', () => {
            props.exportGraph.exportCallback();
            close();
        })), undefined];

        const options =
            computed(() => [width, height, fileName, save]
                .filter(([option, _]) => option.value !== undefined)
                .map(([option, label]) => [option.value, label]));

        return { options };
    },
});
</script>
