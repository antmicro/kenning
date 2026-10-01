<!--
Copyright (c) 2022-2025 Antmicro <www.antmicro.com>

SPDX-License-Identifier: Apache-2.0
-->

<!--
Popup menu for choosing save options.
-->

<!-- eslint-disable vue/no-mutating-props -->
<template>
    <div class="create-menu">
        <div v-for="[option, label] in options" :key="option.id" class="option">
            <component
                :is="option.component"
                :intf="option"
                v-model="saveConfiguration[label]"
            />
        </div>
    </div>
</template>

<script>
import {
    defineComponent, computed, markRaw, ref, watch, toRef,
} from 'vue';

import InputInterface from '../../interfaces/InputInterface.js';
import InputInterfaceComponent from '../../interfaces/InputInterface.vue';
import ButtonInterface from '../../interfaces/ButtonInterface.js';
import CheckboxInterface from '../../interfaces/CheckboxInterface.js';

export default defineComponent({
    props: {
        modelValue: {
            type: Boolean,
            default: false,
        },
        saveConfiguration: {
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

        const createCheckboxOption = (text, label) => [computed(() => {
            if (props.saveConfiguration[label] === undefined) return undefined;

            const option = new CheckboxInterface(text, props.saveConfiguration[label]);
            option.events.setValue.subscribe(this, (v) => {
                props.saveConfiguration[label] = v; // eslint-disable-line vue/no-mutating-props,max-len,no-param-reassign
            });
            option.setDefaultComponent();
            return option;
        }), label];

        const saveConfiguration = toRef(props, 'saveConfiguration');
        const createInputOption = (text, label) => {
            const createComponent = () => {
                const component = new InputInterface(text);
                component.componentName = 'InputInterface';
                component.setComponent(markRaw(InputInterfaceComponent));
                props.saveConfiguration[label] = ''; // eslint-disable-line vue/no-mutating-props,max-len,no-param-reassign
                return component;
            };

            const exists = computed(() => saveConfiguration.value[label] !== undefined);
            const option = ref(exists.value ? createComponent() : undefined);

            watch(exists, (existsNow, existedBefore) => {
                if (existedBefore === existsNow) return;
                option.value = existsNow ? createComponent() : undefined;
            });

            return [option, label];
        };

        const readonly = createCheckboxOption('Make graph read only', 'readonly');
        const hideHud = createCheckboxOption('Disable HUD', 'hideHud');
        const position = createCheckboxOption('Preserve current view location', 'position');
        const hideLayers = createCheckboxOption('Save hidden layers info', 'hideLayers');
        const graph = createCheckboxOption('Save graph', 'graph');
        const minify = createCheckboxOption('Minify', 'minify');
        const graphName = createInputOption('Graph name', 'graphName');
        const dataflowname = createInputOption('File name', 'saveName');

        const save = [computed(() => new ButtonInterface('Save', () => {
            props.saveConfiguration.saveCallback();
            close();
        })), undefined];

        const options = computed(() =>
            [readonly, hideHud, position, hideLayers, graph, minify, graphName, dataflowname, save]
                .filter(([option, _]) => option?.value !== undefined)
                .map(([option, label]) => [option.value, label]));

        return { options };
    },
});
</script>
