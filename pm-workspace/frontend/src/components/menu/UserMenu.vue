<!--
Copyright (c) 2026 Antmicro <www.antmicro.com>

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
    defineComponent, computed, markRaw, toRef, ref,
} from 'vue';

import InputInterface from '../../interfaces/InputInterface.js';
import InputInterfaceComponent from '../../interfaces/InputInterface.vue';
import ButtonInterface from '../../interfaces/ButtonInterface.js';
import getExternalApplicationManager from '../../core/communication/ExternalApplicationManager.js';
import CheckboxInterface from '../../interfaces/CheckboxInterface.js';
import NumberInterface from '../../interfaces/NumberInterface.js';
import IntegerInterface from '../../interfaces/IntegerInterface.js';

export default defineComponent({
    props: {
        buttonTitle: {
            required: true,
            type: String,
        },
        saveConfigurationTemplate: {
            required: true,
            type: Map,
        },
        saveConfiguration: {
            required: true,
            type: Object,
        },
    },
    setup(props) {
        const externalApplicationManager = getExternalApplicationManager();

        const saveConfigurationTemplate = toRef(props, 'saveConfigurationTemplate');
        const saveConfiguration = toRef(props, 'saveConfiguration');
        const createCheckboxOption = (params, label) => {
            const { name } = params;

            return [computed(() => {
                const option = new CheckboxInterface(name, props.saveConfiguration[label]);
                option.events.setValue.subscribe(this, (v) => {
                    props.saveConfiguration[label] = v; // eslint-disable-line vue/no-mutating-props,max-len,no-param-reassign
                });
                option.setDefaultComponent();
                return option;
            }), label];
        };

        const generateNumericalComponent = (params, label, floating) => {
            const { name } = params;
            const min = params?.min ?? Number.MIN_VALUE;
            const max = params?.max ?? Number.MAX_VALUE;

            const NumericalType = floating ? NumberInterface : IntegerInterface;
            return [computed(() => {
                const option = new NumericalType(name, props.saveConfiguration[label], min, max);
                option.events.setValue.subscribe(this, (v) => {
                    props.saveConfiguration[label] = Math.max(Math.min(v, max), min); // eslint-disable-line vue/no-mutating-props,max-len,no-param-reassign
                });
                option.setDefaultComponent();
                return option;
            }), label];
        };

        const createNumberOption = (params, label) =>
            generateNumericalComponent(params, label, true);
        const createIntOption = (params, label) =>
            generateNumericalComponent(params, label, false);

        const createInputOption = (params, label) => {
            const { name } = params;
            const createComponent = () => {
                const component = new InputInterface(name);
                component.componentName = 'InputInterface';
                component.setComponent(markRaw(InputInterfaceComponent));
                return component;
            };

            const option = ref(createComponent());

            return [option, label];
        };

        const typeToComponent = new Map(
            [
                ['text', createInputOption],
                ['number', createNumberOption],
                ['int', createIntOption],
                ['bool', createCheckboxOption],
            ],
        );

        const save = [computed(() => new ButtonInterface(`${props.buttonTitle}`, () => {
            // send API response
            externalApplicationManager.requestUserDialogRead(saveConfiguration.value);
        })), undefined];

        const formOptions = computed(() => {
            const labels = [];
            Object.entries(saveConfigurationTemplate.value).forEach(([key, value]) => {
                const { type } = value;
                const inputConstructor = typeToComponent.get(type);

                labels.push(
                    inputConstructor(value, key),
                );
            });

            return labels;
        });

        const options = computed(() =>
            [...formOptions.value, save]
                .filter(([option, _]) => option?.value !== undefined)
                .map(([option, label]) => [option.value, label]));

        return { options };
    },
});
</script>
