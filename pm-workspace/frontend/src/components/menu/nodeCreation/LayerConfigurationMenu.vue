<!--
Copyright (c) 2025 Antmicro <www.antmicro.com>

SPDX-License-Identifier: Apache-2.0
-->

<!--
Popup menu for configuring node layer.
-->

<!-- eslint-disable vue/no-mutating-props -->
<template>
    <div class="create-menu">
        <div v-for="option in configurationOptions" :key="option.id">
            <component
                :is="option.component"
                :intf="option"
                v-model="newNodeData[
                    option.configurationVModel as keyof NodeDataConfiguration
                ]"
                class="__name-option"
            />
        </div>
        <component :is="setLayer.component" :intf="setLayer" />
    </div>
</template>

<script lang="ts">
import {
    defineComponent, computed,
} from 'vue';
import { ButtonInterface } from '@baklavajs/renderer-vue'; // eslint-disable-line object-curly-newline

import { modifyConfiguration } from '../../../core/nodeCreation/Configuration.ts';
import { menuState, NodeDataConfiguration, configurationState } from '../../../core/nodeCreation/ConfigurationState.ts';

import InputInterface from '../../../interfaces/InputInterface.js';
import newInputInterface from './utils.ts';

interface NodeConfigurationInterface extends InputInterface {
    componentName: string,
    configurationVModel: keyof NodeDataConfiguration
}

export default defineComponent({
    setup() {
        const newNodeData = configurationState.nodeData;

        const close = () => {
            menuState.layerMenu = false;
        };

        const nodeLayer = newInputInterface<NodeConfigurationInterface>('Node layer', 'Default layer', 'layer');

        const setLayer = computed(() => {
            const button: any = new ButtonInterface('Set layer', () => {
                modifyConfiguration();
                close();
            });
            button.componentName = 'ButtonInterface';
            return button;
        });

        const configurationOptions = computed(() =>
            [nodeLayer.value],
        );

        return {
            configurationOptions,
            newNodeData,
            setLayer,
        };
    },
});
</script>
