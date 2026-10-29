<!--
Copyright (c) 2025 Antmicro <www.antmicro.com>

SPDX-License-Identifier: Apache-2.0
-->

<!--
Popup menu for configuring node data.
-->

<!-- eslint-disable vue/no-mutating-props -->
<template>
    <div class="create-menu">
        <div v-for="option in configurationOptions" :key="option.id">
            <component
                :is="option.component"
                :intf="option"
                v-model="newNode[
                    option.configurationVModel as keyof NodeDataConfiguration
                ]"
                class="__name-option"
            />
        </div>
        <component :is="create.component" :intf="create" />
    </div>
</template>

<script lang="ts">
import { defineComponent, computed, markRaw } from 'vue';
import {
    ButtonInterface,
} from '@baklavajs/renderer-vue'; // eslint-disable-line object-curly-newline

import { createNode, duplicateNode, modifyConfiguration } from '../../../core/nodeCreation/Configuration.ts';
import { menuState, NodeDataConfiguration, configurationState } from '../../../core/nodeCreation/ConfigurationState.ts';

import InputInterface from '../../../interfaces/InputInterface.js';
import TextAreaInterface from '../../../interfaces/TextAreaInterface.js';
import TextAreaInterfaceComponent from '../../../interfaces/TextAreaInterface.vue';
import newInputInterface from './utils.ts';

interface NodeConfigurationInterface extends InputInterface {
    componentName: string,
    configurationVModel: keyof NodeDataConfiguration
}

export default defineComponent({
    setup() {
        if (menuState.configurationMenu.addNode) {
            // Reset node configuration
            configurationState.nodeData = {
                name: 'Custom Node',
                category: 'Default category',
                layer: '',
                description: '',
            };
            configurationState.properties = [];
            configurationState.interfaces = [];
            configurationState.editedType = undefined;
            configurationState.success = false;
        }
        const newNode: NodeDataConfiguration = configurationState.nodeData;

        const close = () => {
            menuState.configurationMenu.visible = false;
        };

        const nodeName = newInputInterface<NodeConfigurationInterface>('Node name', 'Custom Node', 'name');
        const nodeCategory = newInputInterface<NodeConfigurationInterface>('Node category', 'Default', 'category');

        if (!configurationState.nodeData.description) {
            configurationState.nodeData.description = '';
        }
        const nodeDescription = computed(() => {
            const option: any = new TextAreaInterface(
                'Node description',
                'Description',
            ).setPort(false);

            option.componentName = 'TextAreaInterface';
            option.configurationVModel = 'description';
            option.setComponent(markRaw(TextAreaInterfaceComponent));
            return option as NodeConfigurationInterface;
        });

        const create = computed(() => {
            let button: any;
            if (menuState.configurationMenu.addNode === true) {
                button = new ButtonInterface('Create', () => {
                    const errors = createNode();
                    configurationState.success = errors.length === 0;
                    close();
                });
            } else if (menuState.configurationMenu.duplicateNode === true) {
                button = new ButtonInterface('Duplicate', () => {
                    const errors = duplicateNode();
                    configurationState.success = errors.length === 0;
                    close();
                });
            } else {
                button = new ButtonInterface('Configure', () => {
                    const errors = modifyConfiguration();
                    configurationState.success = errors.length === 0;
                    close();
                });
            }
            button.componentName = 'ButtonInterface';
            return button;
        });
        const configurationOptions = computed(() =>
            [nodeName.value, nodeCategory.value, nodeDescription.value],
        );

        return {
            configurationOptions,
            create,
            newNode,
        };
    },
});

</script>
