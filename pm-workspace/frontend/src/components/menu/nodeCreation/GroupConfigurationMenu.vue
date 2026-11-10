<!--
Copyright (c) 2026 Antmicro <www.antmicro.com>

SPDX-License-Identifier: Apache-2.0
-->

<!--
Popup menu for group creation.
-->

<!-- eslint-disable vue/no-mutating-props -->
<template>
    <div class="create-menu">
        <div v-for="option in configurationOptions" :key="option.id">
            <component
                :is="option.component"
                :intf="option"
                v-model="newGroup[option.configurationVModel as keyof CurrentProperty]"
                class="__name-option"
            />
        </div>
    </div>
</template>

<script lang="ts">
import { defineComponent, computed } from 'vue';
import {
    ButtonInterface,
} from '@baklavajs/renderer-vue'; // eslint-disable-line object-curly-newline

import { menuState, NodeDataConfiguration } from '../../../core/nodeCreation/ConfigurationState.ts';
import { createGroupFromSelection } from '../../../core/nodeCreation/Configuration.ts';

import InputInterface from '../../../interfaces/InputInterface.js';
import newInputInterface from './utils.ts';

interface NodeConfigurationInterface extends InputInterface {
    componentName: string,
    configurationVModel: keyof NodeDataConfiguration
}
interface GroupConfig {
    name: string,
    color: string,
}

function randomColor() {
    // eslint-disable-next-line no-bitwise
    return `#${((1 << 24) * Math.random() | 0).toString(16).padStart(6, '0')}`;
}

export default defineComponent({
    setup() {
        const newGroup: GroupConfig = {
            name: 'New Group',
            color: randomColor(),
        };
        const close = () => {
            menuState.groupMenu = false;
        };

        const groupName = newInputInterface<NodeConfigurationInterface>('Group name', newGroup.name, 'name');
        const groupColor = newInputInterface<NodeConfigurationInterface>('Group color', newGroup.color, 'color');

        const addGroupMenu = computed(() => {
            const button: any = new ButtonInterface('Create group', () => {
                const prop = {
                    name: newGroup.name,
                    color: newGroup.color,
                };
                createGroupFromSelection(prop.name, prop.color);

                close();
            });
            button.componentName = 'ButtonInterface';
            return button;
        });

        const configurationOptions = computed(() =>
            [groupName.value, groupColor.value, addGroupMenu.value],
        );

        return {
            configurationOptions,
            newGroup,
        };
    },
});

</script>
