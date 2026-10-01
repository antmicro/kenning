<!--
Copyright (c) 2025 Antmicro <www.antmicro.com>

SPDX-License-Identifier: Apache-2.0
-->

<!--
Popup menu with a list of checkboxes.
-->

<!-- eslint-disable vue/no-mutating-props -->
<template>
    <div class="create-menu">
        <div class="list">
            <div v-for="option in listedOptions" :key="option.id">
                <component :is="option.component" :intf="option" tabindex="-1"></component>
            </div>
            <div v-for="option in disabledOptions" :key="option.id">
                <component
                    :is="option.component"
                    :intf="option"
                    class="--disabled"
                    tabindex="-1"></component>
            </div>
        </div>
        <div class="confirm">
            <component :is="confirmButton.component" :intf="confirmButton" />
        </div>
    </div>
</template>

<script lang="ts">
import {
    defineComponent, computed, ref,
} from 'vue';
import { ButtonInterface } from '@baklavajs/renderer-vue'; // eslint-disable-line object-curly-newline
import CheckboxInterface from '../../../interfaces/CheckboxInterface.js';
import { menuState } from '../../../core/nodeCreation/ConfigurationState.ts';

export default defineComponent({
    props: {
        componentName: {
            type: String,
            required: true,
        },
        items: {
            type: Array,
            required: true,
        },
        disabledItems: {
            type: Array,
            required: true,
        },
        disabledReason: {
            type: String,
        },
        confirmAction: {
            type: Function,
            required: true,
        },
        confirmText: {
            type: String,
            required: true,
        },
    },
    setup(props) {
        const close = () => {
            (menuState as any)[props.componentName] = false;
        };

        const selectedItems = ref<any[]>([]);

        function toggleItem(item: any) {
            const i = selectedItems.value.indexOf(item);
            if (i >= 0) {
                selectedItems.value.splice(i, 1);
            } else {
                selectedItems.value.push(item);
            }
        }

        const listedOptions = computed(() => {
            const options: any = ref([]);

            props.items.forEach((item: any) => {
                if (props.disabledItems.find((i) => i === item)) {
                    return;
                }
                const option = new CheckboxInterface(item.name, false);
                option.events.setValue.subscribe(this, () => toggleItem(item));
                options.value.push(option);
            });

            return options.value;
        });
        const disabledOptions = computed(() => {
            const options: any = ref([]);
            props.disabledItems.forEach((item: any) => {
                const option = new CheckboxInterface(`${item.name} ${props.disabledReason ? `(${props.disabledReason})` : ''}`, false, true);
                option.events.setValue.subscribe(this, () => toggleItem(item));
                options.value.push(option);
            });
            return options.value;
        });

        const confirmButton = computed(() => {
            const button: any = new ButtonInterface(props.confirmText, () => {
                props.confirmAction(selectedItems.value);
                close();
            });
            button.componentName = 'ButtonInterface';
            return button;
        });

        return {
            selectedItems,
            listedOptions,
            disabledOptions,
            confirmButton,
        };
    },
});
</script>

<style lang="scss">
    .list {
        flex: 1 1 auto;
        min-height: 0;
        overflow-y: auto;
    }
    .--disabled {
        pointer-events: none;
        cursor: not-allowed;
        opacity: 0.65;
        filter: alpha(opacity=65);
        -webkit-box-shadow: none;
        box-shadow: none;
    }
</style>
