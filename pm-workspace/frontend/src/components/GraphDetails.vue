<!--
Copyright (c) 2025 Antmicro <www.antmicro.com>

SPDX-License-Identifier: Apache-2.0
-->

<template>
    <div class="details-panel">
        <div class="__content">
            <div class="=__header">
                <div class="__title">{{ graphNodeName }}</div>
            </div>
            <div class="__properties" v-if="displayedProperties.length">
                <div class="__title">Properties</div>
                <div v-for="input in displayedProperties" :key="input.id" class="__property">
                    <CustomInterface
                        :node="node"
                        :intf="input"
                        :toggleGroup="toggleGroup"
                        :updateDynamicInterfaces="updateDynamicInterfaces"
                        :sidebar="true"
                        tabindex="-1"
                    />
                </div>
            </div>
            <div class="__properties" v-if="viewModel.settings.editableNodeTypes && inSubgraph">
                <button
                    class="baklava-button __validate-button"
                    @click="addProperties"
                >
                    Add properties
                </button>
            </div>
        </div>
    </div>
</template>

<script>
import { computed } from 'vue';
import CustomInterface from '../custom/CustomInterface.vue';
import { configurationState, menuState } from '../core/nodeCreation/ConfigurationState.ts';
import EditorManager from '../core/EditorManager.js';

export default {
    components: {
        CustomInterface,
    },
    setup() {
        const editorManager = EditorManager.getEditorManagerInstance();
        const viewModel = computed(() => editorManager.baklavaView);
        const node = computed(() => viewModel.value.editor.graph.graphNode);
        const inSubgraph = computed(() => node.value !== undefined);
        // eslint-disable-next-line no-nested-ternary
        const graphNodeName = computed(() => (node.value
            ? (node.value.title !== '' ? node.value.title : node.value.type)
            : viewModel.value.editor.graph.name));
        const displayedProperties = computed(() => (node.value
            ? Object.values(node.value.inputs).filter((intf) => !intf.port)
            : viewModel.value.editor.getExposedProperties()));

        const toggleGroup = (intf) => {
            intf.group.forEach((name) => {
                node.value.inputs[name].hidden = !intf.value;
            });
        };

        const updateDynamicInterfaces = (intf) => {
            node.value.updateDynamicInterfaces(intf);
        };

        const addProperties = () => {
            configurationState.editedType = node.value.type;

            const configuredProperties = displayedProperties.value?.map((prop) => ({
                name: prop?.name,
                type: prop?.type,
                default: prop?.value,
                min: prop?.min,
                max: prop?.max,
                values: prop?.items,
                step: prop?.step,
                readonly: prop?.readonly,
                dtype: prop?.dtype,
            }));
            configurationState.properties = configuredProperties;

            menuState.propertyMenu = true;
        };

        return {
            viewModel,
            node,
            graphNodeName,
            displayedProperties,
            toggleGroup,
            updateDynamicInterfaces,
            addProperties,
            inSubgraph,
        };
    },
};
</script>

<style lang="scss">
.details-panel {
    $settings-width: 435px;
    $settings-maxwidth: calc(100vw - 2 * $spacing-l);

    position: absolute;
    padding: 0;
    top: calc($navbar-height + 1px);
    right: -495px;

    height: calc(100% - $navbar-height - $terminal-container-height);
    width: $settings-width;
    max-width: $settings-maxwidth;

    background-color: #{$gray-600}E6;
    color: white;

    z-index: 4;
    overflow-y: auto;

    & > .__content {
        display: flex;
        flex-direction: column;

        & > div {
            border: 1px solid #393939;
            border-right: 0;
            border-left: 0;
            border-top: 0;

            & > .__title {
                text-align: left;
                font-weight: 600;
                padding: 1.25em 0.625em;

                border: 1px solid #393939;
                border-right: 0;
                border-left: 0;
                border-top: 0;
            }
        }

        & > .__properties {
            & > .__property {
                padding: 0.625em 1.25em 0 1.25em;

                & > .__property-name {
                    font-size: 0.9em;
                }

                & * {
                    box-sizing: border-box;
                }
            }

            & > .__property:last-child {
                padding-bottom: 1.25em;
            }

            & > button {
                margin: 1.25em 1.25em;
            }
        }
    }
}
</style>
