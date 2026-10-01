<!--
Copyright (c) 2022-2026 Antmicro <www.antmicro.com>

SPDX-License-Identifier: Apache-2.0
-->

<template>
    <div class="settings-panel">
        <div class="setting-header">
            <span>Settings</span>
        </div>
        <div class="panel">
            <div v-for="option in settingOptions" :key="option.id">
                <component :is="option.component" :intf="option" tabindex="-1"></component>
            </div>

            <div class="__properties" v-show="disableLayersOptions.length !== 0">
                <div class="option-label">Hide Layers:</div>
                <div v-for="option in disableLayersOptions" :key="option.id">
                    <component :is="option.component" :intf="option" tabindex="-1"></component>
                </div>
            </div>
        </div>
    </div>
</template>

<script>
import { computed, ref } from 'vue';
import getExternalApplicationManager from '../core/communication/ExternalApplicationManager';
import { LOG_LEVEL } from '../core/notifications';

import SelectInterface from '../interfaces/SelectInterface.js';
import CheckboxInterface from '../interfaces/CheckboxInterface.js';
import IntegerInterface from '../interfaces/IntegerInterface.js';
import ButtonInterface from '../interfaces/ButtonInterface.js';
import { loadingScreen } from '../core/utils';

export default {
    props: {
        viewModel: {
            required: true,
        },
    },
    setup(props) {
        const externalApplicationManager = getExternalApplicationManager();
        const metadataChanged = (name, value) => {
            externalApplicationManager.notifyAboutChange('metadata_on_change', {
                metadata: {
                    [name]: value,
                },
            });
        };

        const connectionStyleOption = computed(() => {
            const items = [
                { text: 'Curved', value: 'curved' },
                { text: 'Orthogonal', value: 'orthogonal' },
                { text: 'Alternative orthogonal', value: 'alternativeOrthogonal' },
                { text: 'Straight', value: 'straight' },
                { text: 'A*-based', value: 'aStar' },
            ];
            const option = new SelectInterface(
                'Connection style',
                props.viewModel.connectionRenderer.style,
                items,
            );
            option.events.setValue.subscribe(this, (v) => {
                props.viewModel.connectionRenderer.style = v; // eslint-disable-line vue/no-mutating-props,max-len,no-param-reassign
                metadataChanged('connectionStyle', v);
            });
            return option;
        });

        const randomizedOffsetOption = computed(() => {
            const option = new CheckboxInterface(
                'Randomized offset',
                props.viewModel.connectionRenderer.randomizedOffset,
            );
            option.events.setValue.subscribe(this, (v) => {
                props.viewModel.connectionRenderer.randomizedOffset = v; // eslint-disable-line vue/no-mutating-props,max-len,no-param-reassign
                metadataChanged('randomizedOffset', v);
            });
            return option;
        });

        const switchableInterfacesOption = computed(() => {
            const option = new CheckboxInterface(
                'Switchable interfaces',
                props.viewModel.connectionRenderer.switchableInterfaces,
            );
            option.events.setValue.subscribe(this, (v) => {
                props.viewModel.connectionRenderer.switchableInterfaces = v; // eslint-disable-line vue/no-mutating-props,max-len,no-param-reassign
                metadataChanged('switchableInterfaces', v);
            });
            return option;
        });

        const editableNodeTypes = computed(() => {
            const option = new CheckboxInterface(
                'Modify node types',
                props.viewModel.settings.editableNodeTypes,
            ).setPort(false);
            option.events.setValue.subscribe(this, (v) => {
                props.viewModel.settings.editableNodeTypes = v; // eslint-disable-line vue/no-mutating-props,max-len,no-param-reassign
                metadataChanged('editableTypes', v);
            });
            option.componentName = 'CheckboxInterface';
            return option;
        });

        const hideAnchors = computed(() => {
            const option = new CheckboxInterface(
                'Hide anchors',
                props.viewModel.settings.hideAnchors,
            ).setPort(false);
            option.events.setValue.subscribe(this, (v) => {
                props.viewModel.settings.hideAnchors = v; // eslint-disable-line vue/no-mutating-props,max-len,no-param-reassign
                metadataChanged('hideAnchors', v);
            });
            option.componentName = 'CheckboxInterface';
            return option;
        });

        const showIds = computed(() => {
            const option = new CheckboxInterface(
                'Show IDs',
                props.viewModel.settings.showIds,
            ).setPort(false);
            option.events.setValue.subscribe(this, (v) => {
                props.viewModel.settings.showIds = v; // eslint-disable-line vue/no-mutating-props,max-len,no-param-reassign
                metadataChanged('showIds', v);
            });
            option.componentName = 'CheckboxInterface';
            return option;
        });

        const removeEmptySubgraphs = computed(() => {
            const option = new CheckboxInterface(
                'Remove empty subgraphs',
                props.viewModel.settings.removeEmptySubgraphs,
            ).setPort(false);
            option.events.setValue.subscribe(this, (v) => {
                props.viewModel.settings.removeEmptySubgraphs = v; // eslint-disable-line vue/no-mutating-props,max-len,no-param-reassign
                metadataChanged('removeEmptySubgraphs', v);
            });
            option.componentName = 'CheckboxInterface';
            return option;
        });

        const showHiddenProperties = computed(() => {
            const option = new CheckboxInterface(
                'Show hidden properties',
                props.viewModel.settings.showHiddenProperties,
            ).setPort(false);
            option.events.setValue.subscribe(this, (v) => {
                props.viewModel.settings.showHiddenProperties = v; // eslint-disable-line vue/no-mutating-props,max-len,no-param-reassign
                metadataChanged('showHiddenProperties', v);
            });
            option.componentName = 'CheckboxInterface';
            return option;
        });

        const backgroundGridSize = computed(() => {
            const option = new IntegerInterface(
                'Background grid size',
                props.viewModel.settings.background.gridSize,
            );
            option.events.setValue.subscribe(this, (v) => {
                props.viewModel.settings.background.gridSize = v; // eslint-disable-line vue/no-mutating-props,max-len,no-param-reassign
                metadataChanged('backgroundSize', v);
            });
            return option;
        });

        const movementStep = computed(() => {
            const option = new IntegerInterface(
                'Node movement step',
                props.viewModel.movementStep,
            );
            option.events.setValue.subscribe(this, (v) => {
                props.viewModel.movementStep = v; // eslint-disable-line vue/no-mutating-props,max-len,no-param-reassign
                metadataChanged('movementStep', v);
            });
            return option;
        });

        const LayoutOption = computed(() => {
            const { layoutManager } = props.viewModel.editor;
            const items = layoutManager
                .getAvailableAlgorithms()
                .map((algorithmName) => ({ text: algorithmName, value: algorithmName }));
            const option = new SelectInterface(
                'Autolayout algorithm',
                layoutManager.usedAlgorithm,
                items,
            );
            option.events.setValue.subscribe(this, (v) => {
                layoutManager.useAlgorithm(v);
                metadataChanged('layout', v);
            });
            return option;
        });

        const LayoutApply = computed(() => {
            const button = new ButtonInterface('Apply autolayout', async () => {
                const apply = async () => {
                    await props.viewModel.editor.applyAutolayout();
                    await props.viewModel.editor.swapInterfaces();
                    props.viewModel.editor.editorManager.centerZoom();
                };
                await loadingScreen(apply, props.viewModel.editor.events.setLoad);
            });
            return button;
        });

        const swapInterfaces = computed(() => {
            const button = new ButtonInterface('Swap interfaces', async () => {
                const apply = async () => {
                    props.viewModel.editor.swapInterfaces();
                };
                await loadingScreen(apply, props.viewModel.editor.events.setLoad);
            });
            return button;
        });

        const clearEditor = computed(() => {
            const button = new ButtonInterface('Clean editor', () => {
                props.viewModel.editor.graphs.forEach((graph) => graph.setSpecGraph(false));
                props.viewModel.editor.unregisterGraphs();
                props.viewModel.editor.deepCleanEditor();
                props.viewModel.editor.unregisterNodes();
                props.viewModel.editor.editorManager.setSpecificationLoaded(false);
            });
            return button;
        });

        const disableLayersOptions = computed(() => {
            const options = ref([]);
            const displayedGraphId = props.viewModel.displayedGraph.id;

            props.viewModel.layers.forEach((layer) => {
                const option = new CheckboxInterface(
                    layer.name,
                    (props.viewModel.ignoredLayers.get(
                        displayedGraphId,
                    ) ?? new Set()).has(layer.name),
                );
                option.events.setValue.subscribe(this, () => {
                    if (props.viewModel.ignoredLayers.get(
                        displayedGraphId,
                    ) === undefined) {
                        props.viewModel.ignoredLayers.set(
                            displayedGraphId,
                            new Set(),
                        );
                    }
                    const graphIgnoredLayers = props.viewModel.ignoredLayers.get(
                        displayedGraphId,
                    );
                    if (graphIgnoredLayers.has(layer.name)) {
                        graphIgnoredLayers.delete(layer.name);
                    } else {
                        graphIgnoredLayers.add(layer.name);
                    }
                });
                options.value.push(option);
            });

            return options.value;
        });

        const logLevel = computed(() => {
            const select = new SelectInterface(
                'Verbosity of notifications',
                props.viewModel.logLevel,
                Object.keys(LOG_LEVEL).map((s) => s.toUpperCase()),
            );
            select.events.setValue.subscribe(this, (v) => {
                props.viewModel.logLevel = v; // eslint-disable-line vue/no-mutating-props,max-len,no-param-reassign
            });
            return select;
        });
        const viewsPresent = computed(() =>
            props.viewModel.editor.renderViews.length > 1);

        const viewSelected = computed(() => {
            const option = new SelectInterface(
                'Selected view',
                props.viewModel.editor.currentView,
                props.viewModel.editor.renderViews.map((view) =>
                    ({ text: view, value: view })),
            ).setPort(false);

            option.events.setValue.subscribe(this, (v) => {
                const { editor } = props.viewModel;
                editor.currentView = v;
                editor.centerZoom();
            });

            return option;
        });

        const readonlyOptions = computed(() => {
            if (props.viewModel.editor.readonly) {
                return [];
            }
            const options = [
                logLevel.value,
                connectionStyleOption.value,
                LayoutOption.value,
                LayoutApply.value,
                swapInterfaces.value,
                backgroundGridSize.value,
                clearEditor.value,
                movementStep.value,
                randomizedOffsetOption.value,
                hideAnchors.value,
                switchableInterfacesOption.value,
                showIds.value,
                showHiddenProperties.value,
                removeEmptySubgraphs.value,
            ];
            if (viewsPresent.value) {
                options.push(viewSelected.value);
            }
            if (props.viewModel.settings.toggleableEditableTypes) {
                options.push(editableNodeTypes.value);
            }
            return options;
        });

        return { settingOptions: readonlyOptions, disableLayersOptions };
    },
};
</script>

<style lang="scss">
.settings-panel {
    $settings-width: 435px;
    $settings-maxwidth: calc(100vw - 2 * $spacing-l);

    background-color: #{$gray-600}E6;
    position: absolute;
    padding: $spacing-l;
    color: white;
    top: calc($navbar-height + 1px);
    right: -495px;
    width: $settings-width;
    // viewport - terminal - navbar - padding
    max-height: calc(100% - $navbar-height - $terminal-container-height - 2 * $spacing-l);
    max-width: $settings-maxwidth;
    z-index: 4;
    overflow-y: auto;

    & > .setting-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: $spacing-l;

        & > span {
            color: $white;
            font-size: $fs-large;
        }
    }

    & > .panel {
        display: grid;
        grid-row-gap: $spacing-l;
        user-select: none;

        & > div {
            max-width: $settings-maxwidth;
            & > .option-label {
                padding-bottom: $spacing-s;
                color: $white;
                font-size: $fs-medium;
            }
        }
    }
}
</style>
