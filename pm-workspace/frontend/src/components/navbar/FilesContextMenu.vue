<!--
Copyright (c) 2026 Antmicro <www.antmicro.com>

SPDX-License-Identifier: Apache-2.0
-->

<!--
Defines the editor context menu, the menu for saving/opening/exporting current graph state.
-->
<script>
import { watch } from 'vue';
import jsonlint from 'jsonlint';
import { toSvg } from 'html-to-image';
import DropdownItem from '../DropdownItem.vue';
import Logo from '../../icons/Logo.vue';
import NotificationHandler from '../../core/notifications';
import { brokenImage } from '../../../../resources/broken_image.js';
import { loadingScreen } from '../../core/utils';
import { GraphFactory } from '../../core/NodeFactory.js';
import { menuState, configurationState } from '../../core/nodeCreation/ConfigurationState.ts';
import EditorManager from '../../core/EditorManager';
import { saveGraphConfiguration } from '../saveConfiguration.ts';
import getExternalApplicationManager from '../../core/communication/ExternalApplicationManager';
import {
    startTransaction, commitTransaction,
} from '../../core/History.ts';

export default {
    components: {
        DropdownItem,
        Logo,
    },
    props: {
        externalApp: {
            required: true,
            type: Object,
        },
        setEditTitle: {
            required: true,
            type: Function,
        },
        mobileClasses: {
            required: true,
            type: Object,
        },
        hideHud: {
            required: true,
            type: Boolean,
        },
        saveSpecificationCallback: {
            required: true,
            type: Function,
        },
        exportGraphCallback: {
            required: true,
            type: Function,
        },
        readonly: {
            required: true,
            type: Boolean,
        },
        saveGraphCallback: {
            required: true,
            type: Function,
        },
        requestDataflowExport: {
            required: true,
            type: Function,
        },
        hover: {
            required: true,
            type: Boolean,
        },
    },
    methods: {
        importDataflow() {
            if (!this.externalApp.available) return;
            const file = document.getElementById('request-dataflow-button').files[0];
            if (!file) return;

            const fileReader = new FileReader();
            fileReader.onload = async () => {
                await this.externalApplicationManager.importDataflow(fileReader.result);
            };
            fileReader.readAsText(file);
        },
        /**
         * Loads nodes' specification from JSON structure.
         *
         * @param {string} specText specification to validate and load.
         * @returns {Promise} result after validation and loading.
         */
        async loadDataflow(dataflow) {
            return this.editorManager.loadDataflow(dataflow).then(({ errors, warnings, info }) => {
                if (Array.isArray(warnings) && warnings.length) {
                    NotificationHandler.terminalLog(
                        'warning',
                        'Issue when loading dataflow',
                        warnings,
                    );
                }
                if (Array.isArray(errors) && errors.length) {
                    const messageTitle = process.env.VUE_APP_GRAPH_DEVELOPMENT_MODE === 'true' ?
                        'Softload enabled, errors found while loading the dataflow' :
                        'Dataflow is invalid';
                    NotificationHandler.terminalLog('error', messageTitle, errors);
                    if (Array.isArray(info) && info.length) {
                        NotificationHandler.terminalLog(
                            'error',
                            'Mismatched dataflow version',
                            `${info} Please refer to https://antmicro.github.io/kenning-pipeline-manager/changelogs.html to see if that is the case.`,
                        );
                    }
                } else if (Array.isArray(info) && info.length) {
                    NotificationHandler.terminalLog('info', 'Dataflow loaded', info);
                }
            });
        },
        /**
         * Event handler that Loads a dataflow from a file.
         * It the loading is successful it is loaded as the current dataflow.
         * Otherwise the user is alerted with a feedback message.
         */
        async loadDataflowCallback() {
            const file = document.getElementById('load-dataflow-button').files[0];
            if (!file) return;

            await loadingScreen(async () => {
                let dataflow = null;
                try {
                    dataflow = jsonlint.parse(await file.text());
                } catch (exception) {
                    if (exception instanceof SyntaxError) {
                        NotificationHandler.terminalLog(
                            'error',
                            'Not a proper JSON file',
                            exception.toString(),
                        );
                    } else {
                        NotificationHandler.terminalLog(
                            'error',
                            'Unknown error',
                            exception.toString(),
                        );
                    }
                    return;
                }

                await this.externalApplicationManager.notifyAboutChange('graph_on_change', {
                    dataflow,
                });

                await this.loadDataflow(dataflow);
            }, this.editorManager.baklavaView.editor.events.setLoad);

            document.getElementById('load-dataflow-button').value = '';
        },
        exportToSvg() {
            // Get editor with data flow
            const nodeEditor = document.querySelector('.inner-editor');
            // Exclude nodes hidden in export (e.g. node palette and zoom controls)
            const filter = (node) => !node.classList?.contains('export-hidden');
            toSvg(nodeEditor, { filter, imagePlaceholder: brokenImage })
                .then((dataUrl) => {
                    const downloadLink = document.createElement('a');
                    downloadLink.download = 'dataflow.svg';
                    downloadLink.href = dataUrl;
                    downloadLink.dataset.downloadurl = [
                        dataUrl,
                        downloadLink.download,
                        downloadLink.href,
                    ].join(':');
                    document.body.appendChild(downloadLink);
                    downloadLink.click();
                    document.body.removeChild(downloadLink);
                })
                .catch((error) => {
                    NotificationHandler.showToast('error', `Export to SVG failed: ${error}`);
                });
        },
        /**
         * Event handler that loads a specification passed by the user
         * and tries to load it into a new environment it.
         */
        async loadSpecificationCallback() {
            const file = document.getElementById('load-spec-button').files[0];
            if (!file) return;

            await loadingScreen(async () => {
                const content = await file.text();
                return this.loadSpecification(content);
            }, this.editorManager.baklavaView.editor.events.setLoad);
            document.getElementById('load-spec-button').value = '';
        },
        /**
         * Event handler that creates a specification.
         */
        async createEmptySpecification() {
            this.loadSpecification({
                version: this.editorManager.specificationVersion,
                nodes: [
                    { name: 'Default node', category: 'Default category' },
                ],
                metadata: {
                    editableTypes: true,
                },
            });
        },
        createGraphNodeTypeFromCurrentGraph() {
            menuState.configurationMenu.visible = !menuState.configurationMenu.visible;
            menuState.configurationMenu.addNode = true;
            menuState.configurationMenu.placeNode = false;

            const unwatch = watch(() =>
                menuState.configurationMenu.visible, async (newValue, oldValue) => {
                if (oldValue === true && newValue === false) {
                    const nodeName = configurationState.nodeData.name;
                    let graphNode = this.editorManager.baklavaView.editor.nodeTypes.get(nodeName);
                    let instance = new graphNode.type(); // eslint-disable-line new-cap

                    if (configurationState.success) {
                        // Save current graph
                        const currentGraph = this.editorManager.baklavaView.displayedGraph;
                        const currentGraphState = currentGraph.save();

                        const errors = this.editorManager.addSubgraphToNode(
                            instance,
                            currentGraphState.nodes,
                            currentGraphState.connections,
                        );
                        if (Array.isArray(errors) && errors.length) {
                            NotificationHandler.terminalLog('error', 'Creating subgraph failed', errors);
                        }

                        // Switch to new graph and add new node there
                        const newGraph = GraphFactory(
                            [], [], this.appName, this.editorManager.baklavaView.editor,
                        );
                        this.editorManager.baklavaView.editor.switchToGraph(newGraph);

                        graphNode = this.editorManager.baklavaView.editor.nodeTypes.get(nodeName);
                        instance = new graphNode.type(); // eslint-disable-line new-cap
                        this.editorManager.baklavaView.displayedGraph.addNode(instance);
                        instance.position.x = window.innerWidth / 2;
                        instance.position.y = window.innerHeight / 2;

                        this.editorManager.baklavaView.editor.changeTopLevelGraph(
                            this.editorManager.baklavaView.displayedGraph.id,
                        );
                    }
                }
                unwatch();
            });
        },
        /**
         * Loads nodes' specification from text structure.
         * It first validates the specification file. If the validation is successful the
         * specification is loaded. Otherwise a proper log is printed to the user.
         *
         * @param {string} specText specification to validate and load.
         * @returns {Promise} result after validation and loading.
         */
        async loadSpecification(specText) {
            const validationErrors = EditorManager.validateSpecification(specText);
            if (Array.isArray(validationErrors) && validationErrors.length) {
                NotificationHandler.terminalLog('error', 'Specification is invalid', validationErrors);
                let version;
                try {
                    version = JSON.parse(specText).version;
                } catch { return Promise.resolve(); }

                if (this.editorManager.specificationVersion !== version) {
                    NotificationHandler.terminalLog(
                        'error',
                        'Mismatched specification version',
                        `Specification version (${version}) differs from the current version (${this.editorManager.specificationVersion}). It may result in unexpected behaviour.` +
                        ' Please refer to https://antmicro.github.io/kenning-pipeline-manager/changelogs.html to see if that is the case.',
                    );
                }
                return Promise.resolve();
            }
            return this.editorManager.updateEditorSpecification(specText)
                .then(({ errors, warnings, info }) => {
                    if (Array.isArray(warnings) && warnings.length) {
                        NotificationHandler.terminalLog(
                            'warning',
                            'Issue when loading specification',
                            warnings,
                        );
                    }
                    if (Array.isArray(errors) && errors.length) {
                        NotificationHandler.terminalLog('error', 'Specification is invalid', errors);
                        if (Array.isArray(info) && info.length) {
                            NotificationHandler.terminalLog(
                                'error',
                                'Mismatched specification version',
                                `${info} Please refer to https://antmicro.github.io/kenning-pipeline-manager/changelogs.html to see if that is the case.`,
                            );
                        }
                    } else if (Array.isArray(info) && info.length) {
                        NotificationHandler.terminalLog('info', 'Specification loaded', info);
                    }
                });
        },
        createNewGraphCallback() {
            startTransaction();
            this.editorManager.editor.addNewGraph();
            commitTransaction();
        },
    },
    data() {
        const editorManager = EditorManager.getEditorManagerInstance();
        const externalApplicationManager = getExternalApplicationManager();
        return {
            editorManager,
            saveGraphConfiguration,
            externalApplicationManager,
        };
    },
};

</script>

<template>
    <div class="allbuttons" :class="{ hovered: hover }" >
        <Logo :hover="hover" />
        <div class="dropdown-wrapper">
            <template
                v-if="editorManager.specificationLoaded"
            >
                <DropdownItem
                    id="create-new-graph-button"
                    v-if="!readonly"
                    text="Create new graph"
                    type="'button'"
                    :eventFunction="createNewGraphCallback"
                />
                <template v-if="externalApp.backend && externalApp.connected">
                    <hr />
                    <DropdownItem
                        text="Load file"
                        id="request-dataflow-button"
                        :eventFunction="importDataflow"
                    />
                    <DropdownItem
                        text="Save file"
                        type="button"
                        :eventFunction="(async () => requestDataflowExport(false))"
                    />
                    <DropdownItem
                        text="Save file as..."
                        type="button"
                        :eventFunction="(async () => requestDataflowExport(true))"
                    />
                </template>
                <hr />
            </template>

            <template
                v-if="!externalApp.backend && !hideHud"
            >
                <DropdownItem
                    v-if="!editorManager.specificationLoaded"
                    text="Create empty specification"
                    id="create-spec-button"
                    type="'button'"
                    :eventFunction="createEmptySpecification"
                />
                <DropdownItem
                    text="Load specification"
                    id="load-spec-button"
                    :eventFunction="loadSpecificationCallback"
                />
                <DropdownItem
                    v-if="editorManager.specificationLoaded"
                    text="Save specification as..."
                    type="'button'"
                    :eventFunction="saveSpecificationCallback"
                />
                <hr />
            </template>

            <template v-if="editorManager.specificationLoaded">
                <DropdownItem
                    type="'button'"
                    text="Set graph name"
                    :eventFunction="setEditTitle"
                />
                <DropdownItem
                    v-if="!hideHud"
                    id="load-dataflow-button"
                    text="Load graph file"
                    :eventFunction="loadDataflowCallback"
                />
                <DropdownItem
                    type="'button'"
                    text="Save graph file"
                    :eventFunction="() => saveGraphConfiguration.saveCallback()"
                />
                <DropdownItem
                    type="'button'"
                    text="Save graph as file as..."
                    :eventFunction="saveGraphCallback"
                />
                <hr />
            </template>

            <DropdownItem
                type="'button'"
                text="Export graph to PNG"
                :eventFunction="exportGraphCallback"
            />
            <DropdownItem
                type="'button'"
                text="Export graph to HTML-based SVG"
                :eventFunction="exportToSvg"
            />
            <DropdownItem
                v-if="editorManager.baklavaView.settings.editableNodeTypes"
                type="'button'"
                text="Create node type from graph"
                :eventFunction="createGraphNodeTypeFromCurrentGraph"
            />
        </div>
    </div>
</template>
<style lang="scss" scoped>
.allbuttons {
    flex-grow: 0;
    display: flex;
    width: 3.75em;
    height: 3.75em;

    justify-content: center;
    align-items: center;
    position: relative;
    box-sizing: border-box;
    border-left: 1px solid $gray-500;
}
.hovered > .dropdown-wrapper {
    display: flex;
}
@import './simple_toggle_style.scss'
</style>
