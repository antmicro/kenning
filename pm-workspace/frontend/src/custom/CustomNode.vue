<!--
Copyright (c) 2022-2026 Antmicro <www.antmicro.com>

SPDX-License-Identifier: Apache-2.0
-->

<!--
The baklavajs functionality was fully copied to this component, as the original
implementation uses `<script setup>` which does not work well with `extends` feature.

Aside from the original source code, the custom node implements displaying name tags
for properties that do not display them, support for readonly mode that prevents the user
from moving or deleting the nodes.
 -->

<template>
    <div
        :id="node.id"
        ref="nodeRef"
        class="baklava-node"
        :data-node-id="node.id"
        :class="classes"
        :style="styles"
        :data-node-type="node.type"
        @pointerdown.left.exact="(ev) => select(ev)"
        @pointerdown.left="startDragWrapper($event)"
        oncontextmenu="return false;"
    >
        <div class="interface-cursor">
            <div class="line" :style="interfaceCursorStyle"></div>
        </div>
        <div
            class="__title"
            ref="titleRef"
            :style="nodeTitleStyle"
            @pointerdown.left.exact="onMouseDown"
            @pointerdown.right="openContextMenuTitle"
            v-long-press:500="openContextMenuTitle"
        >
            <img
                class="__title-icon"
                v-if="iconPath !== undefined"
                :src="iconPath"
            >
            <div
                v-if="!renaming"
                class="__title-label" v-html="DOMPurify.sanitize(nodeTitle)"
                ref="titleTextRef"
                :style="nodeTitleLabelStyle"
            >
            </div>
            <input
                v-else
                type="text"
                class="baklava-input"
                v-model="tempName"
                placeholder="Node Name"
                ref="renameField"
                v-click-outside="doneRenaming"
                @keydown="(ev) => ev.stopPropagation()"
                @keydown.enter="doneRenaming"
            />
            <template v-if="nodeStyle.name !== undefined &&
            nodeStyle.name !== null">
                <component
                    v-if="nodeStyle.icon !== undefined"
                    class="__title-icon"
                    :is="nodeStyle.icon"
                    :imgURI="nodeStyle.name"
                />
                <img
                    v-else
                    class="__title-icon"
                    :src="nodeStyle.name"
                />
            </template>
            <icons.Subgraph
                class="__subgraph-icon"
                :style="subgraphStyle"
                v-if="isGraphNode || nodeHasRelatedGraphs"
            />
            <div
                v-if="pillText !== undefined"
                class="pill"
                :style="nodePillStyle"
                v-html="DOMPurify.sanitize(pillText)"
            />
            <!-- disable transition to avoid rendering additional redraw for viewport adjustment -->

        </div>
        <svg
            v-if="customShape !== undefined"
            class="--shape"
            draggable="false"
            ref="svgRef"
            @pointerdown.left.exact="onMouseDown"
            @pointerdown.left="startDragWrapper($event)"
            @pointerdown.right="openContextMenuTitle"
            v-long-press:500="openContextMenuTitle"
            preserveAspectRatio="none"
        >
            <image :href="customShape" preserveAspectRatio="none" width="100%" height="100%"/>
        </svg>
        <!-- Positioned inputs -->
        <template v-for="input in positionedInterfaces">
            <CustomInterface
                :key="input.id"
                :ref="(e)=>positionedInterfaceElementSet.set(input.name,e)"
                v-if="input"
                @pointerdown.left.shift="pickInterface(input, $event)"
                @pointerdown.right="openContextMenuInterface(input, $event)"
                v-long-press-to-right:500
                positioned="true"
                :node="node"
                :intf="input"
                :highlighted="isHighlighted(input)"
                :picked="isPickedInterface(input)"
                :style="positionedInterfaceStyle(input)"
            />
            <!-- eslint-disable-next-line vue/require-v-for-key -->
        </template>

        <div
            class="__content"
            @pointerdown.right="openContextMenuTitle"
            v-if="customShape === undefined"
            >
            <!-- Properties -->
            <div class="__properties" ref="propertiesRef">
                <div
                    v-for="input in displayedProperties"
                    :key="input.id"
                    @pointerdown.right.exact="openContextMenuProperty(input, $event)"
                >
                    <CustomInterface
                        :node="node"
                        :intf="input"
                        :toggleGroup="toggleGroup"
                        :updateDynamicInterfaces="updateDynamicInterfaces"
                        @pointerdown.right.exact="openContextMenuProperty(input, $event)"
                    />
                </div>
            </div>

            <div class="__interfaces">
                <!-- Outputs -->
                <div class="__outputs" ref="rightSocketsRefs">
                    <template v-for="output in displayedRightRows">
                        <CustomInterface
                            :key="output.id"
                            v-if="output"
                            @pointerdown.left.shift="pickInterface(output, $event)"
                            @pointerdown.right.exact="openContextMenuInterface(output, $event)"
                            v-long-press-to-right:500
                            :node="node"
                            :intf="output"
                            :highlighted="isHighlighted(output)"
                            :picked="isPickedInterface(output)"
                            :switchSides="switchSides"
                        />
                        <!-- eslint-disable-next-line vue/require-v-for-key -->
                        <div v-else class="baklava-node-interface --output">&nbsp;</div>
                    </template>
                </div>

                <!-- Inputs -->
                <div class="__inputs" ref="leftSocketsRefs">
                    <template v-for="input in displayedLeftRows">
                        <CustomInterface
                            :key="input.id"
                            v-if="input"
                            @pointerdown.left.shift="pickInterface(input, $event)"
                            @pointerdown.right="openContextMenuInterface(input, $event)"
                            v-long-press-to-right:500
                            :node="node"
                            :intf="input"
                            :highlighted="isHighlighted(input)"
                            :picked="isPickedInterface(input)"
                            :switchSides="switchSides"
                        />
                        <!-- eslint-disable-next-line vue/require-v-for-key -->
                        <div v-else class="baklava-node-interface --input">&nbsp;</div>
                    </template>
                </div>
            </div>

        </div>
    </div>
</template>

<script setup>
/* eslint-disable object-curly-newline */
import { useMouseInElement } from '@vueuse/core';
import { ref, markRaw, computed, toRef, onUpdated, onMounted, nextTick, watch, useTemplateRef, reactive } from 'vue';
import { useViewModel, useGraph } from '@baklavajs/renderer-vue';
import { AbstractNode } from '@baklavajs/core';
import DOMPurify from 'dompurify';

import useGroupDragMove from './useGroupDragMove';
import CustomInterface from './CustomInterface.vue';
import { gridSnapper } from '../core/snappers';
import icons from '../icons/index';
import doubleClick from '../core/doubleClick.js';
import NotificationHandler from '../core/notifications.js';
import { updateInterfacePosition, removeNode } from './CustomNode.js';
import {
    startTransaction, commitTransaction,
} from '../core/History.ts';

import EditorManager, { DEFAULT_GRAPH_NODE_TYPE } from '../core/EditorManager';
import getExternalApplicationManager, { handleExternalAppResponse } from '../core/communication/ExternalApplicationManager';

import { saveGraphConfiguration } from '../components/saveConfiguration.ts';
import { configurationState, menuState } from '../core/nodeCreation/ConfigurationState.ts';
import { prepareNodeForDuplication } from '../core/nodeCreation/Configuration.ts';
import {
    updateSubgraphInterfaces,
    updateSubgraphProperties,
} from '../core/NodeFactory.js';

import notifyEvents from './notifyEvents.js';

import { checkForUnsavedEditorChangesWithToast } from './node_editor/NodeSpecEditorUtils.js';

// Baklavajs implementation

const props = defineProps({
    node: AbstractNode,
    selected: Boolean,
    greyedOut: Boolean,
    hidden: Boolean,
    interfaces: Array,
    ignoredInterfacesType: Array,
});

const emit = defineEmits(['select', 'openContextMenu', 'transformed', 'startDrag', 'stopDrag']);

const { viewModel } = useViewModel();
const { graph } = useGraph();
const movementStep = computed(() => viewModel.value.movementStep);

// Template refs
const svgRef = ref(null);
const nodeRef = ref(null);
const titleTextRef = ref(null);
const titleRef = ref(null);
const propertiesRef = useTemplateRef('propertiesRef');
const renaming = ref(false);
const renameField = ref(null);
const tempName = ref('');
const mouse = reactive(useMouseInElement(nodeRef));

// Reactive values
const node = toRef(props, 'node');
const nodeURLs = viewModel.value.editor.getNodeURLs(props.node.type);
const nodeColor = computed(() => viewModel.value.editor.getNodeColor(node.value));
const nodeTitleColor = computed(() => viewModel.value.editor.getTextColor(nodeColor.value));
const nodeCategory = viewModel.value.editor.getNodeCategory(props.node.type);
const isGraphNode = computed(() => props.node?.subgraph !== undefined);
const nodeHasRelatedGraphs
    = computed(() => viewModel.value.editor.nodeHasRelatedGraphs(node.value));
const pillText = computed(() => viewModel.value.editor.getPillText(node.value));
const pillColor = computed(() => viewModel.value.editor.getPillColor(node.value));
const pillTextColor = computed(() => viewModel.value.editor.getTextColor(pillColor.value));

const editorManager = EditorManager.getEditorManagerInstance();

const customShape = computed(() => {
    const shapeLink = viewModel.value.editor.getShape(node.value.type);
    if (shapeLink === undefined) {
        return undefined;
    }
    const cacheName = `./${shapeLink}`;

    if (Object.hasOwn(editorManager.baklavaView.cache, cacheName)) {
        return editorManager.baklavaView.cache[cacheName];
    }

    return shapeLink;
});

const displayNoResources = !viewModel.value.editor.nodeURLsEmpty();

const displayedInputs = computed(() => Object.values(props.node.inputs).filter((ni) => !ni.hidden));
const displayedOutputs = computed(() =>
    Object.values(props.node.outputs).filter((ni) => !ni.hidden),
);
const isBigBus = (intf) => (intf.port && intf.bus?.type === 'twoSided');
const bigBuses = computed(() =>
    [...Object.values(displayedInputs.value)
        .filter((intf) => isBigBus(intf)),
    ...Object.values(displayedOutputs.value)
        .filter((intf) => isBigBus(intf))]);
const sidebarProperties = computed(() =>
    [...Object.values(displayedInputs.value)
        .filter((intf) => !intf.port),
    ...bigBuses.value],
);
const displayedProperties = computed(() => {
    if (editorManager.baklavaView.settings.showHiddenProperties) {
        return sidebarProperties.value;
    }
    return sidebarProperties.value
        .filter((intf) => !(intf.hideOnDefault && (!intf.value || intf.value === intf.default)));
});

const transformed = () => {
    emit('transformed', props.node.id);
};

const externalApplicationManager = getExternalApplicationManager();
// Watch properties
Object.entries(props.node.inputs).forEach(([name, input]) => {
    if (name.startsWith('property_')) {
        let firstWatch = true;
        watch(input, async (value) => {
            if (!externalApplicationManager.isConnected()) {
                firstWatch = true;
                return;
            }
            if (firstWatch || !editorManager.notifyWhenChanged) {
                firstWatch = false;
                return;
            }
            const data = {
                graph_id: props.node.graphInstance.id,
                node_id: props.node.id,
                properties: [],
            };
            data.properties.push({
                id: value.id,
                new_value: value.value,
            });
            await externalApplicationManager.notifyAboutChange('properties_on_change', data);
        });
    }
});
// Watch interfaces
[...Object.entries(props.node.inputs), ...Object.entries(props.node.outputs)]
    .forEach(([name, intf]) => {
        if (!name.startsWith('property_')) {
            let firstWatch = true;
            watch(intf, async (value) => {
                if (!externalApplicationManager.isConnected()) {
                    firstWatch = true;
                    return;
                }
                if (firstWatch || !editorManager.notifyWhenChanged) {
                    firstWatch = false;
                    return;
                }
                const data = {
                    graph_id: props.node.graphInstance.id,
                    node_id: props.node.id,
                    interfaces: [],
                };
                data.interfaces.push({
                    id: value.id,
                    externalName: value.externalName,
                });
                await externalApplicationManager.notifyAboutChange('interfaces_on_change', data);
            });
        }
    });

// Send message about changed position
const notifyPositionChanged = (position) => {
    nextTick(() => transformed());
    if (!externalApplicationManager.isConnected()) return;
    externalApplicationManager.notifyAboutChange('position_on_change', {
        graph_id: props.node.graphInstance.id,
        node_id: props.node.id,
        position: {
            x: position.x,
            y: position.y,
        },
    });
};
// Create watcher for position
watch(() => props.node.position, notifyPositionChanged);
watch(() => props.node.title, async (value) => {
    const data = {
        graph_id: props.node.graphInstance.id,
        node_id: props.node.id,
        name: value,
    };
    await externalApplicationManager.notifyAboutChange('name_on_change', data);
});

const focusOnRename = () => {
    renameField.value.focus();
    renameField.value.select();
};

// Title context menu
const showContextMenuTitle = ref(false);
const contextMenuTitleX = ref(0);
const contextMenuTitleY = ref(0);
const contextMenuInterfaceItems = ref([]);

const contextMenuTitleItems = computed(() => {
    const items = [];
    items.push({ value: 'sidebar', label: 'Details', icon: icons.Sidebar, endSection: true });
    if (editorManager.baklavaView.settings.editableNodeTypes &&
        node.value.type !== DEFAULT_GRAPH_NODE_TYPE &&
        !viewModel.value.editor.readonly &&
        !(props.node.subgraph && props.node.extending?.length)) {
        items.push(
            { value: 'configure', label: 'Configure' },
            { value: 'property', label: 'Add property' },
            { value: 'interface', label: 'Add interface' },
            { value: 'layer', label: 'Set layer' },
            { value: 'delete-property', label: 'Delete property' },
            { value: 'delete-interface', label: 'Delete interface' },
            { value: 'duplicate', label: 'Duplicate node type', endSection: true },
        );
        if (!isGraphNode.value && !editorManager.editor.preview) {
            items.push(
                { value: 'addSubgraph', label: 'Add subgraph', icon: icons.Subgraph },
                { value: 'addAndEditSubgraph', label: 'Add and edit subgraph', icon: icons.Subgraph },
            );
        }
    }
    if (isGraphNode.value) {
        items.push({ value: 'editSubgraph', label: 'Go to graph', icon: icons.Subgraph });
    }
    if (nodeHasRelatedGraphs.value) {
        const nodeInstance = node.value;
        const nodeType = viewModel.value.editor.nodeTypes.get(props.node.type);
        const toEntries = (graphs) => graphs?.map(({ id, name }) => [id, name]) ?? [];

        // Merge related graphs from node type and instance
        const relatedGraphs = Object.entries({
            ...Object.fromEntries(toEntries(nodeType.relatedGraphs)),
            ...Object.fromEntries(toEntries(nodeInstance.relatedGraphs)),
        });

        relatedGraphs.forEach(([id, name]) => {
            items.push({ value: `gotoRelatedGraph ${id}`, label: `Go to ${name} graph`, icon: icons.Subgraph });
        });
    }
    if (items.length > 1) {
        items.at(-1).endSection = true;
    }

    items.push(...nodeURLs);

    const customContextMenuActions = [
        ...(viewModel.value.contextMenuActions ?? []),
        ...viewModel.value.editor.getNodeContextMenuActions(node.value.type),
        // Add subgraph node actions if the node is a subgraph node
        ...(isGraphNode.value
            ? (viewModel.value.subgraphContextMenuActions ?? [])
            : []),
    ];
    if (customContextMenuActions.length > 0) {
        if (items.length > 0) {
            items.at(-1).endSection = true;
        }
        customContextMenuActions.forEach((customAction) => {
            items.push({
                value: `custom:${customAction.procedureName}`,
                label: customAction.name,
                icon: customAction.iconName ? icons[customAction.iconName] : undefined,
            });
        });
    }

    if (!viewModel.value.editor.readonly) {
        if (items.length > 1) {
            items.at(-1).endSection = true;
        }
        items.push(
            { value: 'rename', label: 'Rename', icon: icons.Pencil },
            { value: 'disconnect', label: 'Disconnect', icon: icons.Disconnect },
            { value: 'delete', label: 'Delete', icon: icons.Bin },
        );
        items.at(-1).endSection = true;
        if (graph.value.selectedNodes.length > 1) {
            items.push({ value: 'groupNodes', label: 'Group Nodes' });
        }
        const groups = (graph.value.groups ?? []).filter((g) => g.nodes.includes(node.value.id));
        if (groups.length > 0) {
            items.push({ value: 'ungroupNodes', label: 'Remove from group' });
        }
    }

    // NOTE: This feature is disabled for now, as it is not fully implemented
    const SUPPORT_NODE_UNWRAPPING = false;
    if (SUPPORT_NODE_UNWRAPPING) {
        if (isGraphNode.value) {
            items.push({ value: 'unwrap', label: 'Unwrap Subgraph', icon: icons.Unwrap });
        }
    }

    return items;
});

const openSidebar = () => {
    const { sidebar } = viewModel.value.displayedGraph;
    sidebar.nodeId = props.node.id;
    sidebar.visible = true;
};

const getSubgraphInterfaces = () => {
    if (!node.value.subgraph) {
        return [];
    }
    const evaluatedIntf = updateSubgraphInterfaces(
        node.value.subgraph.nodes,
        Object.values(node.value.inputs),
        Object.values(node.value.outputs),
    );
    if (Array.isArray(evaluatedIntf) && evaluatedIntf.length) {
        throw new Error(
            `Internal error occurred while getting subgraph interfaces.\n` +
            `Reason: ${evaluatedIntf.join('. ')}`,
        );
    }
    return [...evaluatedIntf.inputs, ...evaluatedIntf.outputs];
};

const getSubgraphProperties = () => {
    if (!node.value.subgraph) {
        return [];
    }
    const evaluatedProp = updateSubgraphProperties(
        node.value.subgraph.nodes,
        Object.values(node.value.inputs),
    );
    return evaluatedProp;
};

const runCustomContextMenuAction = async (procedureName) => {
    const actionDef = [
        ...(viewModel.value.contextMenuActions ?? []),
        ...viewModel.value.editor.getNodeContextMenuActions(node.value.type),
        // Add subgraph node actions if the node is a subgraph node
        ...(isGraphNode.value
            ? (viewModel.value.subgraphContextMenuActions ?? [])
            : []),
    ].find((customAction) => customAction.procedureName === procedureName);

    const params = { node_id: props.node.id };
    if (actionDef?.requireResponse ?? true) {
        const response = await externalApplicationManager.request(procedureName, params);
        // if the response contains a 'filename' field, save 'content' to a file named 'filename'
        if (response?.filename !== undefined) {
            const saveConfiguration = { ...saveGraphConfiguration };
            saveConfiguration.saveName = response.filename;
            saveConfiguration.saveCallbackCustomFormat(
                response.content,
            );
        }
        handleExternalAppResponse(response);
    } else {
        externalApplicationManager.request(procedureName, params);
    }
};

/* eslint-disable default-case */
const onContextMenuTitleClick = async (action) => {
    if (action.startsWith('custom:')) {
        await runCustomContextMenuAction(action.replace(/^custom:/, ''));
        return;
    }

    if (action !== 'delete' && action !== 'disconnect') {
        const nodeRect = nodeRef.value.getBoundingClientRect();
        const nodeData = {
            name: props.node.type,
            category: nodeCategory,
            layer: props.node.layer,
            color: nodeColor,
            isLayerInherited: props.node.simpleInherited?.includes('layer'),
            isCategoryInherited: props.node.simpleInherited?.includes('category'),
            isShaped: customShape.value !== undefined,
        };

        configurationState.editedType = nodeData.name;
        configurationState.nodeData = nodeData;
        configurationState.nodeRect = {
            x: nodeRect.x,
            y: nodeRect.y,
            width: nodeRect.width,
            height: nodeRect.height,
        };

        let nodeInterfaces = [...displayedInputs.value, ...displayedOutputs.value];
        nodeInterfaces = nodeInterfaces.filter((intf) => intf.direction !== undefined);
        const subInterfaces = getSubgraphInterfaces();
        const subProps = getSubgraphProperties();

        const configuredInterfaces = nodeInterfaces?.map((intf) => ({
            name: intf?.name,
            type: intf?.type,
            side: intf?.side,
            direction: intf?.direction,
            maxConnectionsCount: intf?.maxConnectionsCount,
            inSubgraph: subInterfaces.some((i) => i.id === intf.id),
            inherited: intf?.inherited,
            override: intf?.override,
            bus: intf?.bus ? {
                type: intf.bus.type,
                size: intf.bus.size,
            } : undefined,
        }));

        /* eslint-disable no-underscore-dangle */
        const properties = sidebarProperties.value.filter((prop) => !isBigBus(prop));
        const configuredProperties = properties?.map((prop) => ({
            name: prop?.name,
            type: prop?.type,
            default: prop?._value,
            min: prop?.min,
            max: prop?.max,
            values: prop?.items,
            step: prop?.step,
            readonly: prop?.readonly,
            dtype: prop?.dtype,
            inSubgraph: subProps.some((i) => i.id === prop.id),
            inherited: prop?.inherited,
            override: prop?.override,
        }));

        configurationState.properties = configuredProperties;
        configurationState.interfaces = configuredInterfaces;
    }

    if (action.includes('gotoRelatedGraph')) {
        viewModel.value.editor.switchToRelatedGraph(action.replace(/^gotoRelatedGraph /, ''));
    }

    if (action === 'duplicate') {
        prepareNodeForDuplication(props.node.title);
    }

    if (await checkForUnsavedEditorChangesWithToast()) {
        return;
    }

    switch (action) {
        case 'rename':
            tempName.value = props.node.title;
            renaming.value = true;
            await nextTick();
            focusOnRename();
            break;
        case 'disconnect': {
            startTransaction();
            let interfaces = [...displayedInputs.value, ...displayedOutputs.value];
            graph.value.selectedNodes.forEach((n) => {
                interfaces = interfaces.concat(
                    Object.entries(n.inputs).filter(([name, ni]) => !ni.hidden && !name.startsWith('property_')).map(([, ni]) => ni),
                    Object.values(n.outputs).filter((ni) => !ni.hidden),
                );
            });
            const nodeConnections = graph.value.connections.filter(
                (c) =>
                    (interfaces.find((i) => i === c.from || i === c.to) !== undefined),
            );
            nodeConnections.forEach((c) => {
                graph.value.removeConnection(c);
            });
            commitTransaction();
        } break;
        case 'sidebar':
            openSidebar();
            break;
        case 'editSubgraph': {
            const errors = viewModel.value.editor.switchToSubgraph(props.node);
            if (Array.isArray(errors) && errors.length) {
                NotificationHandler.terminalLog('error', 'Dataflow is invalid', errors);
            }
            if (props.node.extending?.length) {
                NotificationHandler.terminalLog('warning', 'Subgraph restricted', 'Editing extending subgraph is not currently supported');
            }
            break;
        }
        case 'unwrap':
            startTransaction();
            removeNode(props.node, true);
            commitTransaction();
            break;
        case 'duplicate':
            menuState.configurationMenu.visible = true;
            menuState.configurationMenu.addNode = false;
            menuState.configurationMenu.duplicateNode = true;
            configurationState.nodeData.name += ' (copy)';
            break;
        case 'configure':
            menuState.configurationMenu.visible = true;
            menuState.configurationMenu.addNode = false;
            menuState.configurationMenu.duplicateNode = false;
            break;
        case 'property':
            menuState.propertyMenu = true;
            break;
        case 'interface':
            menuState.interfaceMenu = true;
            break;
        case 'layer':
            menuState.layerMenu = true;
            break;
        case 'addSubgraph': {
            const errors = editorManager.addSubgraphToNode(props.node);
            if (Array.isArray(errors) && errors.length) {
                NotificationHandler.terminalLog('error', 'Creating subgraph failed', errors);
            }
            break;
        }
        case 'addAndEditSubgraph': {
            let errors = editorManager.addSubgraphToNode(props.node);
            let newNode;
            if (Array.isArray(errors) && errors.length) {
                NotificationHandler.terminalLog('error', 'Creating subgraph failed', errors);
            } else {
                newNode = errors;
            }
            await nextTick();
            errors = viewModel.value.editor.switchToSubgraph(newNode);
            if (Array.isArray(errors) && errors.length) {
                NotificationHandler.terminalLog('error', 'Switching to subgraph failed', errors);
            }
            if (newNode.extending?.length) {
                NotificationHandler.terminalLog('warning', 'Subgraph restricted', 'Editing extending subgraph is not currently supported');
            }
            break;
        }
        case 'delete-property':
            menuState.propertyListMenu = true;
            break;
        case 'delete-interface':
            menuState.interfaceListMenu = true;
            break;
        case 'groupNodes':
            menuState.groupMenu = true;
            break;
        case 'delete':
            startTransaction();
            if (!graph.value.selectedNodes.includes(props.node)) {
                graph.value.selectedNodes.push(props.node);
            }
            graph.value.selectedNodes.forEach((n) => removeNode(n));
            graph.value.selectedNodes = [];
            commitTransaction();
            break;
        // eslint-disable-next-line no-fallthrough
        case 'ungroupNodes':
            if (!graph.value.selectedNodes.includes(props.node)) {
                graph.value.selectedNodes.push(props.node);
            }
            graph.value.selectedNodes.forEach((n) => graph.value.ungroupNode(n));
            break;
    }
    transformed();
};

const canOpenContextMenu = computed(() =>
    (contextMenuTitleItems.value.length === 0 && displayNoResources)
        || contextMenuTitleItems.value.length > 0,
);

const interfacePositions = computed(() => {
    const positionMap = new Map();

    const interfaces = viewModel.value.editor.getNodeInterfacePositions(node.value.type);

    Object.entries(interfaces).forEach(([key, value]) => {
        const x = Math.max(Math.min(value.x, 100), 0) / 100.0;
        const y = Math.max(Math.min(value.y, 100), 0) / 100.0;

        positionMap.set(key, {
            x,
            y,
        });
    });

    return positionMap;
});

const nodeMinimal = computed(() => viewModel.value.editor.getNodeMinimal(node.value));
const nodeClean = computed(() => viewModel.value.editor.getNodeClean(node.value));

const showContextMenuInterface = ref(false);
const showContextMenuProperty = ref(false);

const groupDragMove = useGroupDragMove(
    toRef(props.node, 'position'),
    props.node.id,
    gridSnapper(movementStep),
);

const classes = computed(() => ({
    '--selected': props.selected,
    '--dragging': groupDragMove.dragging.value,
    '--two-column': !!props.node.twoColumn,
    '--greyed-out': props.greyedOut,
    '--hidden': props.hidden,
    '--minimal': nodeMinimal.value,
    '--transparent': customShape.value !== undefined,
    '--shaped': customShape.value !== undefined,
    '--clean': nodeClean.value,
    __readonly: viewModel.value.editor.readonly,
}));

const subgraphStyle = computed(() => {
    if (customShape.value !== undefined) {
        return {
            position: 'absolute',
            left: '-14px',
            top: '-20px',
        };
    }

    return {};
});

const hideSubTitle = computed(() => viewModel.value.editor
    .getNodeStyleHideSubTitle(props.node.type));

const nodeTitle = computed(() => {
    const title = props.node.highlightedTitle ?? props.node.title;
    const type = props.node.highlightedType ?? props.node.type;

    if (title === '' || props.node.title === props.node.type) {
        return type;
    }

    if (hideSubTitle.value) {
        return `${title}`;
    }

    let styles = '';

    if (props.node.width !== 0 && props.node.width !== undefined) {
        styles = 'overflow: hidden; text-overflow: ellipsis;';
    }

    return `${title} <pre class="subtitle" style="${styles}">${type}</pre>`;
});

const select = (event) => {
    emit('select', event);
};
const openContextMenu = (isOpened, x, y, items, ignoreClose, onClick,
    urls = undefined, style = undefined, onClose = () => {}) => {
    emit('openContextMenu', isOpened, x, y, items, ignoreClose, onClick, urls, style, onClose);
};

let abortDrag;
let stopDrag;

const cleanEvents = () => {
    document.removeEventListener('pointermove', groupDragMove.onPointerMove);
    document.removeEventListener('pointermove', transformed);
    document.removeEventListener('keyboard.escape', abortDrag);
    document.removeEventListener('pointerup', stopDrag);
};

abortDrag = () => {
    cleanEvents();
};

stopDrag = () => {
    graph.value.refreshConnections();
    groupDragMove.onPointerUp();
    cleanEvents();
    transformed();
    emit('stopDrag');
};

const startDrag = async (ev) => {
    if (!graph.value.selectedNodes.includes(props.node)) {
        select(ev);
    }
    groupDragMove.onPointerDown(ev);
    document.addEventListener('pointermove', groupDragMove.onPointerMove);
    document.addEventListener('pointermove', transformed);
    document.addEventListener('keyboard.escape', abortDrag);
    document.addEventListener('pointerup', stopDrag);
    transformed();
    emit('startDrag');
};

const doneRenaming = () => {
    const curNode = graph.value.findNodeById(props.node.id);
    graph.value.editNode(curNode);
    curNode.title = tempName.value;
    renaming.value = false;
    transformed();
};

const onRender = () => {
    if (nodeRef.value) {
        viewModel.value.hooks.renderNode.execute({ node: props.node, el: nodeRef.value });
    }
};
onMounted(onRender);
onUpdated(onRender);

// ----------

/**
 * Wrapper that prevents node moving if the editor is in read-only mode.
 *
 * @param ev Event
 */
const startDragWrapper = (ev) => {
    if (ev.target.closest('[no-drag="true"]')) return;
    if (!viewModel.value.editor.readonly) {
        startDrag(ev);
    }
};

const openDoubleClick = doubleClick(700, () => {
    openSidebar();
});

/* eslint-disable vue/no-mutating-props,no-param-reassign */
const onMouseDown = async () => {
    if (await checkForUnsavedEditorChangesWithToast()) {
        return;
    }
    const { sidebar } = viewModel.value.displayedGraph;
    if (sidebar.visible) {
        sidebar.nodeId = props.node.id;
    }

    openDoubleClick();
};

const positionedInterfaces = computed(() => {
    if (!nodeRef.value) {
        return [];
    }

    return Object.values([...displayedInputs.value, ...displayedOutputs.value])
        .filter((intf) => interfacePositions.value.has(intf.name))
        .filter((intf) => !intf.type?.some((t) => props.ignoredInterfacesType?.includes(t)));
},
);

const filterIntfs = (intfs, side) =>
    intfs.filter((intf) => intf.side === side && intf.port)
        .filter((intf) => !interfacePositions.value.has(intf.name))
        .filter((intf) => !intf.type?.some((t) => props.ignoredInterfacesType?.includes(t)))
        .filter((intf) => !isBigBus(intf))
        .sort((intf1, intf2) => intf1.sidePosition - intf2.sidePosition);

const displayedLeftSockets = computed(() =>
    filterIntfs(Object.values([...displayedInputs.value, ...displayedOutputs.value]), 'left'),
);

const displayedRightSockets = computed(() =>
    filterIntfs(Object.values([...displayedInputs.value, ...displayedOutputs.value]), 'right'),
);
const getRows = (sockets) => {
    if (!sockets.length) {
        return [];
    }

    const positionArr = sockets.map((s) => s.sidePosition ?? 0);

    const numOfLines = Math.max(...positionArr);

    let numOfSocket = 0;
    const rows = [];

    for (let i = 0; i <= numOfLines; i += 1) {
        if (sockets[numOfSocket]?.sidePosition === i) {
            rows.push(sockets[numOfSocket]);
            numOfSocket += 1;
        } else {
            rows.push(undefined);
        }
    }
    return rows;
};

const customTitlePadding = computed(() => viewModel.value.editor
    .getNodeStyleTitlePadding(node.value.type) ?? {
    x: 0,
    y: 0,
});

const fitTitle = computed(() => viewModel.value.editor.getNodeStyleFitTitle(props.node.type));

const titleSize = computed(() => {
    const width = (titleTextRef?.value?.offsetWidth ?? 0);
    const height = (titleTextRef?.value?.offsetHeight ?? 0);
    return {
        width,
        height,
    };
});

const minimalWidth = computed(() => {
    // check title size
    const getTitleSize = () => {
        if (props.node.width > 0) {
            return props.node.width;
        }
        if (titleTextRef.value === null || titleRef.value === null) {
            return 0;
        }
        const titleStyle = getComputedStyle(titleRef.value);
        const paddingLeft = Number.parseInt(titleStyle?.paddingLeft ?? 0, 10);
        const paddingRight = Number.parseInt(titleStyle?.paddingRight ?? 0, 10);

        return titleSize.value.width + paddingLeft + paddingRight;
    };
    const titleWidth = getTitleSize();

    const fontSize = 9;

    // search for the longest word, allow wrapping of words
    const leftNames = displayedLeftSockets.value.flatMap((sock) =>
        sock.name.split(' ')).map((s) => s.length);
    const rightNames = displayedRightSockets.value.flatMap((sock) =>
        sock.name.split(' ')).map((s) => s.length);

    const maxLeftTextLength = Math.max(
        ...leftNames, 0,
    );
    const maxRightTextLength = Math.max(
        ...rightNames, 0,
    );

    const LeftInterfaceSize = fontSize * 2 + maxLeftTextLength * fontSize;
    const RightInterfaceSize = fontSize * 2 + maxRightTextLength * fontSize;

    if (props.node.twoColumn) {
        const nodeRequiredSize = 2 * Math.max(LeftInterfaceSize, RightInterfaceSize);
        return Math.max(nodeRequiredSize, titleWidth);
    }
    const nodeRequiredSize = Math.max(LeftInterfaceSize, RightInterfaceSize);

    return Math.max(nodeRequiredSize, titleWidth);
});

const width = computed(() => {
    if (fitTitle.value) {
        return `${titleSize.value.width + customTitlePadding.value.x}px`;
    }
    if (props.node.width !== undefined) {
        if (props.node.width < minimalWidth.value) {
            return `${minimalWidth.value}px`;
        }
        return `${props.node.width}px`;
    }
    if (nodeMinimal.value || nodeClean.value) {
        return 'auto';
    }
    return '300px';
});

const height = computed(() => {
    if (fitTitle.value) {
        return `${titleSize.value.height + customTitlePadding.value.y}px`;
    }
    if (props.node.height !== undefined) {
        return `${props.node.height}px`;
    }
    return 'auto';
});

const styles = computed(() => ({
    top: `${props.node.position?.y ?? 0}px`,
    left: `${props.node.position?.x ?? 0}px`,
    'min-width': fitTitle.value ? '0' : undefined,
    'min-height': fitTitle.value ? '0' : undefined,
    width: width.value,
    height: height.value,
    display: customShape.value === undefined ? 'inherit' : 'block',
    cursor: menuState.addingPositionedInterface ? 'default' : undefined,
}));

// another potential source of issue
const displayedRightRows = computed(() => getRows(displayedRightSockets.value));
const displayedLeftRows = computed(() => getRows(displayedLeftSockets.value));

watch(displayedProperties, () => {
    displayedProperties.value
        .filter((prop) => prop.component === undefined || prop.defaultComponent)
        .filter((prop) => prop.setDefaultComponent !== undefined)
        .forEach((prop) => { prop.setDefaultComponent(); });
}, { immediate: true });

const path = viewModel.value.editor.getNodeIconPath(props.node.type);
const iconPath = viewModel.value.cache[`./${path}`] ?? path;

// Metadata ("icon" is not a string) > Predefined > Cached > Arbitrary URI
const nodeStyle = computed(() => {
    let icon;
    let name = viewModel.value.editor.getStyleIcon(node.value.type);

    if (typeof name === 'object' && name !== null) {
        name = editorManager.getMetadataIcon(name);
    }
    if (icons[name] !== undefined) {
        icon = icons[name];
    } else if (viewModel.value.cache[`./${name}`] !== undefined) {
        icon = icons.Placeholder;
        name = viewModel.value.cache[`./${name}`];
    }

    return { icon, name };
});

// Interface modification

let newSocketIndex;
let newSocketSide = null;
const chosenInterface = ref(undefined);
let chosenProperty;

const leftSocketsRefs = ref(null);
const rightSocketsRefs = ref(null);

const interfaceCursorStyle = ref({
    top: '0px',
    left: '0px',
    display: 'none',
});

const customTitleOffset = computed(() => viewModel.value.editor
    .getNodeStyleTitleOffset(node.value.type) ?? {
    x: 0,
    y: 0,
});

const customShapeTitlePosition = computed(() => {
    const titlePos = viewModel.value.editor.getCustomNodeTitlePosition(node.value.type);

    if (titlePos === undefined) {
        return undefined;
    }

    const x = titlePos?.x ?? 0.0;
    const y = titlePos?.y ?? 0.0;

    return {
        x,
        y,
    };
});

const nodeTitleLabelStyle = computed(() => {
    if (props.node.width !== 0 && props.node.width !== undefined) {
        return {
            // maxWidth: width.value
            overflow: 'hidden',
            textOverflow: 'ellipsis',
        };
    }

    return {};
});

const nodeTitleStyle = computed(() => {
    const style = {
        cursor: 'default',
        backgroundColor: nodeColor.value,
        color: nodeTitleColor.value,
    };

    const position = customShapeTitlePosition.value;

    if (position !== undefined && customShape.value !== undefined) {
        style.position = 'absolute';
        style.left = `calc(${position.x}% + ${customTitleOffset.value.x}px)`;
        style.top = `calc(${position.y}% + ${customTitleOffset.value.y}px)`;
        style.display = 'block';
        style.backgroundColor = 'transparent';
        style.width = 'fit-content';
        style.height = 'fit-content';
        style.padding = 0;
    }

    if (!viewModel.value.editor.readonly) {
        style.cursor = 'drag';
        return style;
    }

    if (canOpenContextMenu.value) {
        style.cursor = 'pointer';
        return style;
    }

    return style;
});

const nodePillStyle = computed(() => ({
    cursor: 'default',
    backgroundColor: pillColor.value,
    color: pillTextColor.value,
}));

const positionedInterfaceElementSet = ref(new Map());

const positionedInterfaceStyle = (inf) => {
    if (!nodeRef.value) {
        return {};
    }
    const infName = inf.name;

    const posMap = interfacePositions.value;

    if (!posMap?.has(infName)) {
        return {};
    }

    const infRef = positionedInterfaceElementSet.value.get(infName);

    if (infRef === null) {
        return {};
    }

    let offsetX = 0;
    let offsetY = 0;

    if (infRef !== undefined) {
        if (infRef.el !== null) {
            offsetX = infRef.el.offsetWidth;
            offsetY = infRef.el.offsetHeight;
        }
    }

    const positions = posMap.get(infName);

    const infX = positions?.x ?? 0;
    const infY = positions?.y ?? 0;

    const x = infX * 100.0;
    const y = infY * 100.0;

    return {
        position: 'absolute',
        left: `calc(${x}% - ${offsetX}px/2)`,
        top: `calc(${y}% - ${offsetY}px/2)`,
    };
};

const isPickedInterface = (intf) => intf === chosenInterface.value;

const assignNewPosition = () => {
    if (newSocketSide === null) {
        newSocketSide = chosenInterface.value.side;
    }
    updateInterfacePosition(
        props.node,
        chosenInterface.value,
        newSocketSide,
        newSocketIndex,
        true,
    );
    newSocketSide = null;
};

const interfaceDragThreshold = ref(null);
const dragInterface = (ev) => {
    if (interfaceDragThreshold.value === null) {
        const rect = nodeRef.value.getBoundingClientRect();
        interfaceDragThreshold.value = rect.left + rect.width / 2;
    }

    let sockets;
    let otherSockets;
    if (interfaceDragThreshold.value > ev.clientX) {
        sockets = leftSocketsRefs.value;
        otherSockets = rightSocketsRefs.value;
        newSocketSide = 'left';
    } else if (interfaceDragThreshold.value < ev.clientX) {
        sockets = rightSocketsRefs.value;
        otherSockets = leftSocketsRefs.value;
        newSocketSide = 'right';
    }

    // Finding the first interface that is lower than the cursor
    let socket = [...sockets.children].findIndex((socketRef) => {
        const boundingRect = socketRef.getBoundingClientRect();
        return boundingRect.bottom > ev.clientY;
    });
    newSocketIndex = socket;

    if (socket === -1) {
        socket = sockets.children.length - 1;
        newSocketIndex = sockets.children.length - 1;
        let el = sockets.children[socket];
        let bottom = 0;
        if (el === undefined) {
            // eslint-disable-next-line prefer-destructuring
            el = otherSockets.children[0];
            newSocketIndex = 0;
        }
        // eslint-disable-next-line prefer-destructuring
        bottom = el.getBoundingClientRect().bottom;
        const style = window.getComputedStyle(el);
        const marginTop = parseFloat(style.marginTop) || 0;
        const marginBottom = parseFloat(style.marginBottom) || 0;
        const totalHeight = el.getBoundingClientRect().height + marginTop + marginBottom;
        let offset = 0;
        let idxAdded = 0;

        while (bottom + offset < ev.clientY) {
            offset += totalHeight;
            newSocketIndex += 1;
            idxAdded += 1;
        }
        interfaceCursorStyle.value = {
            width: '5em', top: `${el.offsetTop + (el.offsetHeight * idxAdded) + el.offsetHeight / 2 - 2.5}px`, position: 'absolute', display: 'block',
        };
    } else {
        const el = sockets.children[socket];
        interfaceCursorStyle.value = {
            top: `${el.offsetTop + el.offsetHeight / 2 - 2.5}px`, display: 'block',
        };
    }

    if (newSocketSide === 'right') {
        interfaceCursorStyle.value.right = '-0.7em';
    } else if (newSocketSide === 'left') {
        interfaceCursorStyle.value.left = '-0.7em';
    }
};

const dropInterface = () => {
    interfaceDragThreshold.value = null;
    assignNewPosition();

    chosenInterface.value = undefined;
    interfaceCursorStyle.value = {
        top: '0px',
        left: '0px',
        right: '0px',
        display: 'none',
    };

    document.removeEventListener('pointermove', dragInterface);
    document.removeEventListener('pointerup', dropInterface);
};

const pickInterface = (intf, ev) => {
    if (viewModel.value.editor.readonly) {
        return;
    }
    chosenInterface.value = intf;
    dragInterface(ev);

    document.addEventListener('pointermove', dragInterface);
    document.addEventListener('pointerup', dropInterface);
};

// Interface context menu

const contextMenuInterfaceSide = ref('left');
const contextMenuInterfaceX = ref(0);
const contextMenuInterfaceY = ref(0);

const createContextMenuInterfaceItems = () => {
    const items = [];
    const posMap = interfacePositions.value;

    if (chosenInterface.value !== undefined && !viewModel.value.settings.disableInterfaceExpose) {
        const intfMode = (chosenInterface.value.externalName === undefined ?
            { value: 'SetExternalName', label: 'Expose Interface', icon: icons.Subgraph } :
            { value: 'UnsetExternalName', label: 'Privatize Interface', icon: icons.Subgraph }
        );
        items.push(intfMode);
    }

    if (!posMap.has(chosenInterface.value.name)) {
        items.push(
            { value: 'SpaceUp', label: 'Space Up' },
            { value: 'SpaceDown', label: 'Space Down' },
            { value: 'MoveUp', label: 'Move Up' },
            { value: 'MoveDown', label: 'Move Down' },
        );

        if (chosenInterface.value !== undefined && chosenInterface.value.side) {
            switch (chosenInterface.value.side) {
                case 'left':
                    items.push({ value: 'MoveRight', label: 'Move Right' });
                    break;
                case 'right':
                    items.push({ value: 'MoveLeft', label: 'Move Left' });
                    break;
            }
        }
    }

    return items;
};

/* eslint-disable default-case */
const onContextMenuInterfaceClick = (action) => {
    switch (action) {
        case 'SetExternalName':
            viewModel.value.editor.exposeInterface(
                graph.value.id,
                chosenInterface.value,
            );

            notifyEvents.exposedInterface.emit([chosenInterface.value, graph.value.id, true]);
            graph.value.events.exposeInterface
                .emit([chosenInterface.value, viewModel.value.editor]);
            break;
        case 'UnsetExternalName':
            graph.value.events.privatizeInterface
                .emit([chosenInterface.value, viewModel.value.editor]);
            viewModel.value.editor.privatizeInterface(
                graph.value.id,
                chosenInterface.value,
            );

            notifyEvents.exposedInterface.emit([chosenInterface.value, graph.value.id, false]);
            break;
        case 'MoveUp':
            if (chosenInterface.value.sidePosition === 0) {
                chosenInterface.value = undefined;
                break;
            }
            newSocketIndex = chosenInterface.value.sidePosition - 1;
            dropInterface();
            break;
        case 'MoveDown':
            newSocketIndex = chosenInterface.value.sidePosition + 1;
            dropInterface();
            break;
        case 'SpaceUp':
        case 'SpaceDown': {
            const sockets =
                chosenInterface.value.side === 'right'
                    ? displayedRightRows.value
                    : displayedLeftRows.value;
            const comparison = action === 'SpaceDown' ? (a, b) => a > b : (a, b) => a >= b;
            Object.values(sockets).forEach((intf) => {
                if (intf !== undefined &&
                comparison(intf.sidePosition, chosenInterface.value.sidePosition)) {
                    intf.sidePosition += 1; // eslint-disable-line no-param-reassign
                }
            });
            break;
        }
        case 'MoveLeft':
            chosenInterface.value.side = 'left';
            break;
        case 'MoveRight':
            chosenInterface.value.side = 'right';
            break;
    }
};

const openContextMenuInterface = async (intf, ev) => {
    showContextMenuInterface.value = false;
    await nextTick();
    if (!viewModel.value.editor.readonly) {
        chosenInterface.value = intf;
        const interfaceName = intf.name;
        contextMenuInterfaceItems.value = createContextMenuInterfaceItems();
        const targetRect = ev.currentTarget.getBoundingClientRect();
        const nodeRect = nodeRef.value.getBoundingClientRect();
        const posMap = interfacePositions.value;

        if (posMap.has(interfaceName)) {
            contextMenuInterfaceX.value = ev.clientX + 10;
            contextMenuInterfaceY.value = ev.clientY + 12.5;
        } else if (chosenInterface.value.side === 'right') {
            contextMenuInterfaceSide.value = 'right';
            contextMenuInterfaceX.value = nodeRect.left + nodeRect.width + 10;
            contextMenuInterfaceY.value = targetRect.top + 12.5;
        } else if (chosenInterface.value.side === 'left') {
            contextMenuInterfaceSide.value = 'left';
            contextMenuInterfaceX.value = nodeRect.left - 10;
            contextMenuInterfaceY.value = targetRect.top + 12.5;
        }

        showContextMenuInterface.value = true;
        openContextMenu(
            showContextMenuInterface,
            contextMenuInterfaceX,
            contextMenuInterfaceY,
            markRaw(createContextMenuInterfaceItems()),
            [leftSocketsRefs, rightSocketsRefs],
            onContextMenuInterfaceClick,
            undefined,
            contextMenuInterfaceSide.value === 'left' && { translate: '-100%' },
            () => {
                chosenInterface.value = undefined;
            },
        );
    }
};

const openContextMenuTitle = async () => {
    if (
        canOpenContextMenu.value &&
        showContextMenuTitle.value === false &&
        showContextMenuInterface.value === false &&
        showContextMenuProperty.value === false
    ) {
        contextMenuTitleX.value = mouse.x + 10;
        contextMenuTitleY.value = mouse.y + 10;
        showContextMenuTitle.value = true;
        openContextMenu(
            showContextMenuTitle,
            contextMenuTitleX,
            contextMenuTitleY,
            markRaw(contextMenuTitleItems.value),
            undefined,
            onContextMenuTitleClick,
            nodeURLs,
        );
    }
};

const toggleGroup = (intf) => {
    intf.group.forEach((name) => {
        props.node.inputs[name].hidden = !intf.value;
    });
};

const updateDynamicInterfaces = (intf) => {
    props.node.updateDynamicInterfaces(intf);
};

/* eslint-disable no-param-reassign */
const switchSides = (intf) => {
    if (intf.side === 'left') {
        updateInterfacePosition(props.node, intf, 'right');
    } else {
        updateInterfacePosition(props.node, intf, 'left');
    }
};

// Property context menu

const contextMenuPropertyX = ref(0);
const contextMenuPropertyY = ref(0);

const createContextMenuPropertyItems = () => {
    const items = [];

    if (chosenProperty !== undefined) {
        const labels = (isBigBus(chosenProperty) ? ['Expose Interface', 'Privatize Interface'] : ['Expose Property', 'Privatize Property']);
        const propertyMode = (chosenProperty.externalName === undefined ?
            { value: 'SetExternalName', label: labels[0], icon: icons.Subgraph } :
            { value: 'UnsetExternalName', label: labels[1], icon: icons.Subgraph }
        );
        items.push(propertyMode);
        if (!chosenProperty.groupProperty && !isBigBus(chosenProperty)) {
            items.push({ value: 'Hide', label: 'Hide', icon: icons.Hide });
        }
    }

    return items;
};

/* eslint-disable default-case */
const onContextMenuPropertyClick = (action) => {
    switch (action) {
        case 'Hide':
            if (chosenProperty !== undefined && !chosenProperty.group) {
                chosenProperty.hidden = true;
            }
            break;
        case 'SetExternalName':
            viewModel.value.editor.exposeInterface(
                graph.value.id,
                chosenProperty,
            );

            notifyEvents.exposedInterface.emit([chosenProperty, graph.value.id, true]);
            graph.value.events.exposeInterface.emit([chosenProperty, viewModel.value.editor]);
            break;
        case 'UnsetExternalName':
            graph.value.events.privatizeInterface.emit([chosenProperty, viewModel.value.editor]);
            viewModel.value.editor.privatizeInterface(
                graph.value.id,
                chosenProperty,
            );

            notifyEvents.exposedInterface.emit([chosenProperty, graph.value.id, false]);
            break;
    }
};

const openContextMenuProperty = async (property) => {
    showContextMenuProperty.value = false;
    await nextTick();
    if (!viewModel.value.editor.readonly) {
        chosenProperty = property;
        const items = createContextMenuPropertyItems();

        if (items.length > 0) {
            contextMenuPropertyX.value = mouse.x + 10;
            contextMenuPropertyY.value = mouse.y + 10;
            showContextMenuProperty.value = true;
            openContextMenu(
                showContextMenuProperty,
                contextMenuPropertyX,
                contextMenuPropertyY,
                markRaw(createContextMenuPropertyItems()),
                [propertiesRef],
                onContextMenuPropertyClick,
            );
        }
    }
};

const isHighlighted = (intf) =>
    (!intf ? false : props.interfaces?.map((p) => p?.id).includes(intf?.id));

</script>

<style lang="scss" scoped>
.interface-cursor {
    position: relative;
    top: 0;
    left: 0;

    & > .line {
        position: absolute;
        height: 0.2em;
        width: 1.4em;
        background-color: $gold;
        z-index: 100;
    }
}
.pill {
    border-radius: 25px;
    transform: translate(-20%, -50%);
    font-size: 0.75em;
    padding: 5px;
    width: auto;
    height: auto;
    z-index: 1;
    pointer-events: none;
    position: absolute;
    top: 0;
    right: 0;
}

</style>
