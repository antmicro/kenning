<!--
Copyright (c) 2022-2026 Antmicro <www.antmicro.com>

SPDX-License-Identifier: Apache-2.0
-->

<!--
Defines the main editor component - canvas in which the pipeline is drawn and
edited by a user

Inherits from baklavajs/rendered-vue/src/editor/Editor.vue

Minimap component is removed whatsoever as they are not used.
Hovered connections are calculated and rendered with an appropriate `isHighlighted` value.
-->

<template>
    <div
        ref="el"
        tabindex="-1"
        class="baklava-editor"
        :class="{
            'baklava-ignore-mouse': !!temporaryConnection || dragging,
            '--temporary-connection': !!temporaryConnection,
        }"
        :style="editorStyle"
        @wheel.self="mouseWheel"
        @keydown="keyDown"
        @keyup="keyUp"
        @mouseleave="!readonly && onRightPointerUp"
        @drop.prevent="!readonly && onDrop($event)"
        @dragenter.prevent
        @dragover.prevent
        oncontextmenu="return false;"
    >
        <slot name="background">
            <background />
        </slot>

        <template v-if="!validating">
            <slot
                name="palette"
                v-if="!(readonly || hideHud)"
            >
                <Palette />
            </slot>

            <div
                class="grouping-container"
                :style="nodeContainerStyle"
                @wheel="mouseWheel"
                @mouseenter="changeHoveredConnections"
                @mousemove="changeHoveredConnections"
                @mouseleave="clearHighlight"
            >
                <RectangleGrouping
                    v-for="group in visibleGroups"
                    :key="group.id"
                    :model-value="group.name"
                    @update:modelValue="(name) => updateGroupName(group.id, name)"
                    :min="group.min"
                    :max="group.max"
                    :color="group.color"
                />
            </div>

            <div
                class="node-container"
                :style="nodeContainerStyle"
                @wheel="mouseWheel"
                @mouseenter="changeHoveredConnections"
                @mousemove="changeHoveredConnections"
                @mouseleave="clearHighlight"
            >
                <CustomNode
                    v-memo="[visibleNodes, ...selectedNodes,
                    currentViewName, ignoredInterfacesTypes]"
                    v-for="node in visibleNodes"
                    :key="node.id + counter.toString()"
                    :node="node"
                    :selected="selectedNodes.includes(node)"
                    :greyedOut="greyedOutNodes.includes(node)"
                    :interfaces="highlightInterfaces"
                    :ignoredInterfacesType="[...ignoredInterfacesTypes]"
                    @openContextMenu="(open, x, y, items, ignore, onclick,
                    urls, styles, onclose) => {
                        updateContextMenu(node, open, x, y, items, ignore, onclick, urls,
                        styles, onclose);
                    }"
                    @select="(ev) => selectNode(node, ev)"
                    @transformed="() => updateGroupsOf(node.id)"
                    @startDrag="() => handleStartDrag(node)"
                    @stopDrag="handleStopDrag"
                />
                <CustomNode
                    v-for="node in ignoredNodes"
                    :key="node.id + counter.toString()"
                    :node="node"
                    :hidden="true"
                />
            </div>
            <svg
                class="custom-connections-container"
                :style="nodeContainerStyle"
                @mouseenter="changeHoveredConnections"
                @mousemove="changeHoveredConnections"
                @mouseleave="clearHighlight"
                @wheel="mouseWheel"
            >
                <PipelineManagerConnection
                    v-memo="[...visibleConnections, currentViewName, draggedNode]"
                    v-for="connection in visibleConnections"
                    :key="connection.id + counter.toString()"
                    :connection="connection"
                    ref="connRefs"
                    :isHighlighted="highlightConnections.includes(connection)"
                    :draggedNode="draggedNode"
                />
                <TemporaryConnection
                    name="temporaryConnection"
                    :temporary-connection="temporaryConnection"
                    v-if="temporaryConnection && temporaryConnectionRender"
                    :connection="temporaryConnection"
                />
            </svg>

            <div class="selection-container">
                <RectangleSelection ref="rectangleSelection"/>
            </div>

            <Zoom @zoom-in="zoomIn" @zoom-out="zoomOut" @center="center" :floating="!hideHud" />

            <Return v-if="preview && isInSubgraph" @click="returnFromSubgraph" />

            <CustomContextMenu
                ref="contextMenuComp"
                v-model="contextMenu.open"
                :x="contextMenu.x"
                :y="contextMenu.y"
                :style="contextMenu.styles"
                :items="contextMenu.items ?? []"
                :ignore-close="contextMenu.ignore ?? []"
                @click="contextMenu.onclick"
                @close="contextMenu.onclose"
                :transition="''"
            />
            <Panel v-show="showWelcome" :blur="false" class="welcome-container-panel">
                <ParentMenu>
                    <WelcomeMenu :loadFiles="loadFiles" />
                </ParentMenu>
            </Panel>
        </template>
    </div>
</template>

<script>
import { EditorComponent, useGraph } from '@baklavajs/renderer-vue';
import {
    defineComponent, nextTick, ref, computed, watch, onMounted, reactive, useTemplateRef,
} from 'vue';
import { useThrottleFn } from '@vueuse/core';

import fuzzysort from 'fuzzysort';
import { BaklavaEvent } from '@baklavajs/events';
import { isJSONRPCRequest, isJSONRPCResponse, JSONRPC } from 'json-rpc-2.0';
import usePanZoom from './panZoom';

import CustomContextMenu from './ContextMenu.vue';
import CustomNode from './CustomNode.vue';
import PipelineManagerConnection from './connection/PipelineManagerConnection.vue';
import TemporaryConnection from './connection/TemporaryConnection.vue';
import Palette from '../components/Palette.vue';
import { provideTemporaryConnection } from './temporaryConnection';
import NotificationHandler from '../core/notifications';
import EditorManager, { loadJsonFromRemoteLocation } from '../core/EditorManager';
import RectangleSelection from './RectangleSelection.vue';
import RectangleGrouping from './RectangleGrouping.vue';
import nodeInsideSelection from './rectangleSelection.js';
import getExternalApplicationManager from '../core/communication/ExternalApplicationManager';
import jsonRPC, { frontendEndpoints } from '../core/communication/rpcCommunication';
import Zoom from '../components/Zoom.vue';
import Return from '../components/Return.vue';
import Panel from '../components/Panel.vue';
import { ParentMenu, WelcomeMenu } from '../components/menu';
import { loadingScreen } from '../core/utils';
import initNotificationEvents from './CustomNotifications.js';

import globalProperties from '../globalProperties.ts';

const RECTANGLE_GROUP_PADDING = 20;

export default defineComponent({
    extends: EditorComponent,
    props: {
        loading: {
            required: true,
            type: Boolean,
        },
    },
    components: {
        CustomNode,
        Panel,
        ParentMenu,
        PipelineManagerConnection,
        TemporaryConnection,
        Palette,
        RectangleSelection,
        RectangleGrouping,
        WelcomeMenu,
        Zoom,
        Return,
        CustomContextMenu,
    },
    emits: ['setLoad'],
    setup(props, { emit }) {
        const {
            el,
            counter,
            nodes,
            connections,
            selectedNodes,
            nodeContainerStyle,
            keyDown,
            keyUp,
        } = EditorComponent.setup(props);

        initNotificationEvents();

        const connRefs = ref([]);
        const { graph } = useGraph();

        const panZoom = usePanZoom();
        const temporaryConnection = provideTemporaryConnection();
        const editorManager = EditorManager.getEditorManagerInstance();

        const highlightConnections = ref([]);
        const highlightInterfaces = ref([]);
        const groups = computed(() => graph.value.groups ?? []);
        const groupsOfNode = ref({});
        const visibleGroups = ref([]);
        const draggedNode = ref(null);

        const readonly = computed(() => props.viewModel.editor.readonly);
        const hideHud = computed(() => props.viewModel.editor.hideHud);

        const rectangleSelection = ref(null);
        const selectedNodesCtrlBackup = [];
        const scale = computed(() => graph.value.scaling);

        const searchQuery = computed(() => props.viewModel.editor.searchQuery);
        const greyedOutNodes = ref([]);

        let pressStartTime = 0;
        const longPressMilisThreshold = 100;

        const editorStyle = computed(() => ({
            '--scale': scale.value,
            cursor: panZoom.dragging.value ? 'move' : 'default',
        }));

        const isInSubgraph = computed(() => props.viewModel.editor.isInSubgraph());

        const preview = computed(() => props.viewModel.editor.preview);

        const contextMenuComp = useTemplateRef('contextMenuComp');
        const contextMenu = reactive({
            open: null,
            x: 0,
            y: 0,
            items: [],
            ignore: [],
            onclick: () => {},
            styles: {},
            onclose: () => {},
        });
        const updateContextMenu = (_node, open, x, y, items, ignore, onclick,
            urls, styles, onclose) => {
            contextMenuComp.value.closeContextMenu();
            nextTick(() => {
                contextMenu.x = x;
                contextMenu.y = y;
                contextMenu.items = items;
                contextMenu.ignore = ignore;
                contextMenu.onclick = onclick;
                contextMenu.urls = urls;
                contextMenu.styles = styles;
                // eslint-disable-next-line no-param-reassign
                open.value = true;
                contextMenu.open = open;
                contextMenu.onclose = onclose;
            });
        };

        const unselectAllNodes = () => {
            /* eslint-disable vue/no-mutating-props,no-param-reassign */
            props.viewModel.displayedGraph.selectedNodes = [];
        };

        const setLoad = new BaklavaEvent();
        setLoad.subscribe('editor', (value) => {
            emit('setLoad', value);
        });
        props.viewModel.editor.events.setLoad = setLoad;

        const appendSelectMultipleNodes = () => {
            graph.value.nodes.forEach((node) => {
                if (graph.value.selectedNodes.includes(node)) {
                    return;
                }

                const selectionBoundingRect = rectangleSelection.value.boundingRect;

                // eslint-disable-next-line max-len
                if (nodeInsideSelection(graph.value, node, node.position, selectionBoundingRect)) {
                    graph.value.selectedNodes.push(node);
                }
            });
        };

        const handleStartDrag = (node) => {
            draggedNode.value = node;
        };

        const handleStopDrag = () => {
            draggedNode.value = null;
        };

        const selectNode = (node, event) => {
            // If select was triggered by event check if CTRL is pressed,
            // otherwise use default method
            if ((event && !(event.ctrlKey)) || (!event && !props.viewModel.commandHandler.pressedKeys.includes('Control'))) {
                unselectAllNodes();
            }

            props.viewModel.displayedGraph.selectedNodes.push(node);
        };

        const selectMultipleNodes = () => {
            graph.value.selectedNodes = [];
            graph.value.nodes.forEach((node) => {
                const selectionBoundingRect = rectangleSelection.value.boundingRect;

                // eslint-disable-next-line max-len
                if (nodeInsideSelection(graph.value, node, node.position, selectionBoundingRect)) {
                    graph.value.selectedNodes.push(node);
                }
            });
        };

        let pointersDown = 0;
        const onPointerDown = (ev) => {
            pointersDown += 1;
            if (ev.target === el.value) {
                panZoom.onPointerDown(ev);
            }
            temporaryConnection.onMouseDown(ev);

            pressStartTime = new Date();
        };

        const onRightPointerDown = (ev) => {
            if (ev.target === el.value) {
                unselectAllNodes();
                rectangleSelection.value.onPointerDown(ev);
            }

            pressStartTime = new Date();
        };

        const onPointerMove = (ev) => {
            panZoom.onPointerMove(ev);
            temporaryConnection.onMouseMove(ev);
            rectangleSelection.value.onPointerMove(ev);
            if (rectangleSelection.value.selecting) {
                selectMultipleNodes();
            }
        };

        const onPointerUp = (ev) => {
            panZoom.onPointerUp(ev);
            temporaryConnection.onMouseUp(ev);

            // handle press & hold
            const currentTime = new Date();
            const elapsedTime = currentTime - pressStartTime;
            if (elapsedTime < longPressMilisThreshold
                && ev.target === el.value) {
                unselectAllNodes();
            }

            if (pointersDown === 1) {
                document.removeEventListener('pointerup', onPointerUp);
                document.removeEventListener('pointermove', onPointerMove);
            }
            pointersDown -= 1;
        };

        const onRightPointerUp = (ev) => {
            // handle press & hold right mouse button
            const currentTime = new Date();
            const elapsedTime = currentTime - pressStartTime;
            if (elapsedTime >= longPressMilisThreshold
                && ev.target === el.value) {
                appendSelectMultipleNodes();
            }
            rectangleSelection.value.onPointerUp();

            document.removeEventListener('pointerup', onRightPointerUp);
            document.removeEventListener('pointermove', onPointerMove);
        };

        const onRightPointerDownCtrl = (ev) => {
            rectangleSelection.value.onPointerDown(ev);
            selectedNodesCtrlBackup.value = graph.value.selectedNodes;
        };

        const onRightPointerUpCtrl = () => {
            graph.value.selectedNodes = selectedNodesCtrlBackup.value;
            appendSelectMultipleNodes();
            rectangleSelection.value.onPointerUp();

            document.removeEventListener('pointerup', onRightPointerUpCtrl);
            document.removeEventListener('pointermove', onPointerMove);
        };

        document.addEventListener('pointerdown', (ev) => {
            // Dragging
            if (ev.button === 0 && !ev.shiftKey) {
                onPointerDown(ev);
                document.addEventListener('pointerup', onPointerUp);
                document.addEventListener('pointermove', onPointerMove);
            }

            // Rectangle selection
            if (readonly.value) {
                return;
            }
            if (ev.button === 2 && ev.ctrlKey) {
                onRightPointerDownCtrl(ev);
                document.addEventListener('pointerup', onRightPointerUpCtrl);
                document.addEventListener('pointermove', onPointerMove);
            } else if (ev.button === 2) {
                onRightPointerDown(ev);
                document.addEventListener('pointerup', onRightPointerUp);
                document.addEventListener('pointermove', onPointerMove);
            }
        });

        const clearHighlight = () => {
            highlightConnections.value.splice(0, highlightConnections.value.length);
        };

        // If a connection is removed by clicking on it all highlighted connections are removed
        watch(connections.value, () => {
            clearHighlight();
        });
        const backgroundSettings = computed(() => props.viewModel.settings.background);
        watch(backgroundSettings.value, () => {
            // eslint-disable-next-line prefer-destructuring
            const color = backgroundSettings.value.color;
            if (color && el.value.style) {
                el.value.style.setProperty('--baklava-editor-background-pattern-default', color, 'important');
            }
        });
        const addHighlight = (connection) => {
            if (!highlightConnections.value.includes(connection)) {
                highlightConnections.value.push(connection);
            }
        };

        const addInterfaceHighlight = (it) => {
            if (!highlightInterfaces.value.includes(it)) {
                highlightInterfaces.value.push(it);
            }
        };
        const clearInterfaceHighlight = () => {
            highlightInterfaces.value.splice(0, highlightInterfaces.value.length);
        };

        watch(temporaryConnection?.temporaryConnection, () => {
            if (typeof (temporaryConnection ?? null) === 'undefined') return;
            if (temporaryConnection?.temporaryConnection === null) return;
            if (temporaryConnection?.temporaryConnection?.value?.from) {
                for (let a = 0; a < nodes.value.length; a += 1) {
                    const viableConnections = [
                        ...(Object.values(nodes.value[a].outputs) || []),
                        ...(Object.values(nodes.value[a].inputs) || []),
                    ];

                    viableConnections.forEach((n) => {
                        if (n.port
                        && n !== temporaryConnection.temporaryConnection.value.from) {
                            const result = graph.value.checkConnection(
                                temporaryConnection.temporaryConnection.value.from,
                                n,
                            );
                            if (!result.connectionAllowed) {
                                addInterfaceHighlight(n);
                            }
                        }
                    });
                }
            } else {
                clearInterfaceHighlight();
            }
        });

        const removeHighlight = (connection) => {
            const index = highlightConnections.value.indexOf(connection);
            if (index >= 0) {
                highlightConnections.value.splice(index, 1);
            }
        };

        const findConnectionsToHighlight = (elementsUnderCursor) => {
            const elementsWithZIndex = elementsUnderCursor.map((element) => {
                const z = window.getComputedStyle(element).zIndex;
                return {
                    z_index: Number.isNaN(parseInt(z, 10)) ? 0 : parseInt(z, 10),
                    element,
                };
            });

            const highestZIndexElement =
                elementsWithZIndex.reduce(
                    (max, curr) => (curr.z_index > max.z_index ? curr : max), elementsWithZIndex[0],
                );

            const hoveredHtml = connRefs.value.filter((conn) =>
                conn.containsPoint([highestZIndexElement.element]),
            );
            // Convert DOM elements to BaklavaJS connections.
            const hoveredConnections = connections.value.filter(
                (conn) => hoveredHtml.filter((htmlEl) => htmlEl.connection === conn).length > 0,
            );

            // Check for hovered interfaces.
            const hoveredInterface = elementsUnderCursor.find((element) => {
                // Check if the element is a port or inside an interface.
                if (element.classList?.contains('__port')) {
                    return true;
                }
                if (element.classList?.contains('__port-bus')) {
                    return true;
                }
                if (element.classList?.contains('baklava-node-interface')) {
                    return true;
                }
                // Check if element is inside an interface.
                return element.closest?.('.baklava-node-interface') !== null;
            });

            let connectionsToHighlight = [];

            if (hoveredConnections.length > 0) {
                const highlighted = connections.value.filter(
                    (conn) => hoveredConnections.filter((hov) => hov.from === conn.from).length > 0,
                );
                const mostSuitableHighlight = highlighted.at(0);
                if (mostSuitableHighlight) {
                    connectionsToHighlight = [mostSuitableHighlight];
                }
            } else if (hoveredInterface) {
                // Find the interface ID and highlight all connections.
                let interfaceElement = hoveredInterface;

                interfaceElement = hoveredInterface.closest('.baklava-node-interface');
                if (interfaceElement?.id) {
                    // Find all connections that involve this interface.
                    connectionsToHighlight = connections.value.filter((conn) =>
                        conn.from?.id === interfaceElement.id
                        || conn.to?.id === interfaceElement.id,
                    );
                }
            }

            return connectionsToHighlight;
        };

        const changeHoveredConnections = (ev) => {
            const elements = document.elementsFromPoint(ev.clientX, ev.clientY);
            const connectionsToHighlight = findConnectionsToHighlight(elements);

            // Toggle highlighting of connections.
            connections.value.forEach((conn) => {
                if (connectionsToHighlight.includes(conn)) {
                    addHighlight(conn);
                } else {
                    removeHighlight(conn);
                }
            });
        };

        const ignoredLayers = computed(() => props.viewModel.editor.getIgnoredLayers());

        const ignoredInterfacesRef = ref([]);

        const ignoredInterfacesTypes = computed(() =>
            props.viewModel.editor.getIgnoredInterfaces(undefined, ignoredLayers.value));

        const ignoredNodesTypes = computed(() =>
            props.viewModel.editor.getIgnoredNodes(undefined, ignoredLayers.value));

        const visibleNodes = computed(() =>
            nodes.value.filter((n) => !ignoredNodesTypes.value.has(n.layer)),
        );
        const updateGroupName = (groupId, newName) => {
            if (!readonly.value) {
                const realGroup = graph.value.groups.find((g) => g.id === groupId);
                if (!realGroup) return;

                realGroup.name = newName;
            }
        };

        function computeGroupBoundsFromDOM(nodesInGroup) {
            if (!nodesInGroup?.length) {
                return { min: { x: 0, y: 0 }, max: { x: 0, y: 0 } };
            }

            const container = el.value.querySelector('.node-container');
            const containerRect = container.getBoundingClientRect();

            let minX = Infinity;
            let minY = Infinity;
            let maxX = -Infinity;
            let maxY = -Infinity;

            nodesInGroup.forEach((nodeId) => {
                const nodeEl = container.querySelector(
                    `[data-node-id="${nodeId}"]`,
                );
                if (!nodeEl) return;

                const rect = nodeEl.getBoundingClientRect();

                const x1 = (rect.left - containerRect.left) / scale.value;
                const y1 = (rect.top - containerRect.top) / scale.value;
                const x2 = (rect.right - containerRect.left) / scale.value;
                const y2 = (rect.bottom - containerRect.top) / scale.value;

                minX = Math.min(minX, x1);
                minY = Math.min(minY, y1);
                maxX = Math.max(maxX, x2);
                maxY = Math.max(maxY, y2);
            });

            return {
                min: { x: minX - RECTANGLE_GROUP_PADDING, y: minY - RECTANGLE_GROUP_PADDING },
                max: { x: maxX + RECTANGLE_GROUP_PADDING, y: maxY + RECTANGLE_GROUP_PADDING },
            };
        }
        const updateGroupsOf = (nodeId) => {
            if (!(nodeId in groupsOfNode.value)) {
                groupsOfNode.value[nodeId] = [];
                visibleGroups.value.filter((g) => g.nodes.some((i) => i === nodeId))
                    .forEach((group) => {
                        groupsOfNode.value[nodeId].push(group);
                    });
            }
            groupsOfNode.value[nodeId].forEach((group) => {
                const { min, max } = computeGroupBoundsFromDOM(group.nodes);

                group.min = min;
                group.max = max;
            });
        };
        const updateVisibleGroups = useThrottleFn(() => {
            groupsOfNode.value = {};
            visibleGroups.value = groups.value.map((group) => {
                const groupNodes = graph.value.nodes.filter((node) =>
                    group.nodes.includes(node.id),
                ).map((n) => n.id);

                const { min, max } = computeGroupBoundsFromDOM(groupNodes);

                return {
                    min,
                    max,
                    ...group,
                };
            });
            visibleGroups.value.forEach((group) => {
                group.nodes.forEach((id) => {
                    if (!(id in groupsOfNode.value)) {
                        groupsOfNode.value[id] = [];
                    }
                    groupsOfNode.value[id].push(group);
                });
            });
        }, 50); // 20fps

        const currentViewName = computed(() => props.viewModel.editor.currentView);

        watch([
            () =>
                groups.value.map((g) => ({
                    id: g.id,
                    name: g.name,
                    nodes: g.nodes?.slice() ?? [],
                })),
            () => currentViewName.value,
        ],
        updateVisibleGroups,
        { deep: false, flush: 'post' },
        );

        const ignoredNodes = computed(() =>
            nodes.value.filter((n) => ignoredNodesTypes.value.has(n.layer)),
        );
        const ignoredNodesId = computed(() => ignoredNodes.value.map((n) => n.id));

        const visibleConnections = computed(() =>
            connections.value.filter(
                (c) =>
                    !c.from.type?.some((t) => ignoredInterfacesTypes.value.has(t)) &&
                    !c.to.type?.some((t) => ignoredInterfacesTypes.value.has(t)) &&
                    !ignoredNodesId.value.includes(c.from.nodeId) &&
                    !ignoredNodesId.value.includes(c.to.nodeId),
            ),
        );

        const highlightedNodes = computed(() =>
            visibleNodes.value.filter(
                (n) => props.viewModel.displayedGraph.selectedNodes.includes(n),
            ),
        );

        const externalApplicationManager = getExternalApplicationManager();

        watch(visibleNodes, async (value, old) => {
            if (!externalApplicationManager.isConnected()) return;
            if (!editorManager.notifyWhenChanged) return;
            const newIds = Object.values(value).map((n) => n.id);
            const oldIds = Object.values(old).map((n) => n.id);
            const nodesAdded = [];
            Object.values(value).forEach((node) => {
                if (!(oldIds.includes(node.id))) {
                    nodesAdded.push(node.save());
                }
            });
            const nodesDeleted = [];
            Object.values(old).forEach((node) => {
                if (!(newIds.includes(node.id))) {
                    nodesDeleted.push(node.id);
                }
            });
            const data = {
                graph_id: graph.value.id,
                nodes: {
                    added: nodesAdded,
                    deleted: nodesDeleted,
                },
            };
            await externalApplicationManager.notifyAboutChange('nodes_on_change', data);
        });

        watch(visibleConnections, async (value, old) => {
            if (!externalApplicationManager.isConnected()) return;
            if (!editorManager.notifyWhenChanged) return;
            const newIds = Object.values(value).map((n) => n.id);
            const oldIds = Object.values(old).map((n) => n.id);
            const connectionsAdded = [];
            Object.values(value).forEach((connection) => {
                if (!(oldIds.includes(connection.id))) {
                    connectionsAdded.push({
                        id: connection.id,
                        from: connection.from.id,
                        to: connection.to.id,
                    });
                }
            });
            const connectionsDeleted = [];
            Object.values(old).forEach((connection) => {
                if (!(newIds.includes(connection.id))) {
                    connectionsDeleted.push({
                        from: connection.from.id,
                        to: connection.to.id,
                    });
                }
            });
            const data = {
                graph_id: graph.value.id,
                connections: {
                    added: connectionsAdded,
                    deleted: connectionsDeleted,
                },
            };
            await externalApplicationManager.notifyAboutChange('connections_on_change', data);
        });

        watch(highlightedNodes, async (value, old) => {
            if (!externalApplicationManager.isConnected()) return;
            if (!editorManager.notifyWhenChanged) return;
            const newIds = Object.values(value).map((n) => n.id);
            const oldIds = Object.values(old).map((n) => n.id);
            const nodesSelected = [];
            Object.values(value).forEach((node) => {
                if (!(oldIds.includes(node.id))) {
                    nodesSelected.push(node.id);
                }
            });
            const nodesUnselected = [];
            Object.values(old).forEach((node) => {
                if (!(newIds.includes(node.id))) {
                    nodesUnselected.push(node.id);
                }
            });
            const data = {
                graph_id: graph.value.id,
                nodes: {
                    selected: nodesSelected,
                    unselected: nodesUnselected,
                },
            };
            await externalApplicationManager.notifyAboutChange('nodes_on_highlight', data);
        });

        const filterNodes = (query) => {
            const threshold = -50;

            const matchingNodes = visibleNodes.value.filter((node) => {
                const { type } = node;

                const resultTitle = fuzzysort.single(query, node.title);
                const resultType = fuzzysort.single(query, type);

                if ((resultTitle !== null && resultTitle.score > threshold) ||
                    (resultType !== null && resultType.score > threshold)) {
                    node.highlightedTitle = fuzzysort.highlight(resultTitle, '<span>', '</span>');
                    node.highlightedType = fuzzysort.highlight(resultType, '<span>', '</span>');

                    return true;
                }
                node.highlightedTitle = node.title;
                node.highlightedType = node.type;
                return false;
            });

            return matchingNodes;
        };

        const nodeTitles = computed(() =>
            nodes.value.map((n) => n.title),
        );

        watch([searchQuery, nodeTitles], () => {
            if (searchQuery.value === undefined || searchQuery.value === '') {
                greyedOutNodes.value = [];
                visibleNodes.value.forEach((node) => {
                    node.highlightedTitle = node.title;
                    node.highlightedType = node.type;
                });
                return;
            }
            const matchingNodes = filterNodes(searchQuery.value);

            const nonMatchingNodes = visibleNodes.value.filter(
                (node) => !matchingNodes.includes(node),
            );

            greyedOutNodes.value = nonMatchingNodes;
        });

        const defaultSpecification = process.env.VUE_APP_SPECIFICATION_PATH !== undefined;
        const defaultDataflow = process.env.VUE_APP_DATAFLOW_PATH !== undefined;
        const verboseLoad =
            process.env.VUE_APP_VERBOSE !== undefined && process.env.VUE_APP_VERBOSE === 'true';

        /**
         * Prepares the Editor based on specification configuration.
         *
         * The function validates the specification, updates the Editor
         * and provides error messages for erroneous content.
         *
         * @param specification The object holding the parsed specification file
         */
        async function updateEditorSpecification(specification) {
            let errors = EditorManager.validateSpecification(specification);
            let warnings;
            let info;

            if (errors.length) {
                NotificationHandler.terminalLog('error', 'Specification is invalid', errors);
                if (editorManager.specificationVersion !== specification.version) {
                    NotificationHandler.terminalLog(
                        'error',
                        'Mismatched specification version',
                        `Specification version (${specification.version}) differs from the current version (${editorManager.specificationVersion}). It may result in unexpected behaviour.` +
                        ' Please refer to https://antmicro.github.io/kenning-pipeline-manager/changelogs.html to see if that is the case.',
                    );
                }
                return errors;
            }

            ({ errors, warnings, info } = await editorManager.updateEditorSpecification(specification)); // eslint-disable-line prefer-const,max-len
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

            return errors;
        }

        /**
         * Prepares the Editor based on dataflow configuration.
         *
         * The function loads the dataflow
         * and provides error messages for erroneous content.
         *
         * @param dataflow The object holding the parsed dataflow file
         * @param centerAtTop determines whether to center the editor
         * at the topmost node after loading
         */
        async function updateDataflow(dataflow, centerAtTop = false) {
            const { errors, warnings, info } = await editorManager.loadDataflow(
                dataflow,
                false,
                false,
                null,
                centerAtTop,
            );

            if (Array.isArray(warnings) && warnings.length) {
                NotificationHandler.terminalLog(
                    'warning',
                    'Issue when loading dataflow',
                    warnings,
                );
            }
            if (Array.isArray(errors) && errors.length) {
                const messageTitle = globalProperties.softLoad ?
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

            return errors;
        }

        /* eslint-disable no-lonely-if */
        onMounted(async () => {
            // Load specification and/or dataflow delivered via window.postMessage
            window.addEventListener('message', async (event) => {
                // TODO: introduce mechanism for checking event.origin against allowed origins
                const data = event.data ?? {};
                if (typeof data === 'string') return;
                data.jsonrpc ??= JSONRPC;

                if (isJSONRPCResponse(data)) {
                    if (event.source === window) return;
                    jsonRPC.client.receive(data);
                    return;
                }

                data.id ??= jsonRPC.client.createID();
                if (!isJSONRPCRequest(data)) return;

                let response = data.method === 'register_external_frontend'
                    ? externalApplicationManager.registerFrontendApplication(event.source, data)
                    : null;

                response ??= event.data.method in frontendEndpoints
                    ? (await jsonRPC.server.receive(data))
                    : (await jsonRPC.requestAdvanced(data, {
                        externalApp: externalApplicationManager.externalApp,
                    }));

                if (response) event.source.postMessage(JSON.parse(JSON.stringify(response)), '*');
            });

            NotificationHandler.setShowNotification(false);
            editorManager.updateMetadata({}); // Defaulting metadata values

            // eslint-disable-next-line no-use-before-define
            await loadingScreen(loadInit, setLoad);
        });

        const loadInit = async () => {
            const escapedsearch = window.location.search.replace(/&amp;/g, '&');
            const urlParams = new URLSearchParams(escapedsearch);
            if (urlParams.has('preview')) {
                const setting = urlParams.get('preview') === 'true';
                props.viewModel.editor.preview = setting;
            }

            let centerAtTop = false;
            if (urlParams.has('center_at_top')) {
                const setting = urlParams.get('center_at_top') === 'true';
                centerAtTop = setting;
            }

            let specText;
            // Try loading default specification and/or dataflow from URLs provided in an
            if (urlParams.has('spec')) {
                const [status, ret] = await loadJsonFromRemoteLocation(urlParams.get('spec'));
                if (status === false) {
                    NotificationHandler.terminalLog(
                        'error',
                        ret,
                    );
                    NotificationHandler.terminalLog(
                        'error',
                        `Failed to load the specification file from: ${urlParams.get('spec')}`,
                    );
                } else {
                    specText = ret;
                }
            } else if (defaultSpecification) {
                // Use raw-loader which does not parse the specification so that it is possible
                // To add a more verbose validation log
                if (verboseLoad) {
                    specText =
                        require(`!!raw-loader!${process.env.VUE_APP_SPECIFICATION_PATH}`).default; // eslint-disable-line global-require,import/no-dynamic-require
                } else {
                    specText = require(process.env.VUE_APP_SPECIFICATION_PATH); // eslint-disable-line global-require,import/no-dynamic-require,max-len
                }
            }

            if (specText !== undefined) {
                if (urlParams.has('include')) {
                    if (specText.include === undefined) specText.include = [];
                    specText.include.push(urlParams.get('include'));
                }
                const errors = await updateEditorSpecification(specText);

                if (errors.length) {
                    NotificationHandler.restoreShowNotification();
                    return;
                }

                let dataflow;
                if (urlParams.has('graph')) {
                    const [status, ret] = await loadJsonFromRemoteLocation(urlParams.get('graph'));
                    if (status === false) {
                        NotificationHandler.terminalLog(
                            'error',
                            ret,
                        );
                        NotificationHandler.terminalLog(
                            'error',
                            `Failed to load the graph file from: ${urlParams.get('graph')}`,
                        );
                    } else {
                        dataflow = ret;
                    }
                } else if (defaultDataflow) {
                    dataflow = require(process.env.VUE_APP_DATAFLOW_PATH); // eslint-disable-line global-require,max-len,import/no-dynamic-require
                }
                if (dataflow) {
                    await updateDataflow(dataflow, centerAtTop);
                }
            }
            NotificationHandler.restoreShowNotification();
        };

        const loadFiles = async (files) => loadingScreen(async () => {
            const notify = NotificationHandler.showToast;

            if (files.length > 2) {
                notify('error', `Up to two files supported, inserted: ${files.length}`);
                return;
            }

            let objects;
            try {
                const parse = async (file) => JSON.parse(await file.text());
                objects = await Promise.all(Array.from(files).map(parse));
            } catch (error) {
                notify('error', 'Dropped file is not in JSON format');
                return;
            }

            const isSpecificationFunc = (o) => Boolean(o.nodes || o.include || o.includeGraphs);
            const isDataflowFunc = (o) => Boolean(o.graphs && !isSpecificationFunc(o));
            const isInvalid = objects.map((o) => !isSpecificationFunc(o) && !isDataflowFunc(o));

            // [invalid, invalid]
            if (isInvalid.every(Boolean)) {
                if (objects.length === 1) notify('error', 'Provided file is not of Pipeline Manager format');
                else notify('error', 'Neither of provided files is of Pipeline Manager format');
                return;
            }

            // Remove invalid
            objects = objects.filter((_, i) => !isInvalid[i]);
            if (isInvalid.some(Boolean)) {
                notify('warning', 'One of the objects is not of Pipeline Manager format');
            }

            const isSpecification = objects.map(isSpecificationFunc);
            const isDataflow = objects.map(isDataflowFunc);
            let specification;
            let dataflow;

            if (isSpecification.every(Boolean)) {
                // [spec], [spec, spec]
                if (objects.length > 1) {
                    notify('warning', 'Two specifications provided, using the first one.');
                }
                [specification] = objects;
            } else if (isDataflow.every(Boolean)) {
                // [dataflow], [dataflow, dataflow]
                if (objects.length > 1) {
                    notify('warning', 'Two dataflows provided, using the first one.');
                }
                [dataflow] = objects;
            } else if (isSpecification.some(Boolean) && isDataflow.some(Boolean)) {
                // [spec, dataflow], [dataflow, spec]
                if (isSpecification[0]) [specification, dataflow] = objects;
                else [dataflow, specification] = objects;
            }

            let isValidSpec = true;
            if (specification) {
                if (externalApplicationManager.usesBackend()) {
                    NotificationHandler.showToast(
                        'warning',
                        'Specification is managed by an external application.',
                    );
                    isValidSpec = false;
                } else {
                    const errors = await updateEditorSpecification(specification);
                    isValidSpec = !errors.length;
                }
            }

            if (isValidSpec && dataflow) {
                await updateDataflow(dataflow);
            }
            updateVisibleGroups();
        }, setLoad);

        const onDrop = async (event) => {
            event.preventDefault();
            event.dataTransfer.dropEffect = 'copy';
            const files = [];
            if (event.dataTransfer.items) {
                [...event.dataTransfer.items].filter(
                    (item) => item.kind === 'file',
                ).forEach(
                    (item) => files.push(item.getAsFile()),
                );
            } else {
                files.push(...event.dataTransfer.files);
            }
            await loadFiles(files);
        };

        const center = () => {
            editorManager.centerZoom();
        };

        const returnFromSubgraph = () => {
            editorManager.returnFromSubgraph();
        };

        const showWelcome =
            computed(() => editorManager.baklavaView.welcome &&
                !editorManager.baklavaView.editor.readonly &&
                !nodes.value.length &&
                !editorManager.editor.isInSubgraph());

        return {
            el,
            counter,
            selectedNodes,
            nodeContainerStyle,
            onRightPointerUp,
            onDrop,
            nodes,
            keyDown,
            keyUp,
            selectNode,
            currentView: editorManager.currentView,
            currentViewName,
            rectangleSelection,
            greyedOutNodes,
            temporaryConnection: temporaryConnection.temporaryConnection,
            temporaryConnectionRender: temporaryConnection.render,
            mouseWheel: panZoom.onMouseWheel,
            dragging: panZoom.dragging,
            changeHoveredConnections,
            highlightConnections,
            connRefs,
            clearHighlight,
            readonly,
            hideHud,
            scale,
            visibleConnections,
            visibleNodes,
            visibleGroups,
            updateGroupName,
            ignoredNodes,
            highlightInterfaces,
            editorStyle,
            zoomIn: panZoom.onZoomIn,
            zoomOut: panZoom.onZoomOut,
            center,
            returnFromSubgraph,
            isInSubgraph,
            preview,
            showWelcome,
            loadFiles,
            validating: editorManager.validating,
            ignoredInterfacesTypes,
            contextMenu,
            updateContextMenu,
            updateVisibleGroups,
            updateGroupsOf,
            ignoredInterfacesRef,
            handleStartDrag,
            handleStopDrag,
            draggedNode,
        };
    },
});
</script>
