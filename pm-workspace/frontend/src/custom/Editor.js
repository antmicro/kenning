/*
 * Copyright (c) 2022-2026 Antmicro <www.antmicro.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 */

/*
 * Custom pipeline editor - Implements logic for adding, removing, editing nodes and
 * connections between them.
 * Inherits from baklavajs/core/src/editor.ts
 */

/* eslint-disable max-classes-per-file */

import { Editor, Graph } from '@baklavajs/core';
import { toRaw, nextTick, reactive } from 'vue';

import { useGraph, useViewModel } from '@baklavajs/renderer-vue';

import createPipelineManagerGraph from './CustomGraph.js';
import LayoutManager from '../core/LayoutManager.js';
import { suppressHistoryLogging } from '../core/History.ts';
import CreateCustomGraphNodeType, { prepareSubgraphInstance } from './CustomGraphNode.js';
import { ir } from '../core/interfaceRegistry.ts';
import { hexToRGB } from '../core/nodeCreation/nodeColors.js';
import GraphTemplate from './CustomGraphTemplate.js';

import globalProperties from '../globalProperties.ts';

/* eslint-disable no-param-reassign */
/* eslint-disable no-underscore-dangle */
export default class PipelineManagerEditor extends Editor {
    preview = false;

    _hideHud = false;

    get hideHud() {
        return this._hideHud || this.preview;
    }

    set hideHud(val) {
        this._hideHud = val;
    }

    _readonly_before_entering = undefined;

    _readonly = false;

    get readonly() {
        return this._readonly || this.preview;
    }

    set readonly(val) {
        this._readonly = val;
    }

    allowLoopbacks = false;

    nodeIcons = new Map();

    nodeColors = new Map();

    defaultView = 'default';

    renderViews = [this.defaultView];

    baseURLs = new Map();

    baseIconUrls = new Map();

    nodeURLs = new Map();

    nodeContextMenuActions = new Map();

    nodeStyles = new Map();

    nodeLists = new Map();

    layoutManager = new LayoutManager();

    subgraphStack = [];

    parentNodes = new Map();

    currentView = this.defaultView;

    // node types created in visual-editor
    additionalNodeTypes = new Set();

    subgraphStackGraphTypeEnum = {
        SUBGRAPH: 'subgraph',
        RELATEDGRAPH: 'relatedGraph',
    };

    editorManager = null;

    newRootGraph = undefined;

    // We load every graph in
    // specification
    preloadedGraphs = reactive([]);

    addPreloadedGraph(graphSpec) {
        this.preloadedGraphs.push({
            spec: graphSpec,
        });
    }

    loadPreloadedGraph(graphSpec) {
        if (graphSpec.spec?.editor !== undefined) {
            this.switchToGraph(graphSpec.spec);
        } else {
            const graphObject = new Graph(this);
            const errors = graphObject.load(graphSpec.spec, true);

            if (errors.length === 0) {
                graphObject.setSpecGraph(true);
                graphSpec.spec = graphObject;
                this.switchToGraph(graphSpec.spec);
            }
        }
    }

    removePreloadedGraphs() {
        this.preloadedGraphs.clear();
    }

    registerGraph(graph) {
        const customGraph = createPipelineManagerGraph(graph);
        super.registerGraph(customGraph);
    }

    resetAllGraphsToSave() {
        this.graphs.forEach((graph) => { graph.toSave = true; });
    }

    /**
     * Saves the state (nodes, connections, layout) of specification graphs.
     * @return {Object} State of the spec graphs in the editor.
     */
    savePreloadedGraphsState() {
        const NameToId = new Map(
            this.preloadedGraphs.map((g) => [g.spec.name, g.spec.id]),
        );

        const preloadedGraphsLoaded = this.preloadedGraphs.filter((g) =>
            g.spec?.editor !== undefined).map((g) => g.spec);
        const preloadedGraphsNotLoaded = this.preloadedGraphs.filter((g) =>
            g.spec?.editor === undefined).map((g) => structuredClone(toRaw(g.spec)));

        const graphs = [
            ...preloadedGraphsNotLoaded,
            ...preloadedGraphsLoaded.map((g) => g.save()),
        ];
        // Assign a proper subgraph
        graphs.forEach((g) => {
            g.nodes.forEach((node) => {
                if (node?.subgraph !== undefined && node?.graphState !== undefined) {
                    const id = NameToId.get(node.graphState.name);
                    node.subgraph = id;
                }
                delete node.graphState;
            });
        });

        return graphs;
    }

    /**
     * Saves the state (nodes, connections, layout) of all graphs in the editor.
     * @return {Object} State of the graphs in the editor.
     * @throws {Error} Throws if there are issues switching to a subgraph.
     */
    save() {
        const lastView = this.currentView;
        this.currentView = this.defaultView;
        const graphs = Array.from(this.graphs).filter((g) => !g.specGraph);

        const graphMap = new Map(
            graphs.map((g) => [g.id, g]),
        );

        const dataflowState = { graphs: [] };

        const visitedGraphs = new Set();

        const saveGraph = (graph) => {
            if (graph.toSave && !visitedGraphs.has(graph.id)) {
                const currentGraphState = graph.save();
                currentGraphState.panning = graph.panning;
                currentGraphState.scaling = graph.scaling;

                currentGraphState.nodes.forEach((node) => {
                    node.color = this.getNodeColor(node);
                });

                visitedGraphs.add(graph.id);
                currentGraphState.nodes.forEach((node) => {
                    if (node?.subgraph !== undefined) {
                        const graphFound = graphMap.get(node.subgraph);
                        if (graphFound !== undefined) {
                            const lastoSave = graphFound.toSave;
                            graphFound.toSave = true;
                            saveGraph(graphFound);
                            graphFound.toSave = lastoSave;
                        }
                        delete node.graphState;
                    }
                });

                dataflowState.graphs.push(currentGraphState);
            }
        };
        graphs.forEach((graph) => {
            saveGraph(graph);
        });

        if (this._graph.toSave) {
            dataflowState.entryGraph = this._graph.id;
        } else if (dataflowState.graphs.length > 0) {
            dataflowState.entryGraph = dataflowState.graphs[0].id;
        }

        this.currentView = lastView;
        return dataflowState;
    }

    /**
     *  Adds a new empty graph to the editor.
     */
    addNewGraph() {
        const graph = new Graph(this);
        graph.name = 'New Graph';

        const template = new GraphTemplate(graph, this);
        this.editorManager.baklavaView.editor.switchToGraph(template, false);
    }

    /**
     * Cleans all graphs in the editor.
     * @param Determines whether the cleaning process should be stored in history
     */
    deepCleanEditor(suppressHistory = true) {
        this.subgraphStack.forEach(this.backFromSubgraph.bind(this));
        this.cleanEditor(suppressHistory);
    }

    /**
     * Cleans up the current graph current graph editor.
     * @param {bool} suppressHistory Determines whether the cleaning process should
     * be stored in history.
     */
    cleanEditor(suppressHistory = true) {
        while (this.isInSubgraph()) {
            this.backFromSubgraph();
        }
        const graphInstance = this._graph;

        suppressHistoryLogging(suppressHistory);
        for (let i = graphInstance.connections.length - 1; i >= 0; i -= 1) {
            graphInstance.removeConnection(graphInstance.connections[i]);
        }
        for (let i = graphInstance.nodes.length - 1; i >= 0; i -= 1) {
            graphInstance.removeNode(graphInstance.nodes[i]);
        }
        this.graphName = undefined;
        suppressHistoryLogging(false);
    }

    unregisterGraphs() {
        this._graph.disableDestroy = true;
        [...this.graphs]
            .filter((graph) => graph !== this._graph && !graph.specGraph)
            .forEach((graph) => {
                this.unregisterGraph(graph);
                graph.destroy?.();
            });
        this._graph.disableDestroy = false;
        this.subgraphStack = [];
    }

    unregisterNodes() {
        this.nodeTypes.forEach((_, nodeKey) => {
            this.unregisterNodeType(nodeKey);
        });
    }

    registerNodeType(type, options) {
        if (this.events.beforeRegisterNodeType.emit({ type, options }).prevented) {
            return;
        }

        const nodeInstance = new type(); // eslint-disable-line new-cap
        this._nodeTypes.set(nodeInstance.type, {
            type,
            category: options?.category ?? 'default',
            title: options?.title ?? nodeInstance.title,
            isCategory: options?.isCategory ?? false,
            color: options?.color,
            subgraphId: options?.subgraphId,
            relatedGraphs: options?.relatedGraphs,
            style: options?.style,
            pill: options?.pill,
        });

        this.events.registerNodeType.emit({ type, options });
    }

    /**
     * Loads the dataflow into the editor.
     *
     * @param state dataflow to load
     * @param {bool} preventCentering determines whether to center the graph after loading
     * @param {bool} loadOnly determines whether to load the graph only without adjusting
     * the graph rendering. Can be used when validating graphs without their browser
     * representation.
     * @param templateName {string|null} name of the template, if the graph is a template
     * @param centerAtTop determines whether to center the editor at the topmost node
     * after loading
     * @returns list of errors that occurred during loading
     */
    async load(
        state,
        preventCentering = false,
        loadOnly = false,
        templateName = null,
        centerAtTop = false,
    ) {
        // All subgraphs should be unregistered to avoid conflicts later when trying to
        // load into subgraph (in that case there may be two subgraphs with the same ID, one
        // of them from the previous session).
        this.unregisterGraphs();
        ir.clearRegistry();
        // Load the node state as it is, wait until vue renders new nodes so that
        // node dimensions can be retrieved from DOM elements and then update the
        // location based on autolayout results. The editor is set to readonly
        // during loading to prevent any changes that may happen between graph load
        // and layout computation
        const readonlySetting = this.readonly;
        this.readonly = true;

        const result = {
            errors: [],
            warnings: [],
        };

        if (!state.graphs.length) {
            return ['No graphs found'];
        }

        let entryGraph;
        if (state.entryGraph) {
            entryGraph = state.graphs.find(
                (graph) => graph.id === state.entryGraph,
            );

            if (!entryGraph) {
                return [`No entry graph found of id '${state.entryGraph}'`];
            }
        } else {
            entryGraph = state.graphs[0]; // eslint-disable-line prefer-destructuring
        }
        const { panning, scaling } = entryGraph;

        const usedSubgraphs = new Set();
        const usedSubgraphParent = {};

        this.nodeColors.clear();
        state.graphs.forEach((graph) => {
            graph.nodes.forEach((n) => {
                n.views?.forEach((v) => {
                    if (!this.renderViews.find((vv) => vv === v.name)) {
                        this.renderViews.push(v.name);
                    }
                });
                if (n.subgraph !== undefined) {
                    if (!this.isGraphNode(n.name)) {
                        result.warnings.push([`Node ${n.name} is not graph node, although it has defined subgraph property.`]);
                        return;
                    }

                    const fittingTemplate = state.graphs.filter(
                        (template) => template.id === n.subgraph,
                    );
                    if (fittingTemplate.length !== 1) {
                        result.errors.push([`Expected exactly one template with ID ${n.name}, got ${fittingTemplate.length}`]);
                    }

                    const { subgraphId } = this.nodeTypes.get(n.name) ?? {};
                    const isInstantiated = n.subgraph !== subgraphId;

                    usedSubgraphs.add(n.subgraph);
                    usedSubgraphParent[n.subgraph] = graph.id;

                    if (isInstantiated || subgraphId === undefined) {
                        [n.graphState] = fittingTemplate;
                    }
                }
                n.relatedGraphs
                    ?.filter(({ id }) => state.graphs.find((el) => el.id === id) === undefined)
                    .forEach(({ id }) => result.errors.push([`The related graph of id ${id} is not defined`]));
                this.setNodeColor(n.id, n.color);
            });
        });
        let graphRootLoad = entryGraph;
        while (usedSubgraphParent[graphRootLoad.id] !== undefined) {
            const searchedId = usedSubgraphParent[graphRootLoad.id];
            graphRootLoad = state.graphs?.find((g) =>
                g.id === searchedId);
        }
        graphRootLoad ??= entryGraph;

        try {
            if (result.errors.length && !globalProperties.softLoad) return result.errors;

            state.graphs?.forEach((graph) => {
                if (!usedSubgraphs.has(graph.id) && graphRootLoad.id !== graph.id) {
                    const graphObject = new Graph(this);
                    result.errors.push(...graphObject.load(graph, true));
                    this.registerGraph(graphObject);
                }
            });

            if (!result.errors.length || globalProperties.softLoad) {
                let graphToLoad;
                if (!templateName) {
                    graphToLoad = graphRootLoad;
                } else {
                    // eslint-disable-next-line new-cap
                    const graphNode = new (this._nodeTypes.get(templateName)).type();

                    const {
                        state: preparedSubgraphState, errors: prepareSubgraphErrors,
                    } = prepareSubgraphInstance(graphNode.template);

                    result.errors.push(...prepareSubgraphErrors);
                    graphToLoad = preparedSubgraphState;
                }
                const noErrors = !result.errors.length || globalProperties.softLoad;
                if (noErrors && !usedSubgraphs.has(graphToLoad.id)) {
                    const loadAll = (graphRootLoad === graphToLoad);
                    result.errors.push(...this._graph.load(graphToLoad, loadAll));
                }
            }
            this.graphs.forEach((graph) => {
                const g = state.graphs?.find((grph) => grph.id === graph.id);
                graph.groups = g?.groups ? g.groups.map((group) => ({ ...group })) : [];
            });
        } catch (err) {
            // If anything goes wrong during dataflow loading, the editor is cleaned and an
            // appropriate error is returned.
            this.cleanEditor();
            this.readonly = readonlySetting;
            result.errors.push(err.toString());
            return result;
        }
        if (Array.isArray(result.errors) && result.errors.length && !globalProperties.softLoad) {
            this.cleanEditor();
            this.readonly = readonlySetting;
            return result;
        }
        this.events.loaded.emit();
        this.graphName = entryGraph.name;
        this.readonly = readonlySetting;

        // If the editor is run outside of a browser, then
        // all functionality that is after this line will fail,
        // as it changes the way the graph is rendered in the browser
        if (typeof window === 'undefined' || loadOnly) return result;

        const dfs = (subgraph, path) => {
            if (subgraph?.nodes !== undefined) {
                for (let i = 0; i < subgraph.nodes.length; i += 1) {
                    if (subgraph.nodes[i].subgraph !== undefined) {
                        if (subgraph.nodes[i].subgraph.id === entryGraph.id) {
                            return [...path, subgraph.nodes[i]];
                        }
                        const returnedPath = dfs(
                            subgraph.nodes[i].subgraph,
                            [...path, subgraph.nodes[i]],
                        );
                        if (returnedPath.length) {
                            return returnedPath;
                        }
                    }
                }
            }
            return [...path];
        };

        // Finding a path to the defined entry and switching to it sequentially
        const path = dfs(this._graph, []);

        try {
            if (this._graph.id !== entryGraph.id) {
                path.forEach((node) => {
                    this.switchToSubgraph(node);
                });
            }
        } catch (err) {
            result.errors.push(err.toString());
        }

        if (this.layoutManager.layoutEngine.activeAlgorithm !== 'NoLayout') {
            await nextTick();
            await this.applyAutolayout(false);
            state = this.hooks.load.execute(state);
        }

        // We need graph switched and sidebar rendered for autozoom
        await nextTick();
        if (panning !== undefined) {
            this._graph.panning = panning;
        }
        if (scaling !== undefined) {
            this._graph.scaling = scaling;
        }
        if (!preventCentering &&
            scaling === undefined &&
            panning === undefined &&
            !centerAtTop) {
            this.centerZoom();
        }
        if (centerAtTop) {
            let topY = this.graph.nodes[0].position.y;
            let topNode = this.graph.nodes[0];
            this.graph.nodes.forEach((node) => {
                if (node.position.y < topY) {
                    topY = node.position.y;
                    topNode = node;
                }
            });

            const { editorWidth, sideBarWidth } = PipelineManagerEditor.editorSize();

            // Due to scaling editor coordinates are also scaled (in this case by a factor of 2)
            const editorCenter = editorWidth;

            this._graph.scaling = 0.5;
            this._graph.panning = {
                x: -topNode.position.x + editorCenter + sideBarWidth,
                y: -topNode.position.y + 75,
            };
        }
        this.graphs.forEach((graph) => {
            if (graph.graphNode !== undefined) {
                const node = graph.graphNode;
                const inputs = { ...node.inputs };
                const outputs = { ...node.outputs };

                const interfaces = [
                    ...Object.values(inputs),
                    ...Object.values(outputs),
                ];
                const connections = node.graphInstance.connections.filter(
                    (c) => interfaces.includes(c.from) || interfaces.includes(c.to),
                );

                node.updateExposedInterfaces(undefined, undefined, true);

                const newInputs = node.inputs;
                const newOutputs = node.outputs;

                const externalNameMap = new Map();
                const idMap = new Map();
                Object.keys(newInputs).forEach((key) => {
                    idMap[newInputs[key].id] = inputs[key].id ?? newInputs[key].id;
                    externalNameMap[idMap[newInputs[key].id]] =
                        inputs[key].externalName ?? newInputs[key].externalName;
                });
                Object.keys(newOutputs).forEach((key) => {
                    idMap[newOutputs[key].id] = outputs[key].id ?? newOutputs[key].id;
                    externalNameMap[idMap[newOutputs[key].id]] =
                        outputs[key].externalName ?? newOutputs[key].externalName;
                });
                node.subgraph.nodes.forEach((n) => {
                    Object.values(n.inputs).forEach((intf) => {
                        intf.id = idMap[intf.id] ?? intf.id;
                    });
                    Object.values(n.outputs).forEach((intf) => {
                        intf.id = idMap[intf.id] ?? intf.id;
                    });
                });
                // restore ids described in configuration, not generated ones
                node.updateExposedInterfaces(undefined, undefined, true);

                const newInterfaces = [...Object.values(node.inputs),
                    ...Object.values(node.outputs)];
                // restore old external names
                newInterfaces.forEach((intf) => {
                    intf.externalName = externalNameMap[intf.id];
                });
                // restore connections that might have disconnected after the first update
                connections.forEach((conn) => {
                    conn.from = newInterfaces.find((intf) => intf.id === conn.from.id) ?? conn.from;
                    conn.to = newInterfaces.find((intf) => intf.id === conn.to.id) ?? conn.to;
                    node.graphInstance.addConnection(conn.from, conn.to);
                });
            }
        });
        return result;
    }

    privatizeInterface(graphId, intf) {
        if (intf.externalName === undefined) return;

        let graph = [...this.graphs].find((g) => g.id === graphId);
        let graphNode = graph?.graphNode; // eslint-disable-line prefer-destructuring

        const { externalName } = intf;

        intf.externalName = undefined;
        if (graphNode === undefined) return;

        const { graphIds, sharedInterface } = ir.getRegisteredInterface(intf.id);
        const graphIdIdx = graphIds.findIndex((id) => id === graphNode.graph.id);
        graphIds.splice(graphIdIdx, graphIds.length - graphIdIdx);

        // Get associated interfaces

        const interfaces = [...Object.values(graphNode.inputs),
            ...Object.values(graphNode.outputs),
        ];
        const exposedInterfaces = interfaces.filter((inf) => inf.name === externalName);

        // Privatize exposed interface in parent graphs
        exposedInterfaces.forEach((inf) => {
            // get graph id

            const graphNested = Array.from(this.graphs)
                .filter((g) => g.nodes.some((n) => n.id === inf.nodeId));

            if (graphNested.length !== 1) {
                return;
            }

            this.privatizeInterface(graphNested[0].id, inf);
        });

        graphNode.updateExposedInterfaces(undefined, undefined, true);

        // Update all graphs that used this exposed interface
        for (let i = graphIdIdx + 1; i < graphIds.length; i += 1) {
            const id = graphIds[i];
            graph = [...this.graphs].find((g) => g.id === id);
            graphNode = graph.graphNode; // eslint-disable-line prefer-destructuring
            graphNode.updateExposedInterfaces(undefined, undefined, true);
        }

        // If sharedInterface is the same as the interface that is privatized, it means
        // that the interface is not shared anymore and its entry should be deleted.
        if (sharedInterface === intf) {
            ir.deleteRegisteredInterface(intf.id);
        }
    }

    /**
     * Exposes passed interface under the `name` name. The node that has the interface has to
     * be in the graph with the `graphId` ID.
     *
     * @param {string} graphId graph which has the node with the interface
     * @param {Object} intf interface to expose
     * @param {string} name name under which the interface will be exposed. If set to `undefined`,
     * external name will be inferred from the name of the interface.
     */
    exposeInterface(graphId, intf, name = undefined) {
        const graph = [...this.graphs].find((g) => g.id === graphId);
        const graphNode = graph.graphNode; // eslint-disable-line prefer-destructuring
        intf.externalName = name ?? graph.resolveNewExposedName(intf.name);
        if (graphNode === undefined) return;
        // After changing the external name, the interface has to be updated in the
        // graph node to reflect the changes in the graph.
        graphNode.updateExposedInterfaces();
    }

    /**
     * Iterates over all nodes of the currently displayed graph
     * to find all exposed properties in the graph.
     *
     * @returns {Array} list of exposed properties in displayed graph
     */
    getExposedProperties() {
        return this._graph.nodes
            .map((n) => Object.values(n.inputs)).flat()
            .filter((intf) => !intf.port && intf.externalName);
    }

    /**
     * Calculates the width, height of the editor and the width of the sidebar.
     * It is assumed that the view is rendered in the browser,
     * otherwise an error will be thrown.
     *
     * @returns {Object} object with calculated sizes
     */
    static editorSize() {
        if (typeof document === 'undefined') {
            throw new Error('The editor is in browserless mode. Cannot obtain editor size.');
        }

        const terminalHeight =
            document.getElementsByClassName('terminal-wrapper')[0]?.offsetHeight ?? 0;
        const navbarHeight = document.getElementsByClassName('wrapper')[0]?.offsetHeight ?? 0;
        const nodePalette = document.getElementsByClassName('baklava-node-palette');
        let sideBarWidth = 0;
        if (nodePalette.length !== 0) {
            const paletteRect = nodePalette[0].getBoundingClientRect();
            sideBarWidth = Math.max(paletteRect.right, 0);
        }

        const editorHeight = window.innerHeight - terminalHeight - navbarHeight;
        const editorWidth = window.innerWidth - sideBarWidth;

        return {
            editorWidth,
            editorHeight,
            sideBarWidth,
        };
    }

    getIgnoredLayers(graphId = undefined) {
        const { viewModel } = useViewModel();
        const { layers } = viewModel.value;
        const { ignoredLayers } = viewModel.value;

        const _graphId = graphId ?? this.graph.id;

        const outputLayers = [];

        layers.forEach((layer) => {
            if ((ignoredLayers.get(_graphId) ?? new Set()).has(layer.name)) {
                outputLayers.push(layer);
            }
        });

        return outputLayers;
    }

    getIgnoredNodes(graphId = undefined, layers = undefined) {
        const _layers = (layers ?? this.getIgnoredLayers(graphId))
            .filter((layer) => layer.nodeLayers);
        const temp = new Set();
        _layers.forEach((layer) => {
            layer.nodeLayers.forEach(temp.add, temp);
        });
        return temp;
    }

    getIgnoredInterfaces(graphId = undefined, layers = undefined) {
        const _layers = (layers ?? this.getIgnoredLayers(graphId))
            .filter((layer) => layer.nodeInterfaces);
        const temp = new Set();
        _layers.forEach((layer) => {
            layer.nodeInterfaces.forEach(temp.add, temp);
        });
        return temp;
    }

    /**
     * Centers the currently displayed graph in the editor.
     * The function calculates the scaling and panning values to center the graph
     * in the editor and sets them in the graph.
     * The function assumes that the editor is rendered in the browser,
     * otherwise an error will be thrown.
     */
    centerZoom() {
        if (!Array.isArray(this._graph.nodes) || this._graph.nodes.length === 0) return;
        if (typeof document === 'undefined') {
            throw new Error('The editor is in browserless mode. Cannot obtain editor size.');
        }

        const {
            editorWidth,
            editorHeight,
            sideBarWidth,
        } = PipelineManagerEditor.editorSize();

        const {
            graphHeight,
            graphWidth,
            leftmostX,
            topmostY,
        } = this._graph.size();

        const margin = 100;

        const scalingY = editorHeight / (graphHeight + 2 * margin);
        const scalingX = editorWidth / (graphWidth + 2 * margin);

        if (scalingX > scalingY) {
            const graphCenter = (graphWidth + 2 * margin) / 2;
            const editorCenter = (editorWidth / 2) * (1 / scalingY);

            const translationX = editorCenter - graphCenter;

            this._graph.panning = {
                x: -(leftmostX - margin - translationX - sideBarWidth / scalingY),
                y: -(topmostY - margin),
            };
            this._graph.scaling = scalingY;
        } else {
            const graphCenter = (graphHeight + 2 * margin) / 2;
            const editorCenter = (editorHeight / 2) * (1 / scalingX);

            const translationY = editorCenter - graphCenter;

            this._graph.panning = {
                x: -(leftmostX - margin - sideBarWidth / scalingX),
                y: -(topmostY - margin - translationY),
            };
            this._graph.scaling = scalingX;
        }
    }

    nodeURLsEmpty() {
        return this.nodeURLs.size === 0;
    }

    getNodeURLs(nodeName) {
        const urls = this.nodeURLs.get(nodeName) || {};

        const fullUrls = [];
        Object.entries(urls).forEach(([urlName, url]) => {
            const t = { ...this.baseURLs.get(urlName) };
            t.url += url;
            fullUrls.push(t);
        });

        return fullUrls;
    }

    _getFromView(node, field) {
        const view = this.currentView;
        const viewEntry = node.views?.find((v) => v.name === view);
        if (viewEntry && viewEntry.style) {
            const val = this.getNodeStyle(viewEntry.style)?.[field];
            if (val) {
                return val;
            }
        }
        return undefined;
    }

    getNodeContextMenuActions(nodeName) {
        return this.nodeContextMenuActions.get(nodeName) || [];
    }

    getNodeIconPath(nodeName) {
        return this.nodeIcons.get(nodeName) || undefined;
    }

    getPillText(node) {
        const viewPill = this._getFromView(node, 'pill');
        if (viewPill?.text) return viewPill.text;
        const nodeType = this.nodeTypes.get(node.type);
        if (nodeType?.pill !== undefined) return nodeType.pill?.text;
        if (nodeType?.style !== undefined) return this.getNodeStyle(nodeType.style)?.pill?.text;
        return undefined;
    }

    getNodeCategory(nodeName) {
        return this.nodeTypes.get(nodeName).category || undefined;
    }

    getNodeColor(node) {
        const viewColor = this._getFromView(node, 'color');
        if (viewColor) return viewColor;
        const nodeColor = this.nodeColors.get(node.id);
        if (nodeColor !== undefined) return nodeColor;

        const nodeType = this.nodeTypes.get(node.type);
        if (nodeType?.color !== undefined) return nodeType.color;
        if (nodeType?.style !== undefined) return this.getNodeStyle(nodeType.style)?.color;
        return undefined;
    }

    setNodeColor(nodeId, color) {
        if (color !== undefined) {
            this.nodeColors.set(nodeId, color);
        }
    }

    getNodeDisplayedPosition(node) {
        const curName = this.currentView;
        const entry = node.views?.find((v) => v.name === curName);
        return entry ? entry.position : node.position;
    }

    getNodeMinimal(node) {
        const viewMin = this._getFromView(node, 'minimal');
        if (viewMin) return viewMin;
        const nodeName = node.type;
        const nodeTypeStyle = this.getNodeTypeStyle(nodeName);
        if (nodeTypeStyle !== undefined) {
            const nodeStyle = this.getNodeStyle(nodeTypeStyle);
            const hasShape = nodeStyle?.shape !== undefined;
            if (nodeStyle?.minimal === undefined && hasShape) {
                return true;
            }

            return nodeStyle?.minimal ?? false;
        }
        return false;
    }

    getNodeInterfacePositions(nodeName) {
        return this.getNodeStyle(this.getNodeTypeStyle(nodeName))?.positions ?? {};
    }

    getCustomNodeTitlePosition(nodeName) {
        return this.getNodeStyle(this.getNodeTypeStyle(nodeName))?.title?.position;
    }

    getNodeStyleFitTitle(nodeName) {
        return this.getNodeStyle(this.getNodeTypeStyle(nodeName))?.title?.fit ?? false;
    }

    getNodeStyleHideSubTitle(nodeName) {
        return this.getNodeStyle(this.getNodeTypeStyle(nodeName))?.title?.hideSubTitle ?? false;
    }

    getNodeStyleTitleOffset(nodeName) {
        return this.getNodeStyle(this.getNodeTypeStyle(nodeName))?.title?.offset;
    }

    getNodeStyleTitlePadding(nodeName) {
        return this.getNodeStyle(this.getNodeTypeStyle(nodeName))?.title?.padding;
    }

    getShape(nodeName) {
        const nodeStyle = this.getNodeTypeStyle(nodeName);
        if (nodeStyle !== undefined) return this.getNodeStyle(nodeStyle)?.shape;
        return undefined;
    }

    getNodeClean(node) {
        const viewMin = this._getFromView(node, 'clean');
        if (viewMin) return viewMin;
        const nodeType = this.nodeTypes.get(node.type);
        if (nodeType?.style !== undefined) return this.getNodeStyle(nodeType.style)?.clean;
        return false;
    }

    getStyleIcon(nodeName) {
        const nodeStyle = this.getNodeTypeStyle(nodeName);
        if (nodeStyle !== undefined) return this.getNodeStyle(nodeStyle)?.icon;
        return undefined;
    }

    getNodeInstanceStyle(node) {
        const nodeType = this.nodeTypes.get(node.type);
        let val;
        const view = this.currentView;
        const viewEntry = node.views?.find((v) => v.name === view);
        if (viewEntry && viewEntry.style) {
            val = viewEntry.style;
        }
        if (val === undefined) {
            val = nodeType.style;
        }
        return val;
    }

    getNodeTypeStyle(nodeType) {
        return this.nodeTypes.get(nodeType)?.style;
    }

    getNodeStyle(style) {
        if (!Array.isArray(style)) {
            style = [style];
        }

        return Object.assign({}, ...style
            .map((styleName) => this.nodeStyles.get(styleName))
            .filter((value) => value !== undefined));
    }

    getPillColor(node) {
        const viewPill = this._getFromView(node, 'pill');
        if (viewPill?.color) return viewPill.color;
        const nodeType = this.nodeTypes.get(node.type);
        const pill = nodeType?.pill !== undefined
            ? nodeType.pill
            : this.getNodeStyle(nodeType.style).pill;
        if (pill !== undefined) return pill.color ?? '#ffffff';
        return undefined;
    }

    /* eslint-disable class-methods-use-this */
    getTextColor(color) {
        if (!color) {
            return 'white';
        }

        // calculate lightness
        const rgb = hexToRGB(color);
        if (rgb === null) {
            return 'white';
        }
        const r = parseInt(rgb[1], 16) / 255;
        const g = parseInt(rgb[2], 16) / 255;
        const b = parseInt(rgb[3], 16) / 255;
        const lightness = (Math.max(r, g, b) + Math.min(r, g, b)) / 2;

        if (lightness > 0.5) {
            return 'black';
        }

        return 'white';
    }

    isGraphNode(nodeName) {
        if (!this.nodeTypes.has(nodeName)) return false;
        return this.nodeTypes.get(nodeName).subgraphId !== undefined;
    }

    nodeHasRelatedGraphs(node) {
        return (node.relatedGraphs ?? this.nodeTypes.get(node.type)?.relatedGraphs) !== undefined;
    }

    addGraphTemplate(template, graphNode) {
        if (this.events.beforeAddGraphTemplate.emit(template).prevented) {
            return;
        }
        this._graphTemplates.push(template);
        this.graphTemplateEvents.addTarget(template.events);
        this.graphTemplateHooks.addTarget(template.hooks);

        if (graphNode.extends) {
            const intfsInherited = this.editorManager.findInheritedInterfaces(graphNode.name)
                .filter((intf) => !graphNode.interfaces?.find(
                    (i) => i.name === intf.name && i.side === intf.side),
                )?.map((intf) => ({ ...intf, inherited: true }));
            const propsInherited = this.editorManager.findInheritedProperties(graphNode.name)
                .filter((prop) =>
                    !graphNode.properties?.find((p) => p.name === prop.name))
                ?.map((prop) => ({ ...prop, inherited: true }));
            const inheritedAttributes = this.editorManager
                .findSimpleInheritedAttributes(graphNode.name);

            graphNode.interfaces = [...graphNode.interfaces ?? [], ...intfsInherited];
            graphNode.properties = [...graphNode.properties ?? [], ...propsInherited];
            graphNode.layer = graphNode.layer ?? inheritedAttributes.layer;
        }

        const customGraphNodeType = CreateCustomGraphNodeType(
            template, graphNode, this.editorManager,
        );
        this.registerNodeType(customGraphNodeType, {
            category: graphNode.category,
            title: graphNode.title,
            isCategory: graphNode.isCategory,
            color: graphNode.color,
            style: graphNode.style,
            pill: graphNode.pill,
            subgraphId: graphNode.subgraphId,
            relatedGraphs: graphNode?.relatedGraphs,
        });

        this.events.addGraphTemplate.emit(template);
    }

    switchGraph(subgraphNode) {
        if (this._switchGraph === undefined) {
            const { switchGraph } = useGraph();
            this._switchGraph = (g) => {
                const prev = this._graph;
                prev.disableDestroy = true;
                this._graph = g;
                switchGraph(g);
                prev.disableDestroy = false;
            };
        }
        // disable history logging for the switch - don't push nodes being created here
        suppressHistoryLogging(true);

        if (subgraphNode.subgraph === undefined) {
            throw Error(
                `Node "${subgraphNode.name}" does contain a subgraph.`,
            );
        }
        this._switchGraph(subgraphNode.subgraph);
        this.graphName = this._graph.name;

        suppressHistoryLogging(false);
        nextTick().then(() => {
            const graph = this.graph.save();
            this.layoutManager.registerGraph(graph);
            this.layoutManager
                .computeLayout(graph)
                .then(this.updateNodesPosition.bind(this))
                .then(() => {
                    nextTick().then(() => {
                        if (
                            !this._graph.wasCentered
                        ) {
                            this.centerZoom();
                            this._graph.wasCentered = true;
                        }
                    });
                });
        });
    }

    /**
    * Switch to a subgraph by obtaining the current graph from the subgraph node
    * and pushing the subgraph node to the stack.
    *
    * @param {Node} subgraphNode A subgraph node containing a subgraph,
    * to which a layout should be switched.
    * */
    switchToSubgraph(subgraphNode) {
        if (subgraphNode && subgraphNode.subgraph) {
            this.subgraphStack.push(subgraphNode.graph);
            this.switchGraph(subgraphNode);
            if (subgraphNode.extending?.length) {
                if (this._readonly_before_entering === undefined) {
                    this._readonly_before_entering = this.readonly;
                }
                this.readonly = true;
            }
        }
    }

    /**
    * Switch to a given graph object.
    *
    * @param {object} targetGraph A graph object to which the layout should be switched
    * @param {bool} subgraphStack Determines whether the graph should be added to subgraph stack
    * */
    switchToGraph(targetGraph, subgraphStack = false) {
        if (!targetGraph) return;

        if (this._switchGraph === undefined) {
            const { switchGraph } = useGraph();
            this._switchGraph = (g) => {
                const prev = this._graph;
                prev.disableDestroy = true;
                this._graph = g;
                switchGraph(g);
                prev.disableDestroy = false;
            };
        }
        // disable history logging for the switch - don't push nodes being created here
        suppressHistoryLogging(true);

        if (subgraphStack) {
            this.subgraphStack.push(this._graph);
        } else if (this.subgraphStack.length > 0) {
            this.subgraphStack.pop();
        }
        this._switchGraph(targetGraph);
        this.graphName = this._graph.name;

        nextTick().then(() => {
            const graph = this.graph.save();
            this.layoutManager.registerGraph(graph);
            this.layoutManager
                .computeLayout(graph)
                .then(this.updateNodesPosition.bind(this))
                .then(() => {
                    nextTick().then(() => {
                        if (
                            !this._graph.wasCentered
                        ) {
                            this.centerZoom();
                            this._graph.wasCentered = true;
                        }
                    });
                });
        });

        suppressHistoryLogging(false);
    }

    /**
    * Switch to a related by matching the ID to a loaded graph instancce
    * and pushing the current graph to the stack.
    *
    * @param {relatedGraphID} An ID of the graph, to which a layout should be switched.
    * */
    switchToRelatedGraph(relatedGraphID, graphStack = true) {
        if (this.editorManager?.baklavaView.displayedGraph.id === relatedGraphID) return;
        const relatedGraph = Array.from(this.graphs).find((item) => item.id === relatedGraphID);
        this.switchToGraph(relatedGraph, graphStack);
    }

    /**
     * Switches back from a displayed graph.
     * The function changes the currently displayed graph and propagates changes in interfaces
     * back to the graph node.
     *
     * It also updates the graph node's interfaces to match the ones in the graph.
     * It checks for existing interface nodes, checks which were added, removed and changed
     * and updates the graph node's interfaces accordingly.
     */
    backFromSubgraph() {
        const newGraph = this.subgraphStack.pop(); // eslint-disable-line no-unused-vars

        suppressHistoryLogging(true);

        this._switchGraph(newGraph);
        this.graphName = this._graph.name;
        if (this._readonly_before_entering !== undefined) {
            this.readonly = this._readonly_before_entering;
            this._readonly_before_entering = undefined;
        }

        suppressHistoryLogging(false);
    }

    /**
    * Changes top level graph to one of the other graphs in the editor.
    *
    * @param {string} graphId ID of the new top level graph.
    * */
    changeTopLevelGraph(graphId) {
        const rootGraph = Array.from(this.graphs).find((graph) => graph.id === graphId);
        this.newRootGraph = rootGraph;
    }

    findInterface(intfId) {
        for (let i = 0; i < this.graph.nodes.length; i += 1) {
            const foundIntf = Object.values(this.graph.nodes[i].inputs).concat(
                Object.values(this.graph.nodes[i].outputs),
            ).find(
                (intf) => intf.id === intfId,
            );
            if (foundIntf) return foundIntf;
        }
        return null;
    }

    unwrapSubgraph(node) {
        const subgraphNodes = Object.values(node.subgraph._nodes);
        // Calculate center point of subgraph nodes
        const meanX = subgraphNodes.map((n) => n.position.x).reduce(
            (sum, value) => sum + value, 0,
        ) / subgraphNodes.length;
        const meanY = subgraphNodes.map((n) => n.position.y).reduce(
            (sum, value) => sum + value, 0,
        ) / subgraphNodes.length;
        // Remove selections
        this.graph.selectedNodes = [];
        // Create, reposition and select subgraph nodes
        subgraphNodes.forEach((subgraphNode) => {
            const state = subgraphNode.save();
            const addedNode = this.graph.addNode(subgraphNode);
            if (addedNode) {
                // Set position relative to removed node
                addedNode.position.x += node.position.x - meanX;
                addedNode.position.y += node.position.y - meanY;
                this.graph.selectedNodes.push(addedNode);
                // Reset connections count
                Object.values(addedNode.inputs).concat(
                    Object.values(addedNode.outputs),
                ).forEach(
                    (intf) => { intf.connectionCount = 0; },
                );
                addedNode.load(state);
            }
        });

        const subgraphNodeConnections = this.graph.connections.filter(
            (c) => c.from.nodeId === node.id || c.to.nodeId === node.id,
        );
        this.graph.removeNode(node);
        Object.values(node.subgraph.connections).concat(
            subgraphNodeConnections,
        ).forEach((connection) => {
            if (connection.from.name === 'Connection' || connection.to.name === 'Connection') { return; }

            // Finding interfaces in newly added nodes that correspond to the ones in the connection
            const fromInterface = this.findInterface(connection.from.id);
            const toInterface = this.findInterface(connection.to.id);

            if (fromInterface && toInterface) {
                const createdConnection = this.graph.addConnection(fromInterface, toInterface);
                (connection.anchors ?? []).forEach((anchor, index) => {
                    let newAnchor;
                    // Only anchors from within the graph node should be shifted
                    if (node.subgraph.connections.includes(connection)) {
                        newAnchor = {
                            x: anchor.x + node.position.x - meanX,
                            y: anchor.y + node.position.y - meanY,
                        };
                    } else {
                        newAnchor = {
                            x: anchor.x,
                            y: anchor.y,
                        };
                    }

                    this.graph.addAnchor(newAnchor, createdConnection, index);
                });
            }
        });
    }

    isInSubgraph() {
        return this.subgraphStack.length > 0;
    }

    async applyAutolayout(resetLocations = true) {
        const nodeToLayers = new Map();
        const infToType = new Map();
        this.graph.nodes.forEach((n) => {
            if (nodeToLayers.has(n.type)) return;
            nodeToLayers.set(n.type, n.layer);
            const interfaces = Object.entries({ ...n.inputs, ...n.outputs })
                .filter((inf) => inf[1]?.direction !== undefined);

            interfaces.forEach((inf) => {
                infToType.set(inf[1].name, inf[1].type);
            });
        });

        const state = this.graph.save();
        if (resetLocations) {
            state.nodes.forEach((node) => {
                node.position = undefined;
            });
        }
        const ignoredNodeSet = this.getIgnoredNodes(this.graph.id);
        const ignoredInterfacesSet = this.getIgnoredInterfaces(this.graph.id);

        state.nodes = state.nodes.filter((n) => !ignoredNodeSet.has(nodeToLayers.get(n.name)));
        state.nodes.forEach((n) => {
            n.interfaces = n.interfaces.filter((inf) => !ignoredInterfacesSet
                .has(infToType.get(inf.name)));
        });

        this.layoutManager.registerGraph(state);
        const updatedGraph = await this.layoutManager.computeLayout(state);
        this.updateNodesPosition(updatedGraph);
    }

    getPosition(arr) {
        const position = arr.findIndex((el, i) => el !== i);
        return position === -1 ? arr.length : position;
    }

    async swapInterfaces() {
        const graphs = this.save();

        graphs.graphs.forEach((state) => {
            if (state.id !== this.editorManager?.baklavaView.displayedGraph.id) return;
            state.nodes.forEach((node) => {
                const toSidedStubs = node.interfaces.filter((i) =>
                    i.bus?.type === 'twoSided').flatMap((i) => i.bus.stubs ?? []);
                const movableInterfaces = node.interfaces.concat(toSidedStubs);
                movableInterfaces.forEach((intf) => {
                    let connections = state.connections.filter(
                        (conn) => conn?.from === intf.id || conn?.to === intf.id,
                    );
                    const busHasId = (i, id) =>
                        i.bus?.type !== undefined &&
                        i.bus.stubs?.find((s) => s.id === id) !== undefined;
                    if (intf.bus?.type === 'oneSided') {
                        const stubConnections = state.connections.filter(
                            (conn) => busHasId(intf, conn?.from) || busHasId(intf, conn?.to),
                        );
                        connections = connections.concat(stubConnections);
                    }
                    // how many nodes are there on the left
                    // and on the right of the current one
                    let lefts = 0;
                    let rights = 0;
                    connections.forEach((connection) => {
                        const key = (connection.from === intf.id || busHasId(intf, connection.from)) ? 'from' : 'to';
                        const nextIntfId = key === 'from' ? connection.to : connection.from;
                        const nextNode = state.nodes.find(
                            (n) => n.interfaces.some((el) => el.id === nextIntfId
                                || busHasId(el, nextIntfId)),
                        );
                        if (nextNode) {
                            if (nextNode.position.x + nextNode.width > node.position.x) {
                                rights += 1;
                            } else {
                                lefts += 1;
                            }
                        }
                    });
                    const newSide = (() => {
                        if (lefts > rights) return 'left';
                        if (rights > lefts) return 'right';
                        return '';
                    })();
                    if (['left', 'right'].includes(newSide)) {
                        const sidePosition = this.getPosition(
                            node.interfaces.filter(
                                (el) => el.side === newSide &&
                                    el !== intf,
                            ).map(
                                (el) => el.sidePosition,
                            ).sort(
                                (sp1, sp2) => sp1 - sp2,
                            ),
                        );
                        intf.sidePosition = sidePosition;
                        intf.side = newSide;
                    }
                });
            });
        });
        await this.load(graphs);
    }

    updateNodesPosition(updatedGraph) {
        updatedGraph.nodes.forEach((updatedState) => {
            const node = this.graph.nodes.filter(
                (nodeInstance) => updatedState.id === nodeInstance.id,
            )[0];
            node.position = updatedState.position;
        });
    }

    updateCurrentSubgraphName(name) {
        this._graph.name = name;
    }
}
