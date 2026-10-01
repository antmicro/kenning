/*
 * Copyright (c) 2022-2026 Antmicro <www.antmicro.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/* eslint-disable max-classes-per-file */

import Ajv2019 from 'ajv/dist/2019.js';
import jsonlint from 'jsonlint-webpack';

import { useBaklava, useCommandHandler, useViewModel } from '@baklavajs/renderer-vue';
import { toRaw, ref, reactive } from 'vue';
import { Graph } from '@baklavajs/core';
import { v4 as uuidv4 } from 'uuid';
import { useHistory } from './History.ts';
import { useClipboard } from './Clipboard.ts';

import PipelineManagerEditor from '../custom/Editor.js';
import InterfaceTypes from './InterfaceTypes.js';
import PropertyStyler from './PropertyStyler.js';
import NodeStyler from './NodeStyler.js';
import InterfaceStyler from './InterfaceStyler.js';

import { CustomNodeFactory, GraphFactory } from './NodeFactory.js';
import unresolvedSpecificationSchema from '../../../resources/schemas/unresolved_specification_schema.json' with {type: 'json'};
import specificationSchema from '../../../resources/schemas/specification_schema.json' with {type: 'json'};
import metadataSchema from '../../../resources/schemas/metadata_schema.json' with {type: 'json'};
import dataflowSchema from '../../../resources/schemas/dataflow_schema.json' with {type: 'json'};
import graphSchema from '../../../resources/schemas/graph_schema.json' with {type: 'json'};
import messageSchema from '../../../resources/schemas/message_schema.json' with {type: 'json'};
import ConnectionRenderer from './ConnectionRenderer.js';
import Specification from './Specification.js';
import validateJSON from './validate-json.js';

import globalProperties from '../globalProperties.ts';
import { textColorToHex, isTextColor, hexToRGB } from './nodeCreation/nodeColors.js';

/* eslint-disable lines-between-class-members */
/**
 * Readonly helper class that reads and stores default values from metadata schema.
 */
class Metadata {
    constructor() {
        Object.entries(metadataSchema.properties).forEach(([name, state]) => {
            this[name] = state.default;
        });
    }
}

export const DEFAULT_CUSTOM_NODE_CATEGORY = 'Default';
export const DEFAULT_CUSTOM_NODE_NAME = 'New Node Type';
export const DEFAULT_CUSTOM_NODE_TYPE = 'New Node Type';

// If a graph node entry does not have a category assigned, this values is used
// as a fallback category
export const DEFAULT_GRAPH_NODE_CATEGORY = 'Graphs';
export const DEFAULT_GRAPH_NODE_NAME = 'New Graph Node';
export const DEFAULT_GRAPH_NODE_TYPE = 'New Graph Node';

// Styles
export const NEW_NODE_STYLE = '__new';
export const EDITED_NODE_STYLE = '__edited';

/**
 * Translates the provided url according to
 * the optional substitution spec provided at compile time.
 *
 * @param loc the encoded URL location of the resource
 * @returns a translated URL
 */
function parseLocation(loc) {
    if (loc.startsWith('data:application/json')) {
        return loc;
    }

    let relativeurl = '{}';
    let urlparams;
    if (typeof document !== 'undefined') {
        const urlparent = document.location.href.split('/').slice(0, -1).join('/');
        relativeurl = `${urlparent}/{}`;
        urlparams = new URLSearchParams(window.location.search);
    }
    const defaultsubs = { https: 'https://{}', http: 'http://{}', relative: `${relativeurl}` };
    const parseOrEmpty = (jsonsub) => (jsonsub && JSON.parse(jsonsub)) ?? {};
    const subs = {
        ...defaultsubs,
        ...parseOrEmpty(urlparams?.get('jsonsubs')),
        ...parseOrEmpty(process.env.VUE_APP_JSON_URL_SUBSTITUTES),
    };
    const parts = loc.split('//');

    if (parts.length < 2) return undefined;

    const key = parts[0].substring(0, parts[0].length - 1);
    const specifiedUrl = parts.slice(1).join('');

    if (!Object.keys(subs).includes(key)) return undefined;
    return subs[key].replace('{}', specifiedUrl);
}

/**
 * Loads the JSON file from the remote location given in URL.
 *
 * @param customLocation the URL location of the resource
 * @returns a tuple of a boolean and a JSON object or an error message.
 * The boolean is true if the JSON was successfully loaded and parsed
 */
export async function loadJsonFromRemoteLocation(customLocation) {
    const location = parseLocation(customLocation);
    if (location === undefined) {
        return [false, `Could not download the resource from:  ${customLocation}.`];
    }
    let fetchedContent;
    try {
        fetchedContent = await fetch(location, { mode: 'cors' });
    } catch (error) {
        return [false, error.message];
    }
    try {
        const jsonContent = await fetchedContent.json();
        return [true, jsonContent];
    } catch (error) {
        return [false, error.message];
    }
}

export default class EditorManager {
    static instance;

    defaultMetadata = new Metadata();

    editor = new PipelineManagerEditor();

    /**
     * @type {import('baklavajs').IBaklavaViewModel & {
     *    editor: PipelineManagerEditor,
     *    settings: any
     * }}
     */
    baklavaView = useBaklava(this.editor);

    specificationLoaded = ref(false);

    validating = ref(false);

    specification = Specification.getInstance();

    updatedMetadata = {};

    relatedGraphsStore = [];

    /**
     * 'externalApplicationManager' property is set when ExternalApplicationManager
     * is initialized, in its constructor, to avoid the cycle problem if it were
     * imported normally.
     */

    externalApplicationManager;

    constructor() {
        this.editor.editorManager = this;

        // Baklava's view registers subgraph input and output nodes
        // This call un-registers them as obsolete
        this.editor.unregisterNodes();

        this.baklavaView.connectionRenderer = new ConnectionRenderer(
            this.baklavaView,
            this.defaultMetadata.connectionStyle,
            this.defaultMetadata.randomizedOffset,
            this.defaultMetadata.switchableInterfaces,
        );

        this.baklavaView.editor.layoutManager.useAlgorithm(this.defaultMetadata.layout);
        this.baklavaView.interfaceTypes = new InterfaceTypes(this.baklavaView);
        this.baklavaView.propertyStyler = new PropertyStyler(this.baklavaView);
        this.baklavaView.nodeStyler = new NodeStyler(this.baklavaView);
        this.baklavaView.interfaceStyler = new InterfaceStyler(this.baklavaView);

        // need to be set here as settings try to use this value
        // before this value can be loaded from specification
        this.baklavaView.layers = this.defaultMetadata.layers;
        this.baklavaView.collapseSidebar = this.defaultMetadata.collapseSidebar;
        this.baklavaView.movementStep = this.defaultMetadata.movementStep;
        this.baklavaView.editor.allowLoopbacks = this.defaultMetadata.allowLoopbacks;
        this.baklavaView.navbarItems = this.defaultMetadata.navbarItems;
        this.baklavaView.contextMenuActions = this.defaultMetadata.contextMenuActions;
        this.baklavaView.subgraphContextMenuActions =
            this.defaultMetadata.subgraphContextMenuActions;
        this.baklavaView.cache = {};
        this.baklavaView.logLevel = this.defaultMetadata.logLevel;
        this.baklavaView.settings.editableNodeTypes = this.defaultMetadata.editableNodeTypes;
        this.baklavaView.settings.hideAnchors = this.defaultMetadata.hideAnchors;
        this.baklavaView.settings.showIds = this.defaultMetadata.showIds;
        this.baklavaView.settings.newGraphNode = this.defaultMetadata.newGraphNode;
        this.baklavaView.settings.newNodeType = this.defaultMetadata.newNodeType;
        this.baklavaView.settings.showHiddenProperties = this.defaultMetadata.showHiddenProperties;
        this.baklavaView.settings.disableInterfaceExpose = this.defaultMetadata
            .disableInterfaceExpose;
        this.baklavaView.settings.removeEmptySubgraphs = this.defaultMetadata.removeEmptySubgraphs;

        this.specificationVersion = unresolvedSpecificationSchema.version;
        this.baklavaView.commandHandler = useCommandHandler();
        this.baklavaView.history = null;
        this.baklavaView.history = useHistory(
            toRaw(this.baklavaView).displayedGraph,
            this.baklavaView.commandHandler,
        );
        this.baklavaView.clipboard = useClipboard(
            toRaw(this.baklavaView).displayedGraph,
            ref(this.baklavaView.editor),
            this.baklavaView.commandHandler,
        );

        this.modifiedNodeSpecificationRegistry = {};
    }

    findChildren(type, children = new Set()) {
        if (children.has(type)) {
            return;
        }
        children.add(type);

        const extendingNodes = this.specification
            .currentSpecification.nodes
            .filter((n) => n?.extends?.some((e) => e === type));

        extendingNodes.forEach((n) => {
            this.findChildren(n.name, children);
        });
    }

    /**
     * Based on edited type and specification will propagate all changes through graphs
     * in the editor.
     * @param {string} type - type of node that was edited.
     * @param {Object} parsedSpecification - a graph that will be processed.
     */
    updateGraphsInEditor(type, parsedSpecification) {
        const graphs = Array.from(this.editor.graphs);
        graphs.forEach((graph) => {
            graph.nodes.filter((n) => n.type === type)
                .forEach((n) => {
                    if (parsedSpecification?.isCategory) {
                        graph.replaceNode(n, n.type);
                    } else {
                        graph.replaceNode(n, parsedSpecification.name);
                    }
                });
        });

        graphs.forEach((graph) => graph.nodes.filter((n) => n?.subgraph)
            .forEach((n) => n.updateExposedInterfaces(undefined, undefined)));
    }

    /**
     * Function which updates node style in editor and specification.
     * @param {string} styleName - a style class which should be updated.
     * @param {Object} newStyle - a new style.
     */
    updateNodeStyle(styleName, newStyle) {
        this.editor.nodeStyles.set(styleName, newStyle);
        // Update specifications metadata
        this.specification.unresolvedSpecification.metadata.styles[styleName] = newStyle;
        this.specification.currentSpecification.metadata.styles[styleName] = newStyle;
    }

    async preprocessSpecification(dataflowSpecification, {
        unmarkNewNodes,
        urloverrides,
        tryMinify,
    }) {
        const errors = [];
        const warnings = [];

        this.globalVisitedSpecs = new Set();
        const toInclude = Object.fromEntries(Object.entries({
            include: dataflowSpecification.include,
            urloverrides: dataflowSpecification.urloverrides,
        }).filter(([_, value]) => value !== undefined));

        const tag = dataflowSpecification?.metadata?.tag;

        if (dataflowSpecification.nodes && tag) {
            /* eslint-disable-next-line no-param-reassign */
            dataflowSpecification.nodes = dataflowSpecification.nodes.map((node) => ({
                ...node,
                tag,
            }));
        }

        const {
            specification: includedSpecification,
            errors: includeErrors,
            warnings: includeWarnings,
        } = await this.downloadNestedImports(toInclude, undefined, urloverrides);
        errors.push(...includeErrors);
        warnings.push(...includeWarnings);
        if (errors.length) {
            return { errors, warnings };
        }

        if (unmarkNewNodes) {
            warnings.push(...EditorManager.unmarkNewNodes(includedSpecification));
        }

        // Include graphs
        const {
            graphs, errors: includeGraphsErrors,
        } = await EditorManager.includeGraphs([
            ...(includedSpecification.includeGraphs ?? []),
            ...(dataflowSpecification.includeGraphs ?? []),
        ]);
        errors.push(...includeGraphsErrors);
        if (errors.length) {
            return { errors, warnings };
        }

        includedSpecification.graphs = (includedSpecification.graphs ?? []).concat(graphs);
        this.specification.includedSpecification =
            JSON.parse(JSON.stringify(includedSpecification));

        // Merge included specification
        const {
            errors: mergeErrors, warnings: mergeWarnings,
        } = EditorManager.mergeObjects(dataflowSpecification, includedSpecification);
        errors.push(...mergeErrors);
        warnings.push(...mergeWarnings);
        if (errors.length) {
            return { errors, warnings };
        }
        delete dataflowSpecification.include; // eslint-disable-line no-param-reassign
        delete dataflowSpecification.includeGraphs; // eslint-disable-line no-param-reassign

        // Ensure proper color definition
        dataflowSpecification.nodes?.forEach((node) => {
            if (node.color) {
                if (isTextColor(node.color)) {
                    // eslint-disable-next-line no-param-reassign
                    node.color = textColorToHex(node.color);
                } else if (hexToRGB(node.color) === null) {
                    warnings.push(`Invalid color value ${node.color} for node ${node.name}, setting a default color.`);
                    // eslint-disable-next-line no-param-reassign
                    node.color = '#343434';
                }
            }
        });

        // Ensure 'subgraph' field for subgraph node instances
        const nameToSubgraphNode = new Map((dataflowSpecification.nodes ?? [])
            .filter(({ subgraphId }) => subgraphId)
            .map((nodeType) => [EditorManager.getNodeName(nodeType), nodeType]));

        const subgraphFieldNotSet = (node) => !node.subgraph;
        const isSubgraphNode = (node) => nameToSubgraphNode.has(node.name);

        (dataflowSpecification.graphs ?? [])
            .flatMap((graph) => graph.nodes)
            .filter(subgraphFieldNotSet)
            .filter(isSubgraphNode)
            // eslint-disable-next-line no-param-reassign
            .forEach((node) => { node.subgraph = nameToSubgraphNode.get(node.name).subgraphId; });

        const {
            idToGraph,
            idToNested,
        } = EditorManager.getGraphDependencies(dataflowSpecification.graphs ?? []);

        let usedGraphs;
        if (tryMinify === true) {
            // Specification graphs
            if (!dataflowSpecification.graphs) {
                warnings.push(`'tryMinify' is enabled, but no graph is found. Skipping`);
            } else {
                const { entryGraph } = dataflowSpecification;

                // Missing 'entryGraph' is reported in 'updateGraphSpecification'
                const existsEntryGraph = dataflowSpecification
                    .graphs.some((g) => g.id === dataflowSpecification.entryGraph);

                if (entryGraph && existsEntryGraph) {
                    // Entry graph and its nested graphs
                    const nested = idToNested.get(dataflowSpecification.entryGraph);
                    usedGraphs = [entryGraph, ...nested].map((id) => idToGraph.get(id));
                } else {
                    // All graphs
                    usedGraphs = dataflowSpecification.graphs;
                }
            }
        } else if (tryMinify) {
            // Dataflow graphs
            usedGraphs = tryMinify.graphs;
        }

        if (usedGraphs) {
            // Minify nodes
            const usedNames = EditorManager.getUsedNames(usedGraphs);
            // eslint-disable-next-line no-param-reassign
            dataflowSpecification.nodes =
                EditorManager.minifySpecificationNodes(dataflowSpecification.nodes, usedNames);

            const usedGraphsIds = new Set(usedGraphs.map((g) => g.id));
            dataflowSpecification.nodes
                .filter(({ subgraphId }) => subgraphId)
                // Unmatched 'subgraphId' are reported in 'updateGraphSpecification'
                .filter(({ subgraphId }) => idToGraph.get(subgraphId))
                .flatMap(({ subgraphId }) => [subgraphId, ...idToNested.get(subgraphId)])
                .filter((subgraphId) => !usedGraphsIds.has(subgraphId))
                .forEach((subgraphId) => {
                    usedGraphsIds.add(subgraphId);
                    usedGraphs.push(idToGraph.get(subgraphId));
                });
            // eslint-disable-next-line no-param-reassign
            dataflowSpecification.graphs = usedGraphs;
        }

        return {
            errors,
            warnings,
            specification: dataflowSpecification,
            idToNested,
        };
    }

    /* eslint-disable max-len */
    /**
     * Loads the dataflow specification passed in `dataflowSpecification`.
     * The specification describes what nodes are available in the editor.
     *
     * If the current editor already has a specification loaded then the editor
     * and its plugins are reinitialized and then the specification is loaded.
     *
     * @param dataflowSpecification Specification to load, can be either an object or a string
     * @param lazyLoad Decides whether to actually load the specification or just store
     * @param unmarkNewNodes Decides whether to remove styling of new nodes.
     * @param urloverrides Override metadata URLs in provided specification.
     * it and check its versioning. Can be used when loading parts of specification manually.
     * @param {object|boolean} tryMinify If set, requests minification based resolved specification or a given dataflow.
     * @returns An object consisting of errors and warnings arrays. If any array is empty
     * the updating process was successful.
     */
    /* eslint-disable no-underscore-dangle,no-param-reassign */
    /* eslint-enable max-len */
    async updateEditorSpecification(
        dataflowSpecification,
        lazyLoad = false,
        unmarkNewNodes = true,
        urloverrides = null,
        tryMinify = false,
    ) {
        if (!dataflowSpecification) return { errors: ['No specification passed'] };

        if (typeof dataflowSpecification === 'string' || dataflowSpecification instanceof String) {
            try {
                dataflowSpecification = jsonlint.parse(dataflowSpecification);
            } catch (error) {
                return { errors: [error], warnings: [] };
            }
        }

        let state;
        let stateNodeId;
        if (this.isSpecificationLoaded()) {
            state = this.saveDataflow();
            stateNodeId = this.baklavaView.displayedGraph.sidebar.nodeId;
            this.clearEditorManagerState();
        }

        const warnings = [];
        const errors = [];
        const info = [];

        const { version } = dataflowSpecification; // eslint-disable-line object-curly-newline,max-len
        if (!this.specification.currentSpecification) {
            if (version === undefined) {
                warnings.push(
                    `Loaded specification has no version assigned. Please update the specification to version ${this.specificationVersion}.`,
                );
            } else if (version !== this.specificationVersion) {
                info.push(
                    `The specification format version (${version}) differs from the current specification format version (${this.specificationVersion}). It may result in unexpected behaviour.`,
                );
            }
        }

        if (unmarkNewNodes) {
            warnings.push(...EditorManager.unmarkNewNodes(dataflowSpecification));
        }

        this.specification.unresolvedSpecification = reactive(
            JSON.parse(JSON.stringify(dataflowSpecification)));
        this.specification.currentSpecification = dataflowSpecification;
        if (!lazyLoad) {
            const {
                errors: preprocessErrors,
                warnings: preprocessWarnings,
                specification: preprocessedSpecification,
                idToNested,
            } = await this.preprocessSpecification(dataflowSpecification, {
                unmarkNewNodes, urloverrides, tryMinify,
            });
            warnings.push(...preprocessWarnings);
            errors.push(...preprocessErrors);
            if (errors.length) return { errors, warnings, info };
            dataflowSpecification = preprocessedSpecification;

            // Update metadata
            const { metadata } = dataflowSpecification;
            errors.push(...this.updateMetadata(metadata, false, true));
            if (errors.length) {
                return { errors, warnings, info };
            }

            // Update graph specification
            const {
                errors: newErrors, warnings: newWarnings,
            } = await this.updateGraphSpecification(dataflowSpecification, idToNested);
            errors.push(...newErrors);
            warnings.push(...newWarnings);
        }

        if (errors.length === 0) {
            this.setSpecificationLoaded(true);
        } else {
            this.clearEditorManagerState();
        }

        if (state !== undefined && dataflowSpecification.entryGraph === undefined) {
            const ret = await this.loadDataflow(state);
            if (!ret.errors.length && stateNodeId) {
                this.baklavaView.displayedGraph.sidebar.nodeId = stateNodeId;
                this.baklavaView.displayedGraph.sidebar.visible = true;
            } else {
                this.relatedGraphsStore.forEach((g) => this.baklavaView.editor.registerGraph(g));
            }
        }

        if (this.externalApplicationManager) {
            const spec = this.specification.currentSpecification;
            await this.externalApplicationManager.notifyAboutChange('specification_on_change', {
                specification: spec,
            });
        }

        return { errors, warnings, info };
    }

    clearEditorManagerState() {
        this.clearHistory();
        this.baklavaView.editor.graphs.forEach((graph) => graph.setSpecGraph(false));
        this.baklavaView.editor.unregisterGraphs();
        this.baklavaView.editor.deepCleanEditor();
        this.baklavaView.editor.unregisterNodes();
        this.baklavaView.editor.nodeStyles.clear();
        this.baklavaView.editor.nodeLists.clear();
        this.setSpecificationLoaded(false);
        this.specification.currentSpecification = {};
        this.specification.includedSpecification = {};
        this.specification.unresolvedSpecification = reactive({});
    }

    isSpecificationLoaded() {
        if (typeof this.specificationLoaded === 'boolean') return this.specificationLoaded;
        return this.specificationLoaded.value;
    }

    setSpecificationLoaded(value) {
        if (typeof this.specificationLoaded === 'boolean') {
            this.specificationLoaded = value;
        } else {
            this.specificationLoaded.value = value;
        }
    }

    setValidating(value) {
        if (typeof this.validating === 'boolean') {
            this.validating = value;
        } else {
            this.validating.value = value;
        }
    }

    /**
     * Downloads nested imports from the specification and returns an object
     * consisting of nodes, graphs, and errors arrays.
     *
     * @param specification Specification to load.
     * @param trace Set of visited specifications to detect circular imports.
     * @returns Merged specification and errors.
     */
    async downloadNestedImports(specification, trace = new Set(), urloverrides = null) {
        const warnings = [];
        const errors = [];

        // Download specifications and verify for circular imports
        const specificationAndTrace = [];
        const currentImports = new Set();
        const include = specification.include ?? [];
        await Promise.all(include.map(async (specificationUrl) => {
            let style;
            if (typeof specificationUrl === 'object' && specificationUrl !== null) {
                ({ url: specificationUrl, style } = specificationUrl);
            }
            if (currentImports.has(specificationUrl)) {
                errors.push(`Specification is included multiple times, skipping ${specificationUrl}`);
                return;
            }
            if (trace.has(specificationUrl)) {
                errors.push(`Circular dependency detected in included specification ${specificationUrl}`);
                return;
            }
            currentImports.add(specificationUrl);

            if (!this.globalVisitedSpecs.has(specificationUrl)) {
                this.globalVisitedSpecs.add(specificationUrl);
                const [status, val] = await loadJsonFromRemoteLocation(specificationUrl);
                if (status === false) {
                    errors.push(`Could not load the included specification from ${specificationUrl}. Reason: ${val}`);
                } else {
                    specificationAndTrace.push(
                        {
                            specification: val,
                            trace: new Set([...trace, specificationUrl]), // Detect circular imports
                            style,
                        },
                    );
                }
            }
        }));

        if (errors.length) {
            return { specification, errors, warnings };
        }

        // Download nested imports
        await Promise.all(specificationAndTrace.map(
            async ({ specification: spec, trace: specTrace, style }) => {
                const {
                    specification: newSpecification, errors: newErrors, warnings: newWarnings,
                } = await this.downloadNestedImports(spec, specTrace, urloverrides);
                errors.push(...newErrors);
                warnings.push(...newWarnings);

                if (style !== undefined) EditorManager.includeWithStyle(newSpecification, style);

                EditorManager.applyUrlOverrides(
                    newSpecification,
                    { ...(specification.urloverrides ?? {}), ...(urloverrides ?? {}) },
                );

                const {
                    errors: mergeErrors, warnings: mergeWarnings,
                } = EditorManager.mergeObjects(specification, newSpecification);
                errors.push(...mergeErrors);
                warnings.push(...mergeWarnings);
            }));
        return { specification, errors, warnings };
    }

    /**
     * Downloads included dataflows from the specification and converts them to the
     * graphs format to be included into the specification.
     *
     * @param includeGraphs Array of included graphs
     * @returns Array graphs and an array of errors that occurred during the process.
     */
    static async includeGraphs(includeGraphs) {
        const errors = [];
        const graphs = [];

        if (includeGraphs.length === 0) {
            return { graphs, errors };
        } if (includeGraphs.length !== new Set(includeGraphs).size) {
            errors.push('Duplicate subgraph includes detected. Aborting.');
            return { graphs, errors };
        }

        const dataflows = [];
        await Promise.all(includeGraphs.map(async (dataflow) => {
            const [status, val] = await loadJsonFromRemoteLocation(dataflow.url);
            if (status === false) {
                errors.push(`Could not load the included dataflow from '${dataflow.url}'. Reason: ${val}`);
                return;
            }

            dataflows.push(val);
        }));

        if (errors.length) return { graphs, errors };

        for (let i = 0; i < includeGraphs.length; i += 1) {
            const dataflow = dataflows[i];
            const dataflowMetadata = includeGraphs[i];

            if (dataflow.graphs.length !== 1) {
                errors.push(`Only single graph dataflows are supported. Aborting loading subgraph include from ${dataflowMetadata.url}.`);
                continue; // eslint-disable-line no-continue
            }

            const targetGraph = dataflow.graphs[0];
            targetGraph.name = dataflowMetadata.name ?? targetGraph.name;

            graphs.push(targetGraph);
        }

        return { graphs, errors };
    }

    /**
     * Registers default nodes, that are always present in the editor.
     * The default nodes are the graph node and the new custom node.
     * If the nodes are already present in the editor, an error is returned.
     *
     * @returns {object} Object consisting of errors and warnings arrays.
     */
    registerDefaultNodes() {
        const errors = [];
        const warnings = [];
        // Adding a default graph node to the editor so that custom graphs can be created
        if (this.editor.nodeTypes.has(DEFAULT_GRAPH_NODE_TYPE)) {
            errors.push(
                `Node name '${DEFAULT_GRAPH_NODE_NAME}' is reserved by the editor, ` +
                'but it was included in the specification. ' +
                'Please change the name of the graph node to avoid conflicts.',
            );
            return { errors, warnings };
        }
        if (this.editor.nodeTypes.has(DEFAULT_CUSTOM_NODE_TYPE)) {
            errors.push(
                `Node name '${DEFAULT_CUSTOM_NODE_NAME}' is reserved by the editor, ` +
                'but it was included in the specification. ' +
                'Please change the name of the graph node to avoid conflicts.',
            );
            return { errors, warnings };
        }
        const customNodeType = {
            name: DEFAULT_CUSTOM_NODE_NAME,
            category: DEFAULT_CUSTOM_NODE_CATEGORY,
        };
        this._registerNodeType(customNodeType);

        const defaultGraphNode = {
            name: DEFAULT_GRAPH_NODE_NAME,
            category: DEFAULT_GRAPH_NODE_CATEGORY,
        };

        const myGraph = GraphFactory(
            [],
            [],
            DEFAULT_GRAPH_NODE_NAME,
            this.baklavaView.editor,
        );

        defaultGraphNode.subgraphId = myGraph.id;

        // If `myGraph` is any array then it is an array of errors
        if (Array.isArray(myGraph) && myGraph.length) {
            errors.push(...myGraph);
        } else {
            this.baklavaView.editor.addGraphTemplate(
                myGraph,
                defaultGraphNode,
            );
        }
        return { errors, warnings };
    }

    replaceNodeSpecificationInEditor(nodeType, nodeSpecification) {
        const errors = [];

        errors.push(...this._unregisterNodeType(nodeType));
        if (errors.length) {
            return errors;
        }

        const ret = this.addNodeToEditorSpecification(
            nodeSpecification,
            nodeType,
        );

        if (ret.errors !== undefined && ret.errors.length) {
            errors.push(...ret.errors);
            return errors;
        }

        this.updateGraphsInEditor(nodeType, nodeSpecification);

        const children = new Set();

        this.findChildren(nodeType, children);
        // there is no need to update a node second time
        children.delete(nodeType);

        // find nodes extending current node in specification
        const extendingNodes = this.specification
            .currentSpecification.nodes
            .filter((n) => children.has(n.name));
        // refresh nodes extending from current node
        extendingNodes.forEach((n) => {
            this.updateGraphsInEditor(n.name, n);
        });

        return errors;
    }

    /**
     * Propagates changes to nodes extending a category node.
     * Called when the category node type is edited.
     *
     * @param {object} nodeSpecification Node specification to add
     * @param {object} nodeToUpdate Node type to update
     */
    updateExtendingNodes(nodeSpecification, nodeToUpdate) {
        const unresolvedChildNodes = this.specification.unresolvedSpecification.nodes
            .filter((node) => node.extends?.includes(nodeToUpdate)) ?? [];
        const resolvedChildNodes = this.specification.currentSpecification.nodes
            .filter((node) => node.extends?.includes(nodeToUpdate)) ?? [];

        const childNodes = [...unresolvedChildNodes, ...resolvedChildNodes];
        childNodes.forEach((node) => {
            node.extends.forEach((parent, i) => {
                if (parent === nodeToUpdate) {
                    node.extends[i] = EditorManager.getNodeName(nodeSpecification);
                }
            });
        });

        const resolvedMap = new Map(
            resolvedChildNodes.map((node) => [node.name, node]),
        );
        resolvedMap.forEach((resolvedNode, nodeName) => {
            this._unregisterNodeType(nodeName);

            const { viewModel } = useViewModel();
            const { editor } = viewModel.value;
            const allNodes = Array.from(editor.graphs).map((graph) => graph.nodes).flat();
            const nodes = allNodes.filter((n) => n.type === nodeName);
            nodes.forEach((n) => {
                Object.entries(structuredClone(toRaw(resolvedNode))).forEach(([key, value]) => {
                    if (value !== undefined && key !== 'interfaces' && key !== 'properties') {
                        n[key] = structuredClone(toRaw(value));
                    }
                });
            });

            this._registerNodeType(structuredClone(toRaw(resolvedNode)));
            this.updateExtendingNodes(structuredClone(toRaw(resolvedNode)), nodeName);
        });
    }

    /**
     * Updates the list of extending nodes in all parent nodes.
     * Called when the name of the node type is changed
     * or the node type has been deleted.
     *
     * @param {object} nodeSpecification Updated node data
     * @param {object} nodeToUpdate Node type to update
     * @param {object} remove Whether the node has been deleted
     */
    updateParentNode(nodeSpecification, nodeToUpdate, remove = false) {
        nodeSpecification?.extends?.forEach((parentType) => {
            const parentSpec = this.baklavaView.editor.parentNodes.get(parentType);
            if (parentSpec.extending?.includes(nodeToUpdate)) {
                if (remove) {
                    parentSpec.extending.splice(
                        parentSpec.extending.indexOf(nodeToUpdate),
                        1,
                    );
                } else {
                    parentSpec.extending.splice(
                        parentSpec.extending.indexOf(nodeToUpdate),
                        1,
                        nodeSpecification.name,
                    );
                }
            }
        });
        if (this.baklavaView.editor.parentNodes.has(nodeToUpdate)) {
            this.baklavaView.editor.parentNodes.delete(nodeToUpdate);
        }
        if (!remove) {
            this.baklavaView.editor.parentNodes.set(nodeSpecification.name, nodeSpecification);
        }
    }

    /**
     * Attaches subgraph to the updated subgraph node.
     *
     * @param {object} node Updated graph node
     */
    refreshSubgraph(node) {
        let graphNode;
        let subgraph;
        this.specification.currentSpecification.graphs?.forEach((graph) => {
            if (node.subgraphId === graph.id) {
                graphNode = node;
                subgraph = graph;
            }
        });

        if (subgraph) {
            const myGraph = GraphFactory(
                subgraph.nodes,
                subgraph.connections,
                subgraph.name,
                this.baklavaView.editor,
            );

            this.baklavaView.editor.addGraphTemplate(
                myGraph,
                graphNode,
            );
        }
    }

    resolveAndRegisterNode(nodeSpecification) {
        const resolvedNodeSpec = structuredClone(nodeSpecification);

        if (resolvedNodeSpec.isCategory) {
            this.baklavaView.editor
                .parentNodes.set(resolvedNodeSpec.name, resolvedNodeSpec);
        }

        const inherited = this.findSimpleInheritedAttributes(resolvedNodeSpec.name,
            this.specification.currentSpecification.nodes);

        if (inherited.category && !resolvedNodeSpec.category) {
            resolvedNodeSpec.category = inherited.category;
            resolvedNodeSpec.simpleInherited ??= [];
            resolvedNodeSpec.simpleInherited.push('category');
        }
        this.specification.currentSpecification.nodes.push(resolvedNodeSpec);
    }

    /**
     * Validates the node specification passed in `nodeSpecification` and if
     * it is correct adds it to the unresolved specification.
     * If there is no current specification loaded then a new one is created.
     * If node to update is provided, the specification is assigned to that node.
     * The node specification should be in the format of the node schema.
     *
     * @param {object} nodeSpecification Node specification to add
     * @param {string|undefined} nodeToUpdate Node type to update
     * @param {boolean} removeUnused Whether keys missing from nodeSpecification should be removed
     * @param {boolean} override A flag that makes function override existing node specification.
     * @returns An object consisting of errors and warnings arrays. If both arrays are empty
     * the updating process was successful.
     */
    addNodeToEditorSpecification(nodeSpecification,
        nodeToUpdate = undefined,
        removeUnused = true,
        override = false) {
        if (override && nodeToUpdate !== undefined) {
            this._unregisterNodeType(nodeToUpdate);
        }
        // Remove undefined fields
        Object.entries(nodeSpecification.properties ?? {}).forEach(([_, value]) => {
            if (value.inherited) {
                const idx = nodeSpecification.properties.indexOf(value);
                nodeSpecification.properties.splice(idx, 1);
                return;
            }
            Object.keys(value).forEach((key) => {
                if (value[key] === undefined || key === 'inherited') {
                    delete value[key];
                }
            });
        });
        Object.entries(nodeSpecification.interfaces ?? {}).forEach(([_, value]) => {
            if (value.inherited) {
                const idx = nodeSpecification.interfaces.indexOf(value);
                nodeSpecification.interfaces.splice(idx, 1);
                return;
            }
            Object.keys(value).forEach((key) => {
                if (value[key] === undefined || key === 'inherited') {
                    delete value[key];
                }
            });
        });

        let validationErrors = this.validateNode(nodeSpecification);
        if (validationErrors.length) return { errors: validationErrors };

        if (this.specification.currentSpecification === undefined) {
            return { errors: ['Current specification cannot be empty'] };
        }
        const invalidParent = Object.values(nodeSpecification.extends ?? []).some((parent) => {
            const nodeSpec = this.specification.currentSpecification.nodes.find(
                (spec) => spec.name === parent,
            );
            return !!(nodeSpec && nodeSpec.subgraphId);
        });

        if (invalidParent) {
            return { errors: ['Extending subgraphs dynamically is not currently supported.'] };
        }

        this.specification.unresolvedSpecification.nodes ??= [];

        // Modify existing node specification
        if (nodeToUpdate !== undefined) {
            const unresolvedNodeSpecification = this.specification.unresolvedSpecification
                .nodes.find(
                    (node) => EditorManager.getNodeName(node) === nodeToUpdate,
                );
            const resolvedNodeSpecification = this.specification.currentSpecification.nodes.find(
                (node) => EditorManager.getNodeName(node) === nodeToUpdate,
            );

            if (resolvedNodeSpecification === undefined) {
                // The node is newly created - it is not in registered specification yet
                Object.entries(nodeSpecification).forEach(([key, value]) => {
                    if (value !== undefined) {
                        unresolvedNodeSpecification[key] = structuredClone(toRaw(value));
                    }
                });
                validationErrors = this._registerNodeType(unresolvedNodeSpecification);
                if (validationErrors.length) {
                    return { errors: validationErrors, warnings: [] };
                }
                this.resolveAndRegisterNode(nodeSpecification);
            } else {
                if (unresolvedNodeSpecification === undefined) {
                    // The node is included - push new spec to unresolvedSpecification to override
                    nodeSpecification.includeName = nodeToUpdate;
                    this.specification.unresolvedSpecification.nodes.push(nodeSpecification);
                } else {
                    if (removeUnused) {
                        Object.keys(unresolvedNodeSpecification)
                            .filter((key) => !(key in nodeSpecification))
                            .forEach((key) => { delete unresolvedNodeSpecification[key]; });
                    }
                    Object.entries(nodeSpecification).forEach(([key, value]) => {
                        if (value !== undefined) {
                            unresolvedNodeSpecification[key] = JSON.parse(JSON.stringify(value));
                        }
                    });
                }
                if (removeUnused) {
                    Object.keys(resolvedNodeSpecification)
                        .filter((key) => !(key in nodeSpecification))
                        .forEach((key) => { delete resolvedNodeSpecification[key]; });
                }
                Object.entries(nodeSpecification).forEach(([key, value]) => {
                    if (value !== undefined) {
                        resolvedNodeSpecification[key] = JSON.parse(JSON.stringify(value));
                    }
                });

                const nodeName = EditorManager.getNodeName(nodeSpecification);
                resolvedNodeSpecification.name = nodeName;

                const extendingNodes = this.specification.currentSpecification.nodes
                    .filter((node) => node.extends?.includes(nodeToUpdate))
                    .map((node) => EditorManager.getNodeName(node));
                if (extendingNodes !== undefined) {
                    resolvedNodeSpecification.extending = extendingNodes;
                }

                validationErrors = this._registerNodeType(resolvedNodeSpecification);
                if (validationErrors.length) {
                    return { errors: validationErrors, warnings: [] };
                }

                if (extendingNodes !== undefined) {
                    this.updateExtendingNodes(
                        JSON.parse(JSON.stringify(nodeSpecification)),
                        nodeToUpdate,
                    );
                }

                if (nodeSpecification.name !== nodeToUpdate) {
                    this.updateParentNode(resolvedNodeSpecification, nodeToUpdate);

                    [
                        ...this.specification.unresolvedSpecification.graphs ?? [],
                        ...this.specification.currentSpecification.graphs ?? [],
                    ].forEach((graph) => {
                        const nodesToUpdate = graph.nodes.filter((n) => n.name === nodeToUpdate);
                        nodesToUpdate.forEach((node) => {
                            node.name = nodeSpecification.name;
                        });
                    });

                    [
                        ...this.specification.unresolvedSpecification.nodes ?? [],
                        ...this.specification.currentSpecification.nodes ?? [],
                    ].forEach((node) => {
                        this.refreshSubgraph(node);
                    });
                }

                if (resolvedNodeSpecification.subgraphId !== undefined) {
                    this.refreshSubgraph(resolvedNodeSpecification);
                }
            }
        } else {
            validationErrors = this._registerNodeType(nodeSpecification);
            if (validationErrors.length) {
                return { errors: validationErrors, warnings: [] };
            }
            this.specification.unresolvedSpecification.nodes.push(nodeSpecification);
            this.resolveAndRegisterNode(nodeSpecification);
        }

        if (this.externalApplicationManager) {
            const spec = this.specification.unresolvedSpecification;
            this.externalApplicationManager.notifyAboutChange('specification_on_change', {
                specification: spec,
            }).then();
        }

        // Keep track of additional node types
        this.editor.additionalNodeTypes.add(nodeSpecification.name);

        return { errors: [], warnings: [], spec: nodeSpecification };
    }
    /**
     * Get complete specification of a node type.
     *
     * @param {string} type - The type of node for which to fetch specification.
     * @returns {Object} Complete specification of provided type.
     */
    getCurrentNodeSpecification(type) {
        const nodeMatchesSpec = (specNode) => {
            const isCategory = specNode.isCategory ?? false;
            const nodeType = type;
            const specNodeType = isCategory
                ? specNode.category?.split('/').slice(-1)[0]
                : specNode.name;
            return nodeType === specNodeType;
        };
        let spec = this.specification.currentSpecification
            .nodes?.find((n) => EditorManager.getNodeName(n) === type);

        if (spec === undefined) {
            spec = this.specification.currentSpecification
                .nodes?.find(nodeMatchesSpec);
        }
        if (spec === undefined) {
            spec = this.specification.unresolvedSpecification
                .nodes?.find(nodeMatchesSpec);
        }
        if (spec === undefined) {
            spec = this.specification.unresolvedSpecification
                .nodes?.find((n) => EditorManager.getNodeName(n) === type);
        }
        return spec;
    }

    /**
     * Removes node type from editor specification.
     *
     * @param {string} nodeType Node type to remove
     */
    removeNodeType(nodeType) {
        this._unregisterNodeType(nodeType);

        // Remove from editor graphs
        this.baklavaView.editor.graphs.forEach((graph) => {
            graph.nodes
                .filter((node) => node.type === nodeType)
                .forEach((node) => graph.removeNode(node));
        });

        // Remove from specification nodes
        const resolvedNodeSpecification = this.specification.currentSpecification.nodes.find(
            (node) => EditorManager.getNodeName(node) === nodeType,
        );
        this.specification.unresolvedSpecification.nodes = this.specification
            .unresolvedSpecification.nodes.filter(
                (node) => EditorManager.getNodeName(node) !== nodeType,
            );
        this.specification.currentSpecification.nodes = this.specification.currentSpecification
            .nodes.filter((node) => EditorManager.getNodeName(node) !== nodeType);

        // Remove from specification graphs
        this.specification.unresolvedSpecification.graphs?.forEach((graph) => {
            graph.nodes = graph.nodes.filter((node) => node.name !== nodeType);
        });
        this.specification.currentSpecification.graphs?.forEach((graph) => {
            graph.nodes = graph.nodes.filter((node) => node.name !== nodeType);
        });

        // Update child nodes
        this.specification.unresolvedSpecification.nodes
            .filter((node) => node.extends !== undefined)
            .forEach((node) => {
                node.extends = node.extends.filter((n) => n !== nodeType);
            });
        this.specification.currentSpecification.nodes
            .filter((node) => node.extends !== undefined)
            .forEach((node) => {
                node.extends = node.extends.filter((n) => n !== nodeType);
            });

        // Update parent nodes
        if (resolvedNodeSpecification !== undefined) {
            this.updateParentNode(resolvedNodeSpecification, nodeType, true);
        }

        if (this.externalApplicationManager) {
            const spec = this.specification.unresolvedSpecification;
            this.externalApplicationManager.notifyAboutChange('specification_on_change', {
                specification: spec,
            }).then();
        }

        // Keep track of additional node types
        this.editor.additionalNodeTypes.delete(nodeType);
    }

    /**
     * Assigns a new subgraph to the node.
     *
     * @param {object} node Node that the subgraph will be attached to
     * @param {Array} nodes Nodes in the subgraph
     * @param {Array} connections Connections in the subgraph
     * @returns An array of errors that occurred.
     */
    addSubgraphToNode(node, nodes = [], connections = []) {
        const isSubgraphNode = node?.template !== undefined;
        if (!isSubgraphNode) {
            this._unregisterNodeType(node.type);
        }
        // Create new graph
        const newGraph = GraphFactory(nodes, connections, node.type, this.baklavaView.editor);
        if (Array.isArray(newGraph) && newGraph.length) {
            return newGraph;
        }

        if (isSubgraphNode) {
            const graph = new Graph(this.editor, newGraph);
            graph.name = node.title;
            this.editor.registerGraph(graph);
            graph.graphNode = node;
            graph.editor = this.editor;
            node.subgraph = graph;

            return node;
        }
        // Update node specification
        const unresolvedNodeSpecification = this.specification
            .unresolvedSpecification.nodes?.find(
                (n) => EditorManager.getNodeName(n) === node.type,
            );
        const resolvedNodeSpecification = this.specification.currentSpecification.nodes?.find(
            (n) => EditorManager.getNodeName(n) === node.type,
        );
        if (resolvedNodeSpecification === undefined) {
            unresolvedNodeSpecification.subgraphId = newGraph.id;
            this.baklavaView.editor.addGraphTemplate(newGraph, unresolvedNodeSpecification);
        } else {
            resolvedNodeSpecification.subgraphId = newGraph.id;
            if (unresolvedNodeSpecification === undefined) {
                if (this.specification.unresolvedSpecification.nodes === undefined) {
                    this.specification.unresolvedSpecification.nodes = [];
                }
                this.specification.unresolvedSpecification.nodes
                    .push(resolvedNodeSpecification);
            } else {
                unresolvedNodeSpecification.subgraphId = newGraph.id;
            }
            this.baklavaView.editor.addGraphTemplate(newGraph, resolvedNodeSpecification);
        }

        // Update editor nodes
        const { viewModel } = useViewModel();
        const { displayedGraph } = viewModel.value;
        const nodesToReplace = displayedGraph.nodes.filter(
            (n) => n.type === node.type,
        );
        let result;
        nodesToReplace.forEach((n) => {
            const newNode = displayedGraph.replaceNode(n, node.type, true);
            if (n === node) {
                result = newNode;
            }
        });
        return result;
    }

    /**
     * Creates a new nodeType and registers it in the editor.
     * If the nodeType is not valid then an array of errors is returned.
     * If the nodeType is valid then it is registered in the editor and an empty array is returned.
     *
     * @param {object} node Node to register
     * @returns An array of errors that occurred during the node registration.
     */
    _registerNodeType(node) {
        if (this.baklavaView.editor.nodeTypes.has(node.name)) {
            return [`Node of type ${node.name} is already registered`];
        }

        // eliminate duplicates from two 'branching' inheritance paths,
        // probably should be checked for during validation.
        const intfsInherited = this.findInheritedInterfaces(node.name).filter((intf) =>
            !node.interfaces?.find((i) => i.name === intf.name && i.side === intf.side))
            ?.map((intf) => ({ ...intf, inherited: true }))
            ?.filter((intf) => !node.interfaces?.find((i) => i.name === intf.name && i.override));
        const propsInherited = this.findInheritedProperties(node.name).filter((prop) =>
            !node.properties?.find((p) => p.name === prop.name))
            ?.map((prop) => ({ ...prop, inherited: true }))
            ?.filter((prop) => !node.properties?.find((p) => p.name === prop.name && p.override));
        const inheritedAttributes = this.findSimpleInheritedAttributes(node.name);

        const myNode = CustomNodeFactory(
            node.name,
            node.layer ?? inheritedAttributes.layer,
            [...node.interfaces ?? [], ...intfsInherited],
            [...node.properties ?? [], ...propsInherited],
            node.interfaceGroups ?? [],
            node.defaultInterfaceGroups ?? [],
            node.twoColumn ?? this.baklavaView.twoColumn ?? false,
            node.description ?? '',
            node.extends ?? [],
            node.extending ?? [],
            node.siblings ?? [],
            node.width ?? 0,
            Object.keys(inheritedAttributes),
            node?.height,
        );

        // If my node is any array then it is an array of errors
        if (Array.isArray(myNode) && myNode.length) {
            return myNode;
        }

        this.baklavaView.editor.registerNodeType(myNode, {
            title: node.name,
            category: node.category ? node.category : inheritedAttributes.category,
            isCategory: node.isCategory ?? false,
            color: node.color,
            style: node.style,
            pill: node.pill,
            subgraphId: node.subgraphId,
            relatedGraphs: node.relatedGraphs,
        });
        if ('icon' in node) {
            const icon = typeof node.icon === 'string' ? node.icon : this.getMetadataIcon(node.icon);
            this.baklavaView.editor.nodeIcons.set(node.name, icon);
        }
        if ('icon' in inheritedAttributes) {
            const icon = typeof inheritedAttributes.icon === 'string' ? inheritedAttributes.icon : this.getMetadataIcon(inheritedAttributes.icon);
            this.baklavaView.editor.nodeIcons.set(node.name, icon);
        }
        if ('urls' in node) {
            Object.entries(node.urls).forEach(([urlName, url]) => {
                if (!this.baklavaView.editor.nodeURLs.has(node.name)) {
                    this.baklavaView.editor.nodeURLs.set(node.name, {});
                }
                this.baklavaView.editor.nodeURLs.get(node.name)[urlName] = url;
            });
        }
        if ('contextMenuActions' in node) {
            this.baklavaView.editor.nodeContextMenuActions.set(node.name, node.contextMenuActions);
        }
        return [];
    }

    /**
     * Takes the first key-value pair of icon mapping and looks up the base value in the metadata.
     * @rapam {object} icon icon mapping.
     * @returns Full URL with substituted base.
     */
    getMetadataIcon(icon) {
        const [[baseName, suffix]] = Object.entries(icon);
        const baseUrl = this.baklavaView.editor.baseIconUrls.get(baseName);
        return `${baseUrl}${baseUrl.endsWith('/') ? '' : '/'}${suffix}`;
    }

    /**
     * Unregisters a node type
     * If there is no such node type then an array of errors is returned.
     *
     * @param {string} nodeType Node to unregister
     * @returns An array of errors that occurred during the node unregistration.
     */
    _unregisterNodeType(nodeType) {
        if (this.baklavaView.editor.nodeTypes.has(nodeType)) {
            this.baklavaView.editor.unregisterNodeType(nodeType);
            return [];
        }
        return [`Node of type ${nodeType} is not registered`];
    }

    /**
     * Creates mapping between graph ID and its descendants.
     *
     * @param {object[]} graphs - Graphs to parse.
     * @returns - Generated mapping.
     */
    static getGraphDependencies(graphs) {
        const graphIds = graphs.map(({ id }) => id).filter(Boolean);

        // Collect all dependencies for each graph
        const idToNested = new Map();
        const idToGraph = new Map(graphs.map((graph) => [graph.id, graph]));
        const dfs = (graphId, visited = new Set()) => {
            const graph = idToGraph.get(graphId);
            if (!graphId || !graph || visited.has(graphId)) return [];
            visited.add(graphId);

            // Collect immediate dependencies
            const subgraphs = graph.nodes
                ?.map((node) => node.subgraph) ?? [];
            const relatedGraphs = graph.nodes
                ?.map((node) => node.relatedGraphs?.map(({ id }) => id) ?? [])
                .flat() ?? [];
            const immediateChildren = [...subgraphs, ...relatedGraphs]
                .filter((id) => id !== undefined);

            // Collect and store dependencies
            const nested = immediateChildren.flatMap((g) => dfs(g, visited));
            idToNested.set(graphId, Array.from(new Set(nested)));

            return [graphId, ...nested];
        };
        graphIds.forEach((g) => dfs(g));
        return { idToGraph, idToNested };
    }

    /* eslint-disable max-len */
    /**
     * Sorts graphs topologically.
     *
     * @param {object[]} graphs Graphs to sort.
     * @param {Map<string, string[]>} idToNested Mapping between graph ID to its nested graphs.
     * @returns {string[]} Sorted graphs IDs.
     */
    /* eslint-enable max-len */
    static sortGraphs(graphs, idToNested) {
        const graphIds = graphs.map(({ id }) => id).filter(Boolean);

        const visited = new Set();
        const idToGraph = new Map(graphs.map((graph) => [graph.id, graph]));
        const sortedIds = [];
        const visit = (id) => {
            if (visited.has(id)) return;
            visited.add(id);

            const nested = idToNested.get(id) || [];
            nested.forEach((nestedId) => visit(nestedId));
            sortedIds.push(id);
        };
        graphIds.forEach((id) => visit(id));
        return sortedIds.map((id) => idToGraph.get(id));
    }

    /**
     * Resolve inheritance in passed node, by adding interfaces and properties from
     * its parents.
     *
     * @param node Node with unresolved inheritance.
     * @returns node with resolved inheritance
     */
    extendNodeSpecification(node) {
        const extendedNode = node;

        if (!node.extends) {
            return extendedNode;
        }

        const { nodeData } = node;

        const updateParentCategory = (n) => {
            if (n.extends.length === 0) {
                return;
            }

            const parentType = n.extends[0];

            const parentSpec = this.baklavaView.editor.parentNodes.get(parentType);
            extendedNode.nodeData.category = parentSpec.category;
        };

        const updateParentLayers = (n) => {
            if (n.extends.length === 0) {
                return;
            }

            const parentType = n.extends[0];

            const parentSpec = this.baklavaView.editor.parentNodes.get(parentType);
            extendedNode.nodeData.layer = parentSpec.layer;
        };

        if (!nodeData.layer) {
            updateParentLayers(node);
        }

        if (!nodeData.category) {
            updateParentCategory(node);
        }

        let inheritedProperties = [];
        let inheritedInterfaces = [];

        const updateParentParams = (n) => {
            n.extends?.forEach(
                (parentType) => {
                    const parentSpec = this.baklavaView.editor.parentNodes.get(parentType);
                    inheritedProperties = [
                        ...inheritedProperties,
                        ...(parentSpec?.properties ?? []),
                    ];
                    inheritedInterfaces = [
                        ...inheritedInterfaces,
                        ...(parentSpec?.interfaces ?? []),
                    ];
                    updateParentParams(parentSpec);
                },
            );
        };

        updateParentParams(node);

        const oldProperties = node.properties ? node.properties.filter(
            (prop) => !inheritedProperties.some((p) => p.name === prop.name)) : [];
        const oldInterfaces = node.interfaces ? node.interfaces.filter(
            (interf) => !inheritedInterfaces.some((i) => i.name === interf.name)) : [];

        extendedNode.interfaces = [...oldInterfaces, ...inheritedInterfaces];
        extendedNode.properties = [...oldProperties, ...inheritedProperties];

        return extendedNode;
    }

    /**
     * Reads and validates part of specification related to nodes and subgraphs
     * @param {object?} dataflowSpecification Specification to load
     * @param {Map<string, string[]>} idToNested Mapping between graph ID to its nested graphs.
     *
     * @returns An object consisting of errors and warnings arrays. If any array is empty
     * the updating process was successful.
     */
    async updateGraphSpecification(dataflowSpecification, idToNested) {
        const warnings = [];

        if (!dataflowSpecification) return { errors: ['No specification passed'], warnings };

        const { nodes, graphs, metadata } = dataflowSpecification;
        if (nodes === undefined && graphs === undefined) return { errors: [], warnings };

        let resolvedNodes = [];

        // Store abstract nodes before removing them
        nodes?.forEach((node) => {
            if (node.abstract) {
                this.baklavaView.editor.parentNodes.set(node.name, node);
            }
        });

        try {
            resolvedNodes = EditorManager.preprocessNodes(nodes ?? []);
        } catch (e) {
            return { errors: [e.message], warnings };
        }

        resolvedNodes.forEach((node) => {
            if (node.isCategory) {
                this.baklavaView.editor.parentNodes.set(node.name, node);
            }
        });

        const errors = [];
        errors.push(...this.validateResolvedSpecification(
            { graphs, nodes: resolvedNodes, metadata },
        ));
        if (errors.length) {
            return { errors, warnings };
        }

        resolvedNodes.forEach((node) => {
            const inherited = this.findSimpleInheritedAttributes(node.name, resolvedNodes);
            if (inherited.category && !node.category) {
                node.category = inherited.category;
                node.simpleInherited ??= [];
                node.simpleInherited.push('category');
            }
        });

        this.specification.currentSpecification.nodes = JSON.parse(JSON.stringify(resolvedNodes));
        this.specification.currentSpecification.graphs = JSON.parse(JSON.stringify(graphs));

        // Resolving siblings, parents and children
        resolvedNodes.forEach((node) => errors.push(
            ...this.resolveAffinities(node, resolvedNodes),
        ));
        if (errors.length) {
            return { errors, warnings };
        }

        if (globalProperties.softLoad) {
            // Check if subgraph id exists
            resolvedNodes.forEach((node) => {
                if (node.subgraphId !== undefined) {
                    // Find that id in graph

                    const templateGraphs = graphs.filter((graph) => graph.id === node.subgraphId);

                    if (templateGraphs.length !== 1) {
                        warnings.push(`Cannot find subgraph with id ${node.subgraphId} for ` +
                            `node ${node.name}`,
                        );

                        node.subgraphId = undefined;
                    }
                }
            });
        }

        resolvedNodes.forEach((node) => {
            errors.push(...this.validateNodeStyle(node));
            errors.push(...this._registerNodeType(node));
        });
        nodes.forEach((n) => {
            if (n.subgraphId !== undefined) {
                return;
            }
            const sid = this.findParentSubgraph(n.name);
            if (sid) {
                n.subgraphId = sid;
                n.subgraphIdInherited = true;
            }
        });

        if (errors.length && !globalProperties.softLoad) {
            return { errors, warnings };
        }

        if (typeof this.validating === 'boolean') {
            this.validating = true;
        } else {
            this.validating.value = true;
        }

        this.setValidating(true);
        this.relatedGraphsStore = [];
        const visitedSubgraphs = new Set();
        if (graphs !== undefined) {
            const idToGraph = new Map(graphs.map((graph) => [graph.id, graph]));
            const sortedGraphs = EditorManager.sortGraphs(graphs, idToNested);

            // eslint-disable-next-line no-restricted-syntax
            const subgraphNodes = nodes.filter(({ subgraphId }) => subgraphId !== undefined);
            const subgraphIdToNodes = new Map(nodes.map((node) => [node.subgraphId, []]));
            subgraphNodes.forEach((node) => subgraphIdToNodes.get(node.subgraphId).push(node));
            const subgraphNodesWithGraphs = sortedGraphs
                .map((graph) => [subgraphIdToNodes.get(graph.id), graph])
                .filter(([graphNodes]) => graphNodes)
                .flatMap(([graphNodes, graph]) =>
                    graphNodes.map((node) =>
                        (node.subgraphIdInherited ? [node, graph, node.name] :
                            [node, graph, graph.name])));

            const relatedGraphIds = nodes
                .map((node) => node.relatedGraphs?.map(({ id }) => id))
                .filter((value) => value !== undefined)
                .flat();
            const validated = new Set([]);
            const validateGraph = async (graph, loadArgs = []) => {
                // Validating the graph after it is registered to see if there are any errors
                // by loading a graph (with nested subgraphs, if applicable)

                const graphIds = idToNested.get(graph.id)
                    .filter((id) => !validated.has(id));
                graphIds.forEach((id) => validated.add(id));
                const graphs_ = graphIds.map((id) => idToGraph.get(id));
                const {
                    errors: loadingErrors,
                    warnings: loadingWarnings,
                } = await this.loadDataflow({ // eslint-disable-line no-await-in-loop
                    graphs: [graph, ...graphs_],
                    version: dataflowSpecification.version,
                }, ...loadArgs);
                this.baklavaView.editor.cleanEditor();

                if (loadingWarnings.length) warnings.push(`Graph '${graph.name ?? graph.id}' is invalid:`, ...loadingWarnings.map((w) => `    ${w}`));
                if (loadingErrors.length) errors.push(`Graph '${graph.name ?? graph.id}' is invalid:`, ...loadingErrors.map((w) => `    ${w}`));
            };

            // Validate subgraphs
            // eslint-disable-next-line no-restricted-syntax
            for (const [node, graph, graphName] of subgraphNodesWithGraphs) {
                visitedSubgraphs.add(graph.id);

                const myGraph = GraphFactory(
                    graph.nodes,
                    graph.connections,
                    graphName,
                    this.baklavaView.editor,
                );

                // If `myGraph` is any array then it is an array of errors
                if (Array.isArray(myGraph) && myGraph.length) {
                    errors.push(...myGraph);
                    continue; // eslint-disable-line no-continue
                }

                this.baklavaView.editor.addGraphTemplate(
                    myGraph,
                    resolvedNodes.find((n) => n.name === node.name),
                );

                const loadArgs = [true, true, node.name];

                this.baklavaView.editor.cleanEditor();
                // eslint-disable-next-line no-await-in-loop
                await validateGraph(graph, loadArgs);
            }

            const subgraphNotFoundErrors = Object.entries(subgraphIdToNodes)
                .filter(([subgraphId]) => !visitedSubgraphs.has(subgraphId))
                .flatMap(([subgraphId, graphNodes]) => graphNodes.map((node) => [subgraphId, node]))
                .map(([subgraphId, node]) => `The subgraph with ID ${subgraphId} for node ${node.name} was not found`);
            errors.push(...subgraphNotFoundErrors);

            // Validate related graphs
            for (const graphId of relatedGraphIds) { // eslint-disable-line no-restricted-syntax
                const graph = graphs.find(({ id }) => id === graphId);
                if (graph === undefined) {
                    errors.push([`The related graph with ID ${graphId} was not found`]);
                    continue; // eslint-disable-line no-continue
                }

                const loadArgs = [true, true];

                // eslint-disable-next-line no-await-in-loop
                await validateGraph(graph, loadArgs);

                if (this.relatedGraphsStore.find((el) => graphId === el.id)) {
                    continue; // eslint-disable-line no-continue
                }

                const newGraph = new Graph(this.baklavaView.editor);
                newGraph.load(graph);
                this.relatedGraphsStore.push(newGraph);
            }
            this.relatedGraphsStore.forEach((g) => this.baklavaView.editor.registerGraph(g));
            this.editor._graph.id = uuidv4();
        }

        this.setValidating(false);

        // Removing duplicate warnings
        const uniqueWarnings = [...new Set(warnings)];

        // Registering default categories
        const { errors: defaultErrors, warnings: defaultWarnings } = this.registerDefaultNodes();
        errors.push(...defaultErrors);
        uniqueWarnings.push(...defaultWarnings);

        // Load entry graph and specification graphs
        const entryGraphId = dataflowSpecification.entryGraph;
        if (!errors.length && entryGraphId !== undefined) {
            const idToGraph = new Map(graphs?.map((graph) => [graph.id, graph]));

            const graph = idToGraph.get(entryGraphId);
            if (graph !== undefined) {
                graphs.flatMap((g) => g.nodes).forEach((node) => delete node.graphState);
                const graphs_ = idToNested.get(graph.id).map((id) => idToGraph.get(id));
                const standaloneGraphs = graphs.filter((g) =>
                    !visitedSubgraphs.has(g.id) && !graphs_.includes(g) && graph !== g);
                const {
                    errors: entryErrors,
                } = await this.loadDataflow({
                    graphs: [graph, ...graphs_, ...standaloneGraphs],
                    version: dataflowSpecification.version,
                    entryGraph: entryGraphId,
                });
                this.relatedGraphsStore.forEach((g) =>
                    this.baklavaView.editor.registerGraph(g));
                if (entryErrors && entryErrors.length !== 0) {
                    entryErrors.forEach((e) => errors.push(e));
                    const newGraphInstance = new Graph(this.baklavaView.editor);
                    this.baklavaView.editor.displayedGraph = newGraphInstance;
                    this.baklavaView.editor.deepCleanEditor();
                    this.baklavaView.editor.unregisterGraphs();
                }
            } else {
                uniqueWarnings.push(`'entryGraph' points to undefined graph: '${entryGraphId}'`);
            }

            // preload every graphs in specification
            graphs.flatMap((g) => g.nodes).forEach((node) => delete node.graphState);
            if (graphs.length > 0) {
                graphs.forEach((g) => {
                    this.editor.addPreloadedGraph(g);
                });
            }
        }

        if (!errors.length) {
            const idToGraph = new Map(graphs?.map((graph) => [graph.id, graph]));
            if (entryGraphId !== undefined && idToGraph.get(entryGraphId) === undefined) {
                uniqueWarnings.push(`'entryGraph' points to undefined graph: '${entryGraphId}'`);
            }
        }

        return { errors, warnings: uniqueWarnings };
    }

    /**
     * Preprocess nodes to be later passed to `resolveInheritance` function.
     *
     * @param nodes coming from specification.
     * @returns preprocessed nodes.
     */
    static preprocessNodes(nodes) {
        nodes.filter((node) => node.isCategory)
            .forEach((node) => { node.name = EditorManager.getNodeName(node); });
        return nodes;
    }

    /**
     * Reads and validates metadata from specification and loads it into the editor.
     * if no metadata is passed it uses a stored specification.
     *
     * @param metadata metadata to load
     * @param overriding tells whether the metadata is updated on dataflow loading
     * @param loading resets updated metadata, should be used when loading new dataflow
     * @returns An array of errors that occurred during the metadata loading.
     */
    updateMetadata(metadata = undefined, overriding = false, loading = false, resetIgnored = true) {
        if (loading) this.updatedMetadata = {};
        let newMetadata;
        if (metadata !== undefined) {
            metadata = { ...this.updatedMetadata, ...metadata };
            newMetadata = JSON.parse(JSON.stringify(metadata));
        }
        if (metadata === undefined && this.specification.currentSpecification) {
            metadata = this.specification.currentSpecification.metadata ?? {};
        }

        if (!metadata) return ['No specification to load provided.'];

        if (overriding) {
            // this.specification.currentSpecification?.metadata should not
            // be over overridden, that is why it needs to be copied before merging
            const tempMetadata = metadata;
            metadata = JSON.parse(JSON.stringify(
                this.specification.currentSpecification?.metadata ?? {}));
            EditorManager.mergeObjects(metadata, tempMetadata);
        }

        this.baklavaView.interfaceTypes.readInterfaceTypes(metadata);
        this.baklavaView.propertyStyler.readPropertyStyles(metadata);
        this.baklavaView.nodeStyler.readNodeStyles(metadata);
        this.baklavaView.interfaceStyler.readInterfaceStyles(metadata);

        if (metadata && 'urls' in metadata) {
            Object.entries(metadata.urls).forEach(([urlName, state]) => {
                this.baklavaView.editor.baseURLs.set(urlName, state);
            });
        }

        if (metadata && 'icons' in metadata) {
            Object.entries(metadata.icons).forEach(([iconName, state]) => {
                this.baklavaView.editor.baseIconUrls.set(iconName, state);
            });
        }

        if (metadata && 'navbarItems' in metadata) {
            this.baklavaView.navbarItems = JSON.parse(JSON.stringify(metadata.navbarItems));
        }

        if (metadata && 'contextMenuActions' in metadata) {
            this.baklavaView.contextMenuActions = JSON.parse(
                JSON.stringify(metadata.contextMenuActions),
            );
        }

        if (metadata && 'subgraphContextMenuActions' in metadata) {
            this.baklavaView.subgraphContextMenuActions = JSON.parse(
                JSON.stringify(metadata.subgraphContextMenuActions),
            );
        }

        this.baklavaView.editor.readonly = metadata?.readonly ?? this.defaultMetadata.readonly;
        this.baklavaView.editor.hideHud = metadata?.hideHud ?? this.defaultMetadata.hideHud;

        this.baklavaView.editor.nodeStyles.set(NEW_NODE_STYLE, { icon: 'NewNode' });
        this.baklavaView.editor.nodeStyles.set(EDITED_NODE_STYLE, { icon: 'EditedNode' });
        Object.entries(metadata?.styles ?? {}).forEach(([key, value]) => {
            this.baklavaView.editor.nodeStyles.set(key, value);
        });
        Object.entries(metadata?.nodeLists ?? {}).forEach(([key, value]) => {
            this.baklavaView.editor.nodeLists.set(key, value);
        });

        this.editor.allowLoopbacks =
            metadata?.allowLoopbacks ?? this.defaultMetadata.allowLoopbacks;
        this.baklavaView.twoColumn = metadata?.twoColumn ?? this.defaultMetadata.twoColumn;
        this.baklavaView.connectionRenderer.style =
            metadata?.connectionStyle ?? this.defaultMetadata.connectionStyle;

        this.baklavaView.movementStep = metadata?.movementStep ?? this.defaultMetadata.movementStep;
        this.baklavaView.settings.background.gridSize =
            metadata?.backgroundSize ?? this.defaultMetadata.backgroundSize;
        this.baklavaView.settings.background.color =
            metadata?.backgroundColor ?? this.defaultMetadata.backgroundColor;
        this.baklavaView.connectionRenderer.randomizedOffset =
            metadata?.randomizedOffset ?? this.defaultMetadata.randomizedOffset;
        this.baklavaView.settings.toggleableEditableTypes =
            metadata?.toggleableEditableTypes ?? this.defaultMetadata.toggleableEditableTypes;
        this.baklavaView.settings.editableNodeTypes =
            metadata?.editableTypes ?? this.defaultMetadata.editableNodeTypes;
        this.baklavaView.settings.hideAnchors =
            metadata?.hideAnchors ?? this.defaultMetadata.hideAnchors;
        this.baklavaView.settings.showIds =
            metadata?.showIds ?? this.defaultMetadata.showIds;
        this.baklavaView.settings.newGraphNode =
            metadata?.newGraphNode ?? this.defaultMetadata.newGraphNode;
        this.baklavaView.settings.newNodeType =
            metadata?.newNodeType ?? this.defaultMetadata.newNodeType;
        this.baklavaView.settings.showHiddenProperties =
            metadata?.showHiddenProperties ?? this.defaultMetadata.showHiddenProperties;
        this.baklavaView.settings.disableInterfaceExpose =
            metadata?.disableInterfaceExpose ?? this.defaultMetadata.disableInterfaceExpose;
        this.baklavaView.connectionRenderer.switchableInterfaces =
            metadata?.switchableInterfaces ?? this.defaultMetadata.switchableInterfaces;
        this.baklavaView.settings.removeEmptySubgraphs =
            metadata?.removeEmptySubgraphs ?? this.defaultMetadata.removeEmptySubgraphs;

        if (resetIgnored) {
            this.baklavaView.ignoredLayers = new Map();
        }
        this.baklavaView.layers = metadata?.layers ?? this.defaultMetadata.layers;
        this.baklavaView.collapseSidebar =
            metadata?.collapseSidebar ?? this.defaultMetadata.collapseSidebar;
        this.baklavaView.editor.layoutManager.useAlgorithm(
            metadata?.layout ?? this.defaultMetadata.layout,
        );
        this.baklavaView.logLevel = metadata?.logLevel ?? this.defaultMetadata.logLevel;
        this.baklavaView.welcome = metadata?.welcome ?? this.defaultMetadata.welcome;
        if (newMetadata) this.updatedMetadata = newMetadata;

        return [];
    }

    /**
     * Finds all inherited interfaces for node Type.
     *
     * @param nodeType type of node for which attributes should be searched
     * @returns array of unique interfaces found in nodeType's parents.
     */
    findInheritedInterfaces(nodeType) {
        const nodeSpec = this.specification.currentSpecification.nodes
            .find((spec) => nodeType === spec.name) ?? [];
        if ((nodeSpec.extends?.length ?? 0) === 0) {
            return [];
        }
        const parentSpec = this.specification.currentSpecification.nodes
            .filter((spec) => nodeSpec.extends?.includes(spec.name)) ?? [];
        let interfaces = [];
        parentSpec.forEach((p) => {
            if (!p) {
                return;
            }
            if (p?.interfaces !== undefined) {
                interfaces = [...interfaces, ...p.interfaces];
            }
            const inherited = this.findInheritedInterfaces(p.name);
            interfaces = [...interfaces, ...inherited];
        });
        const uniqueInterfaces = interfaces.filter((item, pos) => {
            const idx = interfaces.findIndex((item2) =>
                item.name === item2.name && (item2.override ||
                    (item.array === item2.array &&
                        item.direction === item2.direction)));
            return idx === pos;
        });
        return uniqueInterfaces;
    }

    /**
     * Finds all inherited properties for node Type.
     *
     * @param nodeType type of node for which attributes should be searched
     * @returns array of unique properties found in nodeType's parents.
     */
    findInheritedProperties(nodeType) {
        const nodeSpec = this.specification.currentSpecification.nodes
            .find((spec) => nodeType === spec.name) ?? [];
        if ((nodeSpec.extends?.length ?? 0) === 0) {
            return [];
        }
        const parentSpec = this.specification.currentSpecification.nodes
            .filter((spec) => nodeSpec.extends?.includes(spec.name)) ?? [];
        let properties = [];
        parentSpec.forEach((p) => {
            if (!p) {
                return;
            }
            if (p?.properties !== undefined) {
                properties = [...properties, ...p.properties];
            }
            const inherited = this.findInheritedProperties(p.name);
            properties = [...properties, ...inherited];
        });
        const uniqueProperties = properties.filter((item, pos) =>
            properties.findIndex((item2) => item.name === item2.name) === pos,
        );
        return uniqueProperties;
    }

    /**
     * Finds simple inherited attributes: category, layer and icon
     *
     * @param nodeType type of node for which attributes should be searched
     * @param nodes optional list of nodes to search through
     * @returns Object with properties category, layer, icon. If not set, no inherited
     * attribute of that type was found.
     */
    findSimpleInheritedAttributes(nodeType, nodes = undefined) {
        if (!nodes) {
            nodes = this.specification.currentSpecification.nodes;
        }
        const simpleAttributes = ['category', 'layer', 'icon'];
        const nodeSpec = nodes
            .find((spec) => nodeType === spec.name) ?? [];
        if ((nodeSpec.extends?.length ?? 0) === 0) {
            return [];
        }
        const parentSpec = nodes
            .filter((spec) => nodeSpec.extends?.includes(spec.name)) ?? [];
        let attributes = {};
        parentSpec.forEach((p) => {
            if (!p) {
                return;
            }
            const inherited = this.findSimpleInheritedAttributes(p.name);
            attributes = { ...attributes, ...inherited };
            simpleAttributes.forEach((key) => {
                if (p[key] !== undefined) {
                    attributes[key] = p[key];
                }
            });
        });
        return attributes;
    }

    /**
     * Finds the first parent's subgraph id if it exists
     *
     * @param nodeType type of node for which attributes should be searched
     * @returns subgraphId that was found first or undefined if none were found.
     */
    findParentSubgraph(nodeType) {
        const nodeSpec = this.specification.currentSpecification.nodes
            .find((spec) => nodeType === spec.name) ?? [];
        if (nodeSpec.subgraphId !== undefined) {
            return nodeSpec.subgraphId;
        }
        const parentSpec = this.specification.currentSpecification.nodes
            .filter((spec) => nodeSpec.extends?.includes(spec.name)) ?? [];
        let sid;
        parentSpec.forEach((p) => {
            if (!p || sid !== undefined) {
                return;
            }
            sid = this.findParentSubgraph(p.name);
        });
        return sid;
    }

    /**
     * Nodes that have already been resolved and have gone through JSON parsing
     *
     * @param node
     * @param resolvedNodes
     * @returns nodes marked as inherited
     */
    /* eslint-disable class-methods-use-this,no-param-reassign */
    resolveAffinities(node, resolvedNodes) {
        const errors = [];
        // Resolving children
        (node.extends ?? []).forEach((eName) => {
            const extended = resolvedNodes.find((n) => n.name === eName);

            // The extended node could be abstract, in which way it is not in resolved nodes.
            if (extended !== undefined) {
                if (extended.extending === undefined) {
                    extended.extending = [];
                }
                extended.extending.push(node.name);
            }
        });

        const siblings = new Set();
        (node.extends ?? []).forEach((eName) => {
            const extended = resolvedNodes.find((n) => n.name === eName);

            // The extended node could be abstract, in which way it is not in resolved nodes.
            if (extended !== undefined) {
                extended.extending.forEach((e) => siblings.add(e));
            }
        });
        siblings.delete(node.name);
        node.siblings = Array.from(siblings);

        if (node.extends) {
            const copyExtends = [...node.extends];
            node.extends = [];
            copyExtends.forEach((eName) => {
                if (resolvedNodes.find((n) => n.name === eName) !== undefined) {
                    node.extends.push(eName);
                } else {
                    errors.push(
                        `No parent node of type ${eName} found for node type ${node.name}`,
                    );
                }
            });
        }
        return errors;
    }

    /**
     * Validate styles of a node.
     *
     * @param node node to verify.
     * @returns errors if the node has non-existing or repeated styles.
     */
    validateNodeStyle(node) {
        const errors = [];

        if (node.style === undefined) return errors;

        const style = Array.isArray(node.style) ? node.style : [node.style];
        const styleSet = new Set(style);
        if (styleSet.size !== style.length) {
            errors.push(`Repeated styles in "${EditorManager.getNodeName(node)}" node`);
        }

        errors.push(...Array.from(styleSet)
            .filter((styleName) => !this.baklavaView.editor.nodeStyles.has(styleName))
            .map((styleName) => `Non-existing style "${styleName}" in '${EditorManager.getNodeName(node)}' node`));

        return errors;
    }

    /**
     * Serializes and returns current specification in Pipeline Manager format.
     *
     * @returns Serialized specification.
     */
    saveSpecification() {
        const specification =
            JSON.parse(JSON.stringify(this.specification.unresolvedSpecification));

        // Save spec graphs
        const graphs = this.editor.savePreloadedGraphsState();
        specification.graphs = graphs;
        EditorManager.unmarkNewNodes(specification);
        return specification;
    }

    /**
     * Removes NEW_NODE_STYLE node styles.
     *
     * @param specification specification to modify.
     * @returns warnings about removed styles.
     */
    static unmarkNewNodes(specification) {
        const warnings = [];

        const warn = (node) => {
            warnings.push(`Loaded node '${EditorManager.getNodeName(node)}' has '${NEW_NODE_STYLE}' style, removing it.`);
        };

        specification.nodes?.forEach((node) => {
            if (node.style === NEW_NODE_STYLE) {
                warn(node);
                delete node.style;
            } else if (Array.isArray(node.style) && node.style.includes(NEW_NODE_STYLE)) {
                warn(node);
                node.style.splice(node.style.indexOf(NEW_NODE_STYLE), 1);
            }
        });

        return warnings;
    }

    /**
     * Returns the top-level graph.
     *
     * @returns {object} baklava format graph.
     */
    getRootGraph() {
        return this.editor.subgraphStack.length
            ? this.editor.subgraphStack[0]
            : this.editor.graph;
    }

    /**
     * Clears history for every graph.
     */
    clearHistory(callback = () => undefined) {
        this.baklavaView.history.clearHistory(callback);
    }

    /**
     * Serializes and returns current dataflow in Pipeline Manager format.
     *
     * @param {Object} obj Parameters.
     * @param {Boolean} obj.readonly whether the dataflow should be saved in readonly mode
     * @param {Boolean} obj.hideHud whether the dataflow should be saved in hideHud mode
     * @param {Boolean} obj.position whether the dataflow should store panning and scaling values
     * @param {Boolean} obj.hideLayers whether the dataflow should store hidden layers
     * @param {string|null|undefined} obj.graphName graph name which is rendered to the user
     *
     * @returns Serialized dataflow.
     */
    /* eslint-disable-next-line object-curly-newline */
    saveDataflow({ readonly, hideHud, position, hideLayers, graphName } = {}) {
        const save = this.baklavaView.editor.save();
        save.version = this.specificationVersion;

        const entryGraph = save.entryGraph
            ? save.graphs.find((dataflowGraph) => dataflowGraph.id === save.entryGraph)
            : save.graphs[0];

        if (entryGraph && graphName !== null && graphName !== undefined) {
            const rootGraph = this.getRootGraph();
            rootGraph.name = graphName;
            if (rootGraph === this.editor.graph) this.baklavaView.editor.graphName = graphName;
        }

        if (!position) {
            save.graphs.forEach((graph) => {
                delete graph.panning;
                delete graph.scaling;
            });
        }

        // Remove 'hidden' if it aligns with specification
        const nodeToProps = EditorManager.getEditorManagerInstance()
            .specification
            .currentSpecification
            ?.nodes
            .map((node) => {
                const properties = (node.properties ?? [])
                    // Unwrap groups
                    .flatMap((prop) => (
                        prop.group ? [prop, ...Object.entries(prop.group)] : [prop]),
                    ).reduce((acc, prop) => ({ ...acc, [prop.name]: prop }), {});
                return [EditorManager.getNodeName(node), properties];
            })
            .reduce((acc, [name, properties]) => ({ ...acc, [name]: properties }), {});

        save.graphs
            .flatMap((graph) => graph.nodes ?? [])
            .flatMap((node) => node.properties
                .map((prop) => [prop, nodeToProps[EditorManager.getNodeName(node)][prop.name]]))
            .filter(([savedProp, specProp]) => savedProp !== undefined && specProp !== undefined)
            .filter(([savedProp, specProp]) =>
                savedProp?.hidden !== undefined &&
                specProp?.hidden !== undefined &&
                savedProp.hidden === Boolean(specProp.hidden))
            .forEach(([savedProp, _]) => { delete savedProp.hidden; });

        if (hideLayers) {
            save.graphs.forEach((graph) => {
                graph.disabledLayers = [...(this.baklavaView.ignoredLayers.get(graph.id) ?? [])];
            });
        }

        if (save.metadata === undefined) {
            save.metadata = {};
        }

        [
            [readonly, 'readonly'],
            [hideHud, 'hideHud'],
            [this.editor.allowLoopbacks, 'allowLoopbacks'],
            [this.baklavaView.twoColumn, 'twoColumn'],
            [this.baklavaView.connectionRenderer.style, 'connectionStyle'],
            [this.baklavaView.movementStep, 'movementStep'],
            [this.baklavaView.settings.background.gridSize, 'backgroundSize'],
            [this.baklavaView.settings.background.color, 'backgroundColor'],
            [this.baklavaView.connectionRenderer.randomizedOffset, 'randomizedOffset'],
            [this.baklavaView.settings.editableNodeTypes, 'editableTypes'],
            [this.baklavaView.settings.toggleableEditableTypes, 'toggleableEditableTypes'],
            [this.baklavaView.settings.hideAnchors, 'hideAnchors'],
            [this.baklavaView.settings.showIds, 'showIds'],
            [this.baklavaView.settings.newGraphNode, 'newGraphNode'],
            [this.baklavaView.settings.newNodeType, 'newNodeType'],
            [this.baklavaView.settings.showHiddenProperties, 'showHiddenProperties'],
        ].forEach(([currVal, name]) => {
            const m = this.specification.currentSpecification?.metadata ?? {};
            const dm = this.defaultMetadata;

            if (currVal !== (m[name] ?? dm[name])) {
                save.metadata[name] = currVal;
            }
        });

        if (Object.keys(save.metadata).length === 0) {
            delete save.metadata;
        }

        return save;
    }

    /**
     * Loads the dataflow passed in `dataflow` and renders it.
     * If the dataflow is not compatible with the currently loaded specification or is not
     * in the dataflow format, then some of the dataflow may be not loaded and an
     * error is returned. Dataflow should be passed in PipelineManager format (translation
     * to Baklava format is done )
     *
     * @param dataflow Dataflow to load. Can be either an object or a string
     * @param preventCentering Boolean Blocks view in the same spot.
     * @param loadOnly determines whether to load the graph only without adjusting
     * @param templateName {string|null} name of the template, if the graph is a template
     * the graph rendering. Can be used when validating graphs without their browser
     * representation
     * @param centerAtTop determines whether to center the editor at the topmost node
     * after loading.
     * @returns An array of errors that occurred during the dataflow loading.
     * If the array is empty, the loading was successful.
     */
    async loadDataflow(
        dataflow, preventCentering = false, loadOnly = false, templateName = null,
        centerAtTop = false,
    ) {
        let { notifyWhenChanged } = this;

        const status = {
            errors: [],
            warnings: [],
            info: [],
        };

        // Turn off notification during dataflow loading
        this.updateMetadata({ notifyWhenChanged: false }, true, true);

        try {
            const validationErrors = EditorManager.validateDataflow(dataflow);
            if (validationErrors.length && !globalProperties.softLoad) {
                status.errors = validationErrors;

                return status;
            }
            if (validationErrors.length && globalProperties.softLoad) {
                status.warnings.push(validationErrors.toString());
            }

            try {
                if (typeof dataflow === 'string' || dataflow instanceof String) {
                    dataflow = jsonlint.parse(dataflow);
                }

                const specificationVersion = dataflow.version;

                if (specificationVersion === undefined) {
                    status.warnings.push(
                        `Loaded dataflow has no version assigned. Please update the dataflow to version ${this.specificationVersion}.`,
                    );
                } else if (specificationVersion !== this.specificationVersion) {
                    status.info.push(
                        `Dataflow version (${specificationVersion}) differs from the current version (${this.specificationVersion}). It may result in unexpected behaviour.`,
                    );
                }

                if ('metadata' in dataflow && this.specification.currentSpecification !== undefined) {
                    const _errors = EditorManager.validateMetadata(dataflow.metadata);
                    if (Array.isArray(_errors) && _errors.length) {
                        if (!globalProperties.softLoad) {
                            status.errors.push(..._errors);
                            return status;
                        }

                        status.warnings.push(..._errors);
                    }

                    notifyWhenChanged = dataflow.metadata.notifyWhenChanged ?? notifyWhenChanged;

                    this.updateMetadata(
                        { ...dataflow.metadata, notifyWhenChanged: false },
                        true,
                        true,
                    );
                }
                if (this.baklavaView.displayedGraph !== undefined) {
                    // Delete baklava internal history listeners
                    this.baklavaView.history.unsubscribeFromGraphEvents(
                        this.baklavaView.displayedGraph,
                        Symbol('HistoryToken'),
                    );
                }

                let isWebpack = true;
                try {
                    isWebpack = window.isWebpack;
                } catch {
                    isWebpack = false;
                }

                if (!isWebpack) {
                    this.baklavaView.ignoredLayers = new Map(
                        dataflow.graphs.map(
                            (graph) => [graph.id, new Set(graph.disabledLayers ?? [])],
                        ),
                    );
                    const result = await this.baklavaView.editor.load(
                        dataflow,
                        preventCentering,
                        loadOnly,
                        templateName,
                        centerAtTop,
                    );

                    if (!globalProperties.softLoad) {
                        status.errors.push(...(result.errors ?? []));
                        status.warnings.push(...(result.warnings ?? []));
                    } else {
                        status.warnings.push(...(result.errors ?? []), ...(result.warnings ?? []));
                    }

                    this.baklavaView.history.graphSwitch(
                        this.baklavaView.displayedGraph,
                        this.baklavaView.displayedGraph,
                    );
                }

                const arr = this.verifyExposedInterfaceNamesMatchExternalNames(dataflow);

                if (globalProperties.softLoad) {
                    status.warnings.push(...arr);
                } else {
                    status.errors.push(...arr);
                }

                return status;
            } catch (err) {
                const msg = `Unrecognized format. Make sure that the passed dataflow is correct.${err.toString()}`;

                if (globalProperties.softLoad) {
                    status.warnings.push(msg);
                } else {
                    status.errors.push(msg);
                }

                return status;
            }
        } finally {
            // Restore previous state or use value from loaded dataflow
            this.updateMetadata({ notifyWhenChanged }, true, false, false);
        }
    }

    /**
     * Verify whether exposed interfaces' names match their counterparts' external names.
     *
     * @param {object} dataflow Dataflow with external names to verify.
     * @returns {Array} Array with errors that occurred
     * during verification.
     */
    verifyExposedInterfaceNamesMatchExternalNames(dataflow) {
        const errors = [];

        // Collect same-id interfaces to groups.
        const sameIdInterfaces = new Map();
        dataflow.graphs.forEach((graph) => {
            graph.nodes.forEach((node) => {
                node.interfaces.forEach((intf) => {
                    if (sameIdInterfaces.has(intf.id)) {
                        sameIdInterfaces.get(intf.id).push(intf);
                    } else {
                        sameIdInterfaces.set(intf.id, [intf]);
                    }
                });
            });
        });

        // Process groups of two or more interfaces to find mismatches.
        sameIdInterfaces.forEach((interfaces, sharedId) => {
            // Single-copy interface is not exposed.
            if (interfaces.length < 2) {
                return;
            }

            const exposedInterfaces = interfaces.filter(
                (intf) => intf.externalName !== undefined,
            );

            if (exposedInterfaces.length < interfaces.length - 1) {
                errors.push(`The interface with id = ${sharedId} seems to ` +
                    `be exposed but lacks "externalName" property.`);
                return;
            }

            const bottomers = new Set();
            let numMatched = 0;
            interfaces.forEach((intf) => {
                const found = interfaces.find((intf2) => intf2.externalName === intf.name &&
                    intf2 !== intf && !bottomers.has(intf2));
                if (found) {
                    // each interface only once can be treated as 'underlying'
                    bottomers.add(found);
                    numMatched += 1;
                }
            });
            // expect number of interfaces - 1, bottommost interface will be a leaf
            // < for simpler logic (allowing cycles)
            if (numMatched < interfaces.length - 1) {
                const errorMessage =
                    `Mismatch between "externalName" of the original interface ` +
                    `and "name" of the exposed version of the interface, ` +
                    `for the interface with id = ${sharedId}\n`;
                errors.push(errorMessage);
            }
        });

        return errors;
    }

    /**
     * Static function used to get the instance of the EditorManager in a singleton manner.
     * If there is no existing instance of the EditorManager then a new one is created.
     *
     * @returns {EditorManager} Instance of EditorManager.
     */
    static getEditorManagerInstance() {
        if (!EditorManager.instance) {
            EditorManager.instance = new EditorManager();
        }
        return EditorManager.instance;
    }

    /**
     * Static function used to apply string replacement mapping on urls.
     * @param specification - specification to be modified.
     * @param overrides - mapping between old and new values.
     */
    static applyUrlOverrides(specification, overrides) {
        Object.entries(overrides).forEach(([oldValue, newValue]) => {
            // Icons
            Object.entries(specification.metadata?.icons ?? {}).forEach(([key, value]) => {
                specification.metadata.icons[key] = value.replaceAll(oldValue, newValue);
            });

            // urls
            Object.values(specification.metadata?.urls ?? {}).forEach((item) => {
                item.url = item.url.replaceAll(oldValue, newValue);
            });
        });
    }

    /**
     * Adds style to all nodes in the specification.
     *
     * @param {object} specification specification to modify.
     * @param {string|Array<string>} style style to be applied
     */
    static includeWithStyle(specification, style) {
        specification.nodes?.forEach((node) => {
            const mergeStyles = EditorManager.mergeStyles(node.style, style);
            if (mergeStyles !== undefined) node.style = mergeStyles;
        });
    }

    /**
     * Merges two styles.
     *
     * @param {string|Array<string>|undefined} style1 first style.
     * @param {string|Array<string>|undefined} style2 second style.
     * @returns {Array<string>|undefined} merged style.
     */
    static mergeStyles(style1, style2) {
        if (style1 === style2) return style1;
        if ((style1 && style2) === undefined) return style1 ?? style2;

        [style1, style2] = [style1, style2]
            .map((style) => (Array.isArray(style) ? style : [style]))
            .map((style) => new Set(style));

        style1 = style1.difference(style2);

        return Array.from(style1.union(style2));
    }

    /**
     * Static helper function to merge two object instances into a single.
     * The following rules are applied:
     * - If the property is an array then it is concatenated
     * - If the property is an object then it is merged with preference to the first object
     * - If the property is a simple type then it is overwritten with the first object
     * - On type mismatch (array/object), the first object is used
     *
     * @param primaryObject First object to merge
     * @param secondaryObject Second object to merge
     * @returns Primary object with merged properties from the secondary object
     */
    static mergeObjects(primaryObject, secondaryObject) {
        const warnings = [];
        const errors = [];

        // Check if any of the object is undefined
        secondaryObject = secondaryObject ?? {};
        if (primaryObject === undefined || Object.keys(primaryObject).length === 0) {
            return { errors, warnings };
        }

        // Merge object
        Object.entries(secondaryObject).forEach(([key, value]) => {
            if (Array.isArray(value) && Array.isArray(primaryObject[key])) {
                if (key === 'graphs') {
                    // Merge graphs by ID
                    const objsToMerge = [value, primaryObject[key]]
                        .map((graphs) => graphs.map((graph) => [graph.id, graph]))
                        .map(Object.fromEntries);
                    primaryObject[key] = Object.values(Object.assign({}, ...objsToMerge));
                    return;
                }

                if (key !== 'nodes') {
                    primaryObject[key].push(...value);
                    return;
                }

                // Merge nodes by names
                try {
                    // Rename
                    const renameMapping = Object.fromEntries(primaryObject[key]
                        .filter((node) => node.includeName)
                        .map((node) => [node.includeName, EditorManager.getNodeName(node)]));

                    const usedKeys = new Set();
                    const conditionalRename = (node) => {
                        const name = EditorManager.getNodeName(node);
                        const mapped = renameMapping[name];
                        if (mapped !== undefined) { usedKeys.add(name); }
                        return mapped ?? name;
                    };

                    const objsToMerge = [
                        value.map((node) => [conditionalRename(node), node]),
                        primaryObject[key].map((node) => [EditorManager.getNodeName(node), node]),
                    ].map(Object.fromEntries);

                    // Merge
                    primaryObject[key] = Object.values(Object.assign({}, ...objsToMerge));

                    // Check usage of `includeName` directive
                    const unusedKeys = Object.keys(renameMapping)
                        .filter((name) => !usedKeys.has(name));

                    if (unusedKeys.length) {
                        warnings.push(`Unused include names: ${unusedKeys}`);
                    }
                } catch (error) {
                    errors.push(error);
                }
            } else if (typeof value === 'object' && typeof primaryObject[key] === 'object') {
                // For example, metadata is an object and it has to be merged instead of overwritten
                const {
                    errors: mergeErrors, warnings: mergeWarnings,
                } = EditorManager.mergeObjects(primaryObject[key], value);
                errors.push(...mergeErrors);
                warnings.push(...mergeWarnings);
            } else if (primaryObject[key] === undefined) {
                primaryObject[key] = value;
            }
        });
        return { errors, warnings };
    }

    /**
     * Extracts name of the node.
     *
     * @param {Object} node to check.
     * @returns {string} name of the node.
     * @throws {Error} Throws if there are any naming inconsistencies.
     */
    static getNodeName(node) {
        if (!node.isCategory) {
            if (node.name === undefined) {
                throw new Error(`Non-category node has to define field 'name'`);
            }
            return node.name;
        }
        const name = node.category.split('/').at(-1);
        if (node.name !== undefined && node.name !== name) {
            throw new Error(`Node '${node.name}' is a category node and has a name defined different than ${name}`);
        }
        return name;
    }

    /**
     * Validates JSON data using given JSON schema. If passed `data` is a string that represents
     * text of specification file then more information about potential errors - like the exact
     * line of error - is returned.
     *
     * @param data Specification file to validate. Can be either a parsed JSON object
     * or a textual file
     * @param schema Schema to use
     * @param reference Reference to part of the schema, e.g. node, interface or property.
     * @param additionalAjvOptions Additional options to pass to the Ajv constructor
     * @returns An array of errors. If the array is empty, the validation was successful.
     */
    static validateJSONWithSchema(data, schema, reference = '', additionalAjvOptions = {}) {
        const ajv = new Ajv2019({
            allowUnionTypes: true,
            // Schema used in compile() may be already included in `schemas`.
            addUsedSchema: false,
            formats: {
                hex: /^0x[a-fA-F0-9]+$/,
            },
            schemas: [
                unresolvedSpecificationSchema,
                specificationSchema,
                metadataSchema,
                dataflowSchema,
                graphSchema,
            ],
            ...additionalAjvOptions,
        });
        ajv.addKeyword('version');
        return validateJSON(ajv, schema, data, reference);
    }

    validateResolvedSpecification(specification) {
        const validationErrors = EditorManager.validateSpecification(
            specification, specificationSchema);
        if (validationErrors.length) return validationErrors;

        // Validating category nodes
        const { nodes } = specification;
        const categoryNodes = nodes.filter((node) => node.isCategory);
        const definedCategories = {};

        // Finding multiple category nodes defining the same category
        const errors = [];
        categoryNodes.forEach((node) => {
            if (node.name in definedCategories) {
                errors.push(`Category '${node.category}' has multiple nodes defining it.`);
            } else {
                definedCategories[node.name] = node.category.split('/').slice(0, -1).join('/');
            }
        });

        // Nodes have to extend the first category node in their category path.
        // For example, if we have two category nodes A and C and we have a node e
        // which has a category 'A/b/C/d/e' then it has to extend C (and C has to extend A)
        const nodeNames = new Set();
        nodes.forEach((node) => {
            if (node.category === undefined) {
                node.category = '';
            }
            const categories = node.category.split('/');

            for (let i = categories.length - 1; i >= 0; i -= 1) {
                const categoryNodeName = categories[i];
                const remainingCategories = categories.slice(0, i).join('/');

                if (
                    categoryNodeName in definedCategories &&
                    node.name !== categoryNodeName &&
                    remainingCategories === definedCategories[categoryNodeName]
                ) {
                    if (
                        node.extends === undefined ||
                        !node.extends.includes(categoryNodeName)
                    ) {
                        errors.push(`Node '${node.name}' does not extend its category node '${categoryNodeName}'.`);
                    }
                    break;
                }
            }

            // Nodes that extend from a category node have to be in their subtree i.e. have a common
            // category prefix with the category node
            for (let i = 0; i < (node.extends ?? []).length; i += 1) {
                const extendedNode = node.extends[i];
                const inherited = this.findSimpleInheritedAttributes(node.name);
                const category = node.category ? node.category : (inherited.category ?? '');
                if (extendedNode in definedCategories) {
                    const commonPrefix = definedCategories[extendedNode] !== '' ?
                        `${definedCategories[extendedNode]}/${extendedNode}` : extendedNode;

                    if (!category.includes(commonPrefix)) {
                        errors.push(
                            `Node '${node.name}' extends from a category node '${extendedNode}' but is not in its category`,
                        );
                        break;
                    }
                }
            }

            // Finding multiple nodes with the same name
            if (nodeNames.has(node.name)) {
                errors.push(`Node '${node.name}' is defined multiple times`);
            }
            nodeNames.add(node.name);
        });

        return errors;
    }

    /**
     * Validates a single node passed in `nodeSpecification` using jsonSchema.
     *
     * @param nodeSpecification Node to validate
     * @returns An array of errors. If the array is empty, the validation was successful.
     */
    /* eslint-disable class-methods-use-this */
    validateNode(nodeSpecification, schema = unresolvedSpecificationSchema) {
        return EditorManager.validateJSONWithSchema(nodeSpecification, schema, '#/$defs/node');
    }

    /**
     * Validates a single property passed in `propertySpecification` using jsonSchema.
     *
     * @param propertySpecification Property to validate
     * @returns An array of errors. If the array is empty, the validation was successful.
     */
    validateNodeProperty(
        propertySpecification,
        schema = unresolvedSpecificationSchema,
    ) {
        return EditorManager.validateJSONWithSchema(propertySpecification, schema, '#/$defs/property');
    }

    /**
     * Validates a single interface passed in `interfaceSpecification` using jsonSchema.
     *
     * @param interfaceSpecification Interface to validate
     * @returns An array of errors. If the array is empty, the validation was successful.
     */
    validateNodeInterface(
        interfaceSpecification,
        schema = unresolvedSpecificationSchema,
    ) {
        return EditorManager.validateJSONWithSchema(interfaceSpecification, schema, '#/$defs/interface');
    }

    /**
     * Validates specification passed in `specification` using jsonSchema.
     *
     * @param specification Specification to validate
     * @returns An array of errors. If the array is empty, the validation was successful.
     */
    static validateSpecification(specification, schema = unresolvedSpecificationSchema) {
        return EditorManager.validateJSONWithSchema(specification, schema);
    }

    /**
     * Validates metadata in JSON format using schema from unresolvedSpecificationSchema.
     *
     * @param jsonmetadata metadata in JSON format to validate
     * @return An array of errors. If the array is empty, the validation was successful.
     */
    static validateMetadata(jsonmetadata) {
        return EditorManager.validateJSONWithSchema(jsonmetadata, metadataSchema);
    }

    /**
     * Validates metadata in JSON format using schema from dataflowSchema.
     *
     * @param dataflow dataflow in JSON format to validate
     * @return An array of errors. If the array is empty, the validation was successful.
     */
    static validateDataflow(dataflow) {
        return EditorManager.validateJSONWithSchema(dataflow, dataflowSchema);
    }

    /**
     * Validates message in JSON format using schema from messageSchema.
     *
     * @param message message in JSON format to validate
     * @return An array of errors. If the array is empty, the validation was successful.
     */
    static validateMessage(message) {
        return EditorManager.validateJSONWithSchema(message, messageSchema);
    }

    /**
     * Switches the editor state to main graph
     */
    returnFromSubgraph() {
        this.baklavaView.editor.backFromSubgraph(this.baklavaView.displayedGraph);
    }

    /**
     * Finds a graph that contains a node linking to the current graph.
     *
     * @return A graph object that contains a node linking to the current graph
     * or undefined if the graph is not linked to anything.
     */
    getParentGraph() {
        let parentGraph;
        Array.from(this.baklavaView.editor.graphs)
            .filter((graph) => graph.id !== this.baklavaView.displayedGraph.id)
            .forEach((graph) => {
                graph.nodes.forEach((node) => {
                    if (node.subgraph?.id === this.baklavaView.displayedGraph.id) {
                        parentGraph = graph;
                    }
                });
            });
        return parentGraph;
    }

    /**
     * Switches the editor state to a given graph object.
     * @param graph A graph object that the layout should be switched to.
     */
    switchToGraph(graph) {
        this.baklavaView.editor.switchToGraph(graph);
    }

    /**
     * Centers the editor view and resets zoom level
     */
    centerZoom() {
        this.baklavaView.editor.centerZoom();
    }

    /**
     * Updates name of currently displayed graph
     */
    updateSubgraphName(name) {
        this.editor.updateCurrentSubgraphName(name);
    }

    get notifyWhenChanged() {
        return this.updatedMetadata.notifyWhenChanged ??
            this.specification.currentSpecification?.metadata?.notifyWhenChanged ??
            this.defaultMetadata.notifyWhenChanged;
    }

    /**
     * Creates array of used node names.
     * @param {any} dataflow - Dataflow with graphs containing target nodes.
     * @returns {string[]} Used node names.
     */
    static getUsedNames(graphs) {
        return graphs
            ?.map((graph) => graph.nodes.map(this.getNodeName))
            .flat() ?? [];
    }

    /**
     * Removes unused nodes.
     *
     * @param {any[]} nodes - Initial nodes.
     * @param {string[]} nodeNames - Used node names.
     * @returns {any[]} Minified nodes.
     */
    static minifySpecificationNodes(nodes, nodeNames) {
        const nameToNodeMapping =
            Object.fromEntries(nodes.map((node) => [this.getNodeName(node), node]));
        const nameToNode = (name) => nameToNodeMapping[name];

        const resolvedNames = [];
        const resolve = (node) => {
            if (node === undefined) return;
            const name = this.getNodeName(node);
            if (resolvedNames.includes(name)) return;
            resolvedNames.push(name);
            node.extends?.map(nameToNode).forEach(resolve);
        };
        nodeNames.map(nameToNode).forEach(resolve);

        nodes = resolvedNames
            .map(nameToNode)
            .filter((node) => node !== undefined);

        return nodes;
    }

    isInSubgraph() {
        return this.editor.isInSubgraph();
    }
}
