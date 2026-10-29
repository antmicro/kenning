/*
 * Copyright (c) 2022-2024 Antmicro <www.antmicro.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * This module contains all possible JSON-RPC requests that frontend can
 * receive and process.
 *
 * Each exported function is automatically registered as JSON-RPC method
 * when JSON-RPC server is created in fetchRequest.
 * Function name have to match with specification (resources/api_specification).
 */

import { useViewModel } from '@baklavajs/renderer-vue';
import { v4 as uuidv4 } from 'uuid';
import runInfo from './runInformation';
import EditorManager from '../EditorManager';
import NotificationHandler from '../notifications';
import { MAIN_TERMINAL, terminalStore } from '../stores';
import { saveSpecificationConfiguration } from '../../components/saveConfiguration.ts';

// eslint-disable-next-line import/no-cycle
import getExternalApplicationManager from './ExternalApplicationManager';
import { checkTerminalExistence, TerminalView } from './utils';
import { prepareSubgraphInstance } from '../../custom/CustomGraphNode';

/* eslint-disable import/prefer-default-export */
/* eslint-disable camelcase */
/* eslint-disable @typescript-eslint/no-explicit-any */

const editorManager = EditorManager.getEditorManagerInstance();

/**
 * Finds graph of id `graph_id`.
 * An error with appropriate message is thrown if any error occurs.
 *
 * @param graph_id Id of a graph to find
 * @returns found graph
 */
function getGraph(graph_id: string): any {
    const { viewModel } = useViewModel();
    const graph = [...viewModel.value.editor.graphs].find((g) => g.id === graph_id);
    if (graph === undefined) {
        throw new Error(`Graph with id '${graph_id}' does not exist.`);
    }
    return graph;
}

/**
 * Finds node of id `node_id` in graph of id `graph_id`.
 * An error with appropriate message is thrown if any error occurs.
 *
 * @param graph_id Id of a graph to find
 * @param node_id Id of a node to find
 * @returns found node
 */
function getNode(graph_id: string, node_id: string): any {
    const graph = getGraph(graph_id);

    const node = graph.nodes.find((n: any) => n.id === node_id);
    if (node === undefined) {
        throw new Error(`Node with id '${node_id}' does not exist.`);
    }
    return node;
}

/**
 * Finds property of id `id` or name `name` in node.
 * One of those values has to be defined.
 * An error with appropriate message is thrown if any error occurs.
 *
 * @param node Node instance that is searched
 * @param id id of the property
 * @param name name of the property
 * @returns found property
 */
function getProperty(node: any, id?: string, name?: string) {
    let prop;

    if (id !== undefined) {
        prop = Object.values(
            node.inputs,
        ).find((p : any) => p.id === id) as any;

        // If not property found or it is not a property, but an interface
        if (prop === undefined || prop.side !== undefined) {
            throw new Error(`Property with id '${id}' does not exist.`);
        }
    } else {
        prop = Object.values(
            node.inputs,
        ).find((p : any) => p.name === name) as any;

        // If not property found or it is not a property, but an interface
        if (prop === undefined || prop.side !== undefined) {
            throw new Error(`Property with name '${name}' does not exist.`);
        }
    }

    return prop;
}

/**
 * Finds interface of id `id` or name `name` in node.
 * One of those values has to be defined.
 * An error with appropriate message is thrown if any error occurs.
 *
 * @param node Node instance that is searched
 * @param id id of the interface
 * @param name name of the interface
 * @returns found interface
 */
function getInterface(node: any, id?: string, name?: string) {
    let intf;

    if (id !== undefined) {
        intf = [...Object.values(node.inputs), ...Object.values(node.outputs)]
            .find((p : any) => p.id === id) as any;

        // If not interface found or it is not an interface, but a property
        if (intf === undefined || intf.side === undefined) {
            throw new Error(`Interface with id '${id}' does not exist.`);
        }
    } else {
        intf = [...Object.values(node.inputs), ...Object.values(node.outputs)]
            .find((p : any) => p.name === name) as any;

        // If not interface found or it is not an interface, but a property
        if (intf === undefined || intf.side === undefined) {
            throw new Error(`Interface with name '${name}' does not exist.`);
        }
    }

    return intf;
}

/**
 * Finds connection between interfaces with id `from` and `to` in graph of id `graph_id`.
 * An error with appropriate message is thrown if any error occurs.
 *
 * @param graph_id Id of a graph to find
 * @param from Id of a first interface
 * @param to Id of a second interface
 * @returns found node
 */
function getConnection(graph_id: string, from: string, to: string): any {
    const graph = getGraph(graph_id);

    const connection = graph.connections.find((c: any) => c.from.id === from && c.to.id === to);
    if (connection === undefined) {
        throw new Error(`Connection from ${from} to ${to} does not exist.`);
    }
    return connection;
}

/**
 * @returns status of the frontend
 */
export function frontend_status_get() {
    return { status: 'ready' };
}

/**
 * @returns currently used specification
 */
export function frontend_specification_get(params: { graph: boolean, minify: boolean }) {
    const saveConfiguration = saveSpecificationConfiguration;
    saveConfiguration.graph = params ? params.graph : false;
    saveConfiguration.minify = params ? params.minify : false;
    return { specification: saveConfiguration.getSave() };
}

/**
 * @returns currently used dataflow
 */
export function graph_get() {
    return {
        dataflow: editorManager.saveDataflow(),
    };
}

type ModifyPropertiesParamsType = {
    graph_id: string,
    node_id: string,
    properties: {
        id?: string,
        name?: string,
        new_value: any,
    }[],
    interfaces: {
        id?: string,
        name?: string,
        externalName: any,
    }[]
};

/**
 * Updates values of properties specified in `params`.
 * An error with appropriate message is thrown if any error occurs.
 */
export function properties_change(
    params: ModifyPropertiesParamsType,
) {
    const node = getNode(params.graph_id, params.node_id);

    // First iteration to validate that every property exists
    // eslint-disable-next-line no-restricted-syntax
    for (const property of params.properties) {
        getProperty(node, property.id, property.name);
    }

    // Second iteration to actually alter the values
    // eslint-disable-next-line no-restricted-syntax
    for (const property of params.properties) {
        const prop = getProperty(node, property.id, property.name);
        prop.value = property.new_value;
    }
}

export function interfaces_change(
    params: ModifyPropertiesParamsType,
) {
    const node = getNode(params.graph_id, params.node_id);
    // First iteration to validate that every property exists
    // eslint-disable-next-line no-restricted-syntax
    for (const intf of params.interfaces) {
        getInterface(node, intf.id, intf.name);
    }

    // Second iteration to actually alter the values
    // eslint-disable-next-line no-restricted-syntax
    for (const intfInfo of params.interfaces) {
        const intf = getInterface(node, intfInfo.id, intfInfo.name);
        intf.externalName = intfInfo.externalName;
    }
    const graph = getGraph(params.graph_id);
    // eslint-disable-next-line prefer-destructuring
    const graphNode = graph.graphNode;
    graphNode.updateExposedInterfaces(undefined, undefined, true);
}

type ModifyPositionParamsType = {
    graph_id: string,
    node_id: string,
    position: {
        x: number,
        y: number,
    }
};

/**
 * Updates values of properties specified in `params`.
 * An error with appropriate message is thrown if any error occurs.
 */
export function position_change(
    params: ModifyPositionParamsType,
) {
    const node = getNode(params.graph_id, params.node_id);

    node.position.x = params.position.x;
    node.position.y = params.position.y;
}

type ModifyTitleParamsType = {
    graph_id: string,
    node_id: string,
    name: string,
};

/**
 * Changes node title.
 */
export function name_change(
    params: ModifyTitleParamsType,
) {
    const node = getNode(params.graph_id, params.node_id);

    node.title = params.name;
}

type ModifyNodesParamsType = {
    graph_id: string,
    nodes: {
        added: any[],
        deleted: string[],
    },
    remove_with_connections: boolean,
}

/**
 * Creates and deletes nodes based on received `params`.
 */
export async function nodes_change(params: ModifyNodesParamsType) {
    const { viewModel } = useViewModel();
    const graph = getGraph(params.graph_id);
    params.nodes.added.forEach((n) => {
        const info = viewModel.value.editor.nodeTypes.get(n.name);
        if (!info) {
            throw new Error(`Node type not found for name ${n.name}`);
        } else {
            const node = new info.type(); // eslint-disable-line new-cap
            node.id = n.id;
            if (n.color) {
                const editor = viewModel.value.editor as any;
                editor.setNodeColor(n.id, n.color);
            }
            graph.addNode(node);
            const errors = node.load(n);
            if (Array.isArray(errors) && errors.length) throw new Error(errors.join('\n'));
        }
    });

    params.nodes.deleted.forEach((n) => {
        const node = getNode(params.graph_id, n);
        if (params.remove_with_connections ?? true) {
            graph.removeNode(node);
        } else {
            graph.removeNodeOnly(node);
        }
    });
}

type ModifyConnectionsParamsType = {
    graph_id: string,
    connections: {
        added: any[],
        deleted: {from: string, to: string}[],
    },
}

/**
 * Creates and deletes connections based on received `params`.
 */
export async function connections_change(params: ModifyConnectionsParamsType) {
    const graph = getGraph(params.graph_id);

    params.connections.added.forEach((c) => {
        const fromIntf = graph.findNodeInterface(c.from);
        if (!fromIntf) throw new Error(`Interface with id ${c.from} does not exist`);
        const toIntf = graph.findNodeInterface(c.to);
        if (!toIntf) throw new Error(`Interface with id ${c.to} does not exist`);
        const connection = graph.addConnection(fromIntf, toIntf);
        if (!connection) throw new Error(`Connection from ${c.from} to ${c.to} cannot be created`);
    });
    params.connections.deleted.forEach((c) => {
        graph.removeConnection(getConnection(params.graph_id, c.from, c.to));
    });
}

/**
 * Loads received dataflow.
 */
export async function graph_change(params: { dataflow: any, loadingScreen: boolean }) {
    const externalApplicationManager = getExternalApplicationManager();
    await externalApplicationManager.conditionalLoadingScreen(
        params.loadingScreen,
        async () => externalApplicationManager.updateDataflow(params.dataflow),
    );
}

type GetPropertiesParamsType = {
    graph_id: string,
    node_id: string,
    properties?: {
        id?: string,
        name?: string,
    }[],
};

type GetPropertiesReturnType = {
    id: string,
    name: string,
    value: any,
}[];

/**
 * Returns an array of properties specified in `params`.
 * An error with appropriate message is thrown if any error occurs.
 */
export async function properties_get(
    params: GetPropertiesParamsType,
): Promise<GetPropertiesReturnType> {
    const node = getNode(params.graph_id, params.node_id);

    const properties: GetPropertiesReturnType = [];

    if (params.properties === undefined) {
        // eslint-disable-next-line no-restricted-syntax
        for (const property of Object.values((node as any).inputs) as any) {
            if (property.side === undefined) {
                properties.push({
                    id: property.id,
                    name: property.name,
                    value: property.value,
                });
            }
        }

        return properties;
    }

    // eslint-disable-next-line no-restricted-syntax
    for (const property of params.properties) {
        const prop = getProperty(node, property.id, property.name);

        properties.push({
            id: prop.id,
            name: prop.name,
            value: prop.value,
        });
    }

    return properties;
}

type GetNodeParamsType = {
    graph_id: string,
    node_id: string,
};

/**
 * Returns a serialized node state specified in `params`.
 * An error with appropriate message is thrown if any error occurs.
 */
export async function node_get(
    params: GetNodeParamsType,
): Promise<{ node: any }> {
    const node = getNode(params.graph_id, params.node_id);

    return {
        node: node.save(),
    };
}

/**
 * Sets width of progress bar.
 * If there is not run in progress, throws error.
 */
export function progress_change(params: {progress: number, method: string}) {
    const procedureInfo = runInfo.get(params.method);
    if (!procedureInfo.inProgress) {
        throw new Error('No run in progress');
    }
    if (!procedureInfo.progressBar) throw new Error(`Progress bar does not exist for method: ${params.method}`);
    if (params.progress === -1) {
        procedureInfo.progressBar.classList.add('animate');
        return;
    }
    if (params.progress > 100 || params.progress < 0) throw new Error(`Progress has to be in [0, 100] or -1. Received: ${params.progress}`);
    procedureInfo.progressBar.style.width = `${params.progress}%`;
    procedureInfo.progressBar.classList.remove('animate');
}

/**
 * Updates the editor's metadata.
 */
export function metadata_change(params: { metadata: any }) {
    editorManager.updateMetadata(params.metadata, true);
}

/**
 * Triggers action centering the editor.
 */
export function viewport_center() {
    editorManager.baklavaView.editor.centerZoom();
}

type TerminalAdd = {
    name: string,
    readonly: boolean,
};

/**
 * Creates new terminal instance
 */
export function terminal_add(params: TerminalAdd) {
    const status = terminalStore.createTerminalInstance(params.name, params.readonly);
    if (status === false) {
        throw new Error(`Terminal instance of name '${params.name}' already exists`);
    }
}

export function terminal_remove(params: { name: string }) {
    checkTerminalExistence(params.name, true);
    terminalStore.remove(params.name);
}

type TerminalWrite = {
    name: string,
    message: string
};

/**
 * Writes a single message to a chosen terminal
 */
export function terminal_write(params: TerminalWrite) {
    if (!(params.name in terminalStore.logs)) {
        terminalStore.createTerminalInstance(params.name);
    }
    terminalStore.add(params.message, params.name);
}

export function terminal_clear(params: { name: string | undefined }) {
    const name = params?.name ?? MAIN_TERMINAL;
    checkTerminalExistence(name, true);
    terminalStore.clear(name);
}

export function terminal_hide() {
    terminalStore.manager?.hide();
}

export function terminal_show(params: { name: string | undefined }) {
    const name = params?.name ?? MAIN_TERMINAL;
    if (name !== undefined) checkTerminalExistence(name, true);
    terminalStore.manager?.show(name);
}

export function terminal_view(params: TerminalView) {
    params.names?.forEach(((name) => checkTerminalExistence(name, true)));
    terminalStore.manager?.view(params);
}

export function terminal_get_instances() {
    return terminalStore.getInstances();
}

type Notification = {
    type: string,
    title: string,
    details: string,
    timestamp?: string
};

export function notification_send(params: Notification) {
    NotificationHandler.terminalLog(params.type, params.title, params.details, params.timestamp);
}

export async function specification_change(
    params: { specification: any, urloverrides: any, loadingScreen: boolean, tryMinify: boolean },
) {
    const externalApplicationManager = getExternalApplicationManager();
    await externalApplicationManager.conditionalLoadingScreen(
        params.loadingScreen,
        async () => externalApplicationManager.updateSpecification(params.specification, params),
    );
}

export async function specification_graph_change(
    params: {
        specification: any,
        dataflow: any,
        urloverrides: any,
        loadingScreen: boolean,
        tryMinify: boolean,
    },
) {
    const externalApplicationManager = getExternalApplicationManager();
    const tryMinify = params.tryMinify && params.dataflow ? params.dataflow : params.tryMinify;
    await externalApplicationManager.conditionalLoadingScreen(params.loadingScreen, async () => {
        const error = await externalApplicationManager.updateSpecification(
            params.specification, { ...params, tryMinify },
        );
        if (!error) {
            await externalApplicationManager.updateDataflow(params.dataflow);
        }
    });
}

export async function specification_preprocess(
    params: {
        specification: any,
        dataflow: any,
        urloverrides: any,
        loadingScreen: boolean,
        tryMinify: boolean,
    },
): Promise<{ specification: any } | undefined> {
    const externalApplicationManager = getExternalApplicationManager();
    const tryMinify = params.tryMinify && params.dataflow ? params.dataflow : params.tryMinify;
    const resolveSpecification = await externalApplicationManager.preprocessSpecification(
        params.specification, { ...params, tryMinify },
    );
    return resolveSpecification ? { specification: resolveSpecification } : undefined;
}

export async function specification_expand({ specification }: { specification: any }) {
    const assertNoErrors = (errors: string[]) => {
        if (errors.length > 0) {
            throw new Error(errors.join('\n'));
        }
    };

    // Validate
    const validationErrors = EditorManager.validateSpecification(specification);
    assertNoErrors(validationErrors);

    // Collect specification data
    const specGraphs = specification.graphs ?? [];
    const specGraphMap = new Map(
        specGraphs.map((graph: any) => [graph.id, structuredClone(graph)]),
    );
    const subgraphIds = specification.nodes
        .map((node: any) => node.subgraphId)
        .filter(Boolean);

    // Initialize loop data
    const isNotSubgraph = (graph: any) => !subgraphIds.includes(graph.id);
    const finalGraphs = specGraphs.filter(isNotSubgraph);
    const processingQueue = subgraphIds.map((id: string) => ({
        subgraphId: id,
        instanceId: id,
    }));

    while (processingQueue.length) {
        const { subgraphId, instanceId } = processingQueue.shift();

        // Prepare graph
        const graph: any = specGraphMap.get(subgraphId);
        const { state, errors } = prepareSubgraphInstance(graph, { mode: 'spec' });
        assertNoErrors(errors);

        // Add nested graphs to the queue
        state.nodes
            .filter((node: any) => node.subgraph && subgraphIds.includes(node.subgraph))
            .forEach((node: any) => {
                const nestedSubgraphId = node.subgraph;
                const nestedInstanceId = uuidv4();
                node.subgraph = nestedInstanceId; // eslint-disable-line no-param-reassign
                processingQueue.push({
                    subgraphId: nestedSubgraphId,
                    instanceId: nestedInstanceId,
                });
            });

        // Save graph
        delete state.graphLoadingState;
        finalGraphs.push({ ...state, id: instanceId });
    }

    return {
        specification: {
            ...specification,
            graphs: finalGraphs,
        },
    };
}

type ModifyNodesHighlightType = {
    graph_id: string,
    nodes: {
        selected: string[],
        unselected: string[],
    }
}

/**
 * Highlights nodes provided in params
 */
export async function nodes_highlight(
    params: ModifyNodesHighlightType,
) {
    const graph = getGraph(params.graph_id);

    graph.selectedNodes = graph.selectedNodes.filter((node: any) =>
        !params.nodes.unselected.includes(node.id),
    );

    params.nodes.selected.forEach((node_id: string) => {
        const node = getNode(params.graph_id, node_id);
        if (!graph.selectedNodes.includes(node)) {
            graph.selectedNodes.push(node);
        }
    });
}

export function register_external_frontend() {
    throw new Error('Frontend registering available only through "postMessage" interface.');
}

export function return_from_subgraph() {
    const isInSubgraph = editorManager.isInSubgraph();

    if (isInSubgraph) {
        editorManager.returnFromSubgraph();
    }

    return { has_returned: isInSubgraph };
}
