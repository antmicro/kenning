/*
 * Copyright (c) 2022-2024 Antmicro <www.antmicro.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 */

import { NodeInterface, Graph } from '@baklavajs/core';
import { v4 as uuidv4 } from 'uuid';
import { parseInterfaces, generateProperties } from '../core/interfaceParser.js';
import { updateInterfacePosition } from './CustomNode.js';
import {
    CustomNode,
    newProperty,
    parseProperties,
    createProperties,
    createBaklavaInterfaces,
    updateSubgraphInterfaces,
    updateSubgraphProperties,
} from '../core/NodeFactory.js';
import { ir } from '../core/interfaceRegistry.ts';

import globalProperties from '../globalProperties.ts';

import { terminalStore } from '../core/stores.js';

/**
  * Function used to update properties of the graph node based on its specification.
  *
  * @param {Object} graphNode graph node object
  *
  * @returns properties ready to be loaded
  */
function prepareProperties(prop) {
    const properties = prop ?? [];

    const parsedProperties = parseProperties(properties);
    if (Array.isArray(parsedProperties) && parsedProperties.length) {
        return parsedProperties.map((error) => `Graph node invalid. ${error}`);
    }
    return createProperties(parsedProperties);
}

/**
  * Function used to update interfaces of the graph node based on its specification.
  * This includes only interfaces of the node, not external interfaces of the subgraph.
  *
  * @param {Object} graphNode graph node object
  *
  * @returns interfaces ready to be loaded
  */
function prepareInterfaces(intfs) {
    const interfaces = intfs ?? [];

    const parsedInterfaces = parseInterfaces(interfaces, [], []);
    if (Array.isArray(parsedInterfaces) && parsedInterfaces.length) {
        return parsedInterfaces.map((error) => `Graph node invalid. ${error}`);
    }
    return createBaklavaInterfaces(parsedInterfaces);
}

/**
 * The function uses its internal static template to create a new graph state
 * based on the template. It creates a new graph node ID and maps all interfaces
 * to new IDs. The function is used to create a new graph state that can be loaded
 * by the graph node.
 *
 * @param {object} template - Graph template
 * @param {object} opts
 * @param {object|undefined} [opts.graphLoadingState] - Information about loaded graphs.
 * @param {string|undefined} [opts.newSubgraphNodeId] - New ID of the subgraph node.
 * @param {'resolved'|'spec'} opts.mode - New ID of the subgraph node.
 *
 * @returns {any} graph state ready to be loaded
 */
export function prepareSubgraphInstance(
    template,
    { graphLoadingState, newSubgraphNodeId, mode } = {},
) {
    // eslint-disable-next-line no-param-reassign
    mode ??= 'resolved';

    const errors = [];

    // Ensure subgraph nodes have IDs to distinguish instances in nested layers
    const predicate = (node) => node.subgraph !== undefined && node.id === undefined;
    if (template.nodes.some(predicate)) errors.push('Node defining a nested graph in a subgraph template should have their own ID');
    if (errors.length) return { errors };

    const isRoot = graphLoadingState === undefined;
    // eslint-disable-next-line no-param-reassign
    graphLoadingState ??= {
        // New subgraph node ID -> Specification subgraph node ID
        newToSpecNodeIds: new Map(),
        // Subgraph node ID -> (Exposed Interface name -> New Interface ID)
        newInterfaceIds: new Map(),
        // New connection ID -> Specification connection ID
        newToSpecConnIds: new Map(),
        // New interface ID -> Specification interface ID
        newToSpecIntfIds: new Map(),
    };

    /* Nodes */

    const { newToSpecNodeIds } = graphLoadingState;
    const specSubgraphNodeId = newToSpecNodeIds.get(newSubgraphNodeId);

    const createNewNodeId = (node) => {
        const newId = uuidv4();
        if (node.id) newToSpecNodeIds.set(newId, node.id);
        return newId;
    };

    const addNewNodeId = (node) => ({ ...node, id: createNewNodeId(node) });

    /* Interfaces and connections */

    const connections = template.connections.map((c) => {
        const newId = uuidv4();
        if (c.id) graphLoadingState.newToSpecConnIds.set(newId, c.id);
        return { ...c, id: newId };
    });
    const { newInterfaceIds } = graphLoadingState;

    const getOrGenerate = (map, key, factory = uuidv4) => {
        if (!map.has(key)) map.set(key, factory());
        return map.get(key);
    };

    const getOrGenerateInterfaceId = (node, intf) => {
        // Exposed interface - generate and store
        // Note: All interfaces in a subgraph node are treated as exposed,
        // therefore all of them are looked up in the subgraph.
        // This may be subject to change in future
        const nodeId = node.subgraph !== undefined ? node.id : specSubgraphNodeId;
        const idMap = getOrGenerate(newInterfaceIds, nodeId, () => new Map());

        // Regular interface
        const nonExposed = node.subgraph === undefined && intf.externalName === undefined;

        const interfaceName = `${intf.externalName ?? intf.name}_${intf.side ?? 'left'}`;
        // Exposed interface is not indicated in the nested subgraph node
        const exposedNotInSubgraphNode = !isRoot
            && intf.externalName !== undefined
            && (specSubgraphNodeId === undefined || !idMap.has(interfaceName));

        return (nonExposed || exposedNotInSubgraphNode)
            ? uuidv4()
            : getOrGenerate(idMap, interfaceName);
    };

    const createNewInterfaceId = (oldId, node) => {
        const interfaces = mode === 'resolved'
            ? Object.values(node.inputs).concat(Object.values(node.outputs))
            : node.interfaces;

        const intf = interfaces.find((i) => i.id === oldId);

        const newId = getOrGenerateInterfaceId(node, intf);
        if (intf.id) graphLoadingState.newToSpecIntfIds.set(newId, intf.id);

        // Side-effect: modify connections to have new interface IDs
        /* eslint-disable no-param-reassign */
        connections.filter((c) => c.from === oldId).forEach((c) => { c.from = newId; });
        connections.filter((c) => c.to === oldId).forEach((c) => { c.to = newId; });
        /* eslint-enable no-param-reassign */

        return newId;
    };

    const addNewId = (intf, node) => ({ ...intf, id: createNewInterfaceId(intf.id, node) });

    const mapNodeInterfaces = (interfaces, node) => Object.entries(interfaces)
        .reduce((acc, [name, intf]) => ({ ...acc, [name]: addNewId(intf, node) }), {});

    const addNewInterfaceIds = mode === 'resolved'
        ? (node) => ({
            ...node,
            inputs: mapNodeInterfaces(node.inputs, node),
            outputs: mapNodeInterfaces(node.outputs, node),
        }) : (node) => ({
            ...node,
            interfaces: node.interfaces.map((intf) => addNewId(intf, node)),
        });

    // Apply changes

    return {
        state: {
            id: uuidv4(),
            name: template.name,
            nodes: template.nodes.map(addNewInterfaceIds).map(addNewNodeId),
            connections,
            graphLoadingState,
        },
        errors,
    };
}

export default function CreateCustomGraphNodeType(template, graphNode, editorManager) {
    const intfsInherited = editorManager.findInheritedInterfaces(graphNode.name).filter((intf) =>
        !graphNode.interfaces?.find((i) => i.name === intf.name))
        ?.map((intf) => ({ ...intf, inherited: true }));
    const propsInherited = editorManager.findInheritedProperties(graphNode.name).filter((prop) =>
        !graphNode.properties?.find((p) => p.name === prop.name))
        ?.map((prop) => ({ ...prop, inherited: true }));

    const [nodeInputs, newNodeOutputs] = prepareInterfaces(
        [...graphNode.interfaces ?? [], ...intfsInherited],
    );
    const dynamicProperties = generateProperties(graphNode.interfaces ?? []);

    const properties = prepareProperties(
        [...graphNode.properties ?? [], ...propsInherited, ...dynamicProperties.value ?? []],
    );
    const newNodeInputs = {
        ...nodeInputs,
        ...properties,
    };

    return class CustomGraphNodeType extends CustomNode {
        type = graphNode.name;

        title = graphNode.name;

        template = template;

        nodeInterfaces = { ...newNodeInputs, ...newNodeOutputs };

        constructor() {
            super(
                graphNode.name,
                graphNode.layer,
                newNodeInputs,
                newNodeOutputs,
                graphNode.twoColumn ?? editorManager.baklavaView.twoColumn ?? false,
                graphNode.description ?? '',
                graphNode.extends ?? [],
                graphNode.extending ?? [],
                graphNode.siblings ?? [],
                graphNode.width ?? 0,
                graphNode.relatedGraphs,
            );
        }

        save() {
            const state = super.save();
            state.subgraph = this.subgraph?.id;
            state.graphState = this.subgraph?.save();
            return state;
        }

        /* eslint-disable no-param-reassign */
        load(state, privatize = false) {
            this.hooks.beforeLoad.execute(state);

            const out = parseInterfaces(state.interfaces ?? [], [], []);
            if (Array.isArray(out) && out.length) {
                return out;
            }
            let { inputs, outputs } = out;
            inputs = Object.values(inputs);
            outputs = Object.values(outputs);

            if (state.graphState !== undefined) {
                delete state.graphState.interfaces;
                delete state.subgraph;

                // Loading the subgraph of the graph node first, before creating
                // interfaces based on the nodes in the subgraph. Thanks to that
                // the originally exposed interfaces (coming from regular nodes)
                // are found first.
                const errors = this.subgraph.load(state.graphState);
                if (errors.length) {
                    return errors;
                }
            }

            this.updateExposedInterfaces(inputs, outputs, privatize);
            // Default position should be undefined instead of (0, 0) so that it can be set
            // by autolayout
            this.position = state.position;
            this.title = state.instanceName ?? '';

            const errors = super.load(state);

            this.events.loaded.emit(this);
            return errors;
        }

        /**
         * Function used to update exposed interfaces and properties of the graph node based on the
         * nodes inside the graph, their interfaces and their external names.
         *
         * @param {Array} inputs inputs of the graph node. If not provided, the function
         * will use the current inputs of the graph node.
         * @param {Array} outputs outputs of the graph node. If not provided, the function
         * will use the current outputs of the graph node.
         * @param {boolean} privatize whether to check for privatized (removed) interfaces
         */
        updateExposedInterfaces(inputs = undefined, outputs = undefined, privatize = false) {
            // when subgraph is detached remove all exposed interfaces
            if (this.subgraph === undefined) {
                const currentInterfaces = { ...this.inputs, ...this.outputs };

                const defaultInterfaces = [
                    ...Object.entries(newNodeInputs),
                    ...Object.entries(newNodeOutputs),
                ];
                this.privatizeInterfaces(defaultInterfaces, currentInterfaces);
                return;
            }

            // Update interfaces based on subgraph interfaces and their external names
            const evaluatedIntf = updateSubgraphInterfaces(
                this.subgraph.nodes,
                inputs ?? Object.values(this.inputs),
                outputs ?? Object.values(this.outputs),
            );
            if (Array.isArray(evaluatedIntf) && evaluatedIntf.length) {
                throw new Error(
                    `Internal error occurred while exposing an interface.\n` +
                    `Reason: ${evaluatedIntf.join('. ')}`,
                );
            }
            const evaluatedProp = updateSubgraphProperties(
                this.subgraph.nodes,
                inputs ?? Object.values(this.inputs),
            );

            // After resolving exposed interfaces, the graph node is updated accordingly.
            this.updateGraphNodeInterfaces(evaluatedIntf.inputs, evaluatedIntf.outputs, privatize);
            this.updateProperties(evaluatedProp);
        }

        /**
         * Function called when the node is created using the nodePalette
         *
         * @param {Object|undefined} graphLoadingState Information about loaded graphs.
         * @param {string|undefined} nodeId New ID of the subgraph node.
         */
        onPlaced(graphLoadingState, nodeId) {
            this.initialize(graphLoadingState, nodeId);
        }

        /**
         * Creates a new graph node instance based on the template, loads its graph state
         * and updates exposed interfaces based on the nodes in the subgraph.
         *
         * @param {Object|undefined} graphLoadingState  Information about loaded graphs.
         * @param {string|undefined} nodeId New ID of the subgraph node.
         */
        initialize(graphLoadingState, nodeId) {
            if (this.subgraph) {
                this.subgraph.destroy();
            }
            const graph = new Graph(this.template.editor);
            const { state, errors } = prepareSubgraphInstance(
                this.template,
                { graphLoadingState, newSubgraphNodeId: nodeId },
            );

            if (!errors.length) {
                errors.push(...graph.load(state, graphLoadingState?.loadAll ?? false));
            }
            // When soft load is enabled ignore it
            if (errors.length && !globalProperties.softLoad) {
                throw new Error(
                    `Internal error occurred while initializing ${graph.type} graph. ` +
                    `Reason: ${errors.join('. ')}`,
                );
            } else if (errors.length) {
                terminalStore.addParsed(`Internal error occurred while initializing ${graph.type} graph.`,
                    `Reason: ${errors.join('. ')}`);
            }

            graph.template = this.template;
            this.subgraph = graph;
            graph.graphNode = this;

            this.updateExposedInterfaces([], []);

            this._title = this.template.name; // eslint-disable-line no-underscore-dangle
            this.events.update.emit(null);
        }

        /**
         * Function used to check which interfaces had been privatized from within
         * the subgraph and need to be removed as a result.
         *
         * @param {Array} newInterfaces interfaces to be added to the graph node
         * @param {Array} currentInterfaces interfaces already in the graph node
         */
        privatizeInterfaces(newInterfaces, currentInterfaces) {
            Object.entries(currentInterfaces).forEach(([nodeKey, nodeIntf]) => {
                // If current interface cannot be found in `newInterfaces`, it means that
                // it was removed.
                if (newInterfaces.find((intf) => intf.id === nodeIntf.id) === undefined) {
                    // Only remove subgraph interfaces, not properties or node interfaces
                    if (!Object.keys(this.nodeInterfaces).some((key) => key === nodeKey)) {
                        const container = nodeIntf.direction === 'output' ? 'output' : 'input';
                        this.removeInterface(container, nodeKey);
                    }
                }
            });
        }

        /**
         * Function used to update interfaces of the graph node. It makes use of InterfaceRegistry
         * object to create interfaces that share part of the state of the exposed interfaces.
         *
         * @param {Array} newInputs inputs to be added to the graph node
         * @param {Array} newOutputs outputs to be added to the graph node
         * @param {boolean} privatize whether to check for privatized (removed) interfaces
         */
        updateGraphNodeInterfaces(newInputs, newOutputs, privatize = false) {
            const newInterfaces = [...newInputs, ...newOutputs];
            let currentInterfaces = { ...this.inputs, ...this.outputs };

            if (privatize) {
                this.privatizeInterfaces(newInterfaces, currentInterfaces);
            }

            newInterfaces.forEach((nodeIntf) => {
                // If new interface cannot be found in the current interfaces, it means that
                // it has to be created
                const foundIntf = Object.values(currentInterfaces).find(
                    (intf) => intf.id === nodeIntf.id,
                );

                if (foundIntf === undefined) {
                    const ni = new NodeInterface(nodeIntf.name);
                    Object.assign(ni, nodeIntf);
                    ir.pushGraphIdToRegistry(ni.id, this.subgraph.id);
                    ir.createSharedInterface(ni);

                    const container = nodeIntf.direction === 'output' ? 'output' : 'input';
                    this.addInterface(container, `${nodeIntf.direction}_${nodeIntf.name}`, ni);
                } else {
                    const nodeKey = Object.entries(currentInterfaces)
                        .find(([, intf]) => intf.id === nodeIntf.id)?.[0];

                    Object.assign(foundIntf, nodeIntf);

                    const key = `${nodeIntf.direction}_${nodeIntf.name}`;
                    this.updateInterfaceKey(nodeIntf.direction, foundIntf, nodeKey, key);
                }
            });
            currentInterfaces = { ...this.inputs, ...this.outputs };

            Object.values(currentInterfaces).forEach((nodeIntf) => {
                updateInterfacePosition(this, nodeIntf, nodeIntf.side, undefined, false, false);
            });
        }

        /**
         * Function used to update interfaces of the graph node. It makes use of InterfaceRegistry
         * object to create interfaces that share part of the state of the exposed interfaces.
         *
         * @param {Array} newInputs inputs to be added to the graph node
         * @param {Array} newOutputs outputs to be added to the graph node
         * @param {boolean} privatize whether to check for privatized (removed) interfaces
         */
        updateProperties(newProperties, privatize = false) {
            if (privatize) {
                this.privatizeInterfaces(newProperties, this.inputs);
            }

            newProperties.forEach((property) => {
                const foundProperty = Object.values(this.inputs).find(
                    (prop) => prop.id === property.id,
                );

                if (foundProperty === undefined) {
                    const np = newProperty(property);
                    Object.assign(np, property);
                    this.addInput(`property_${property.name}`, np);
                } else {
                    Object.assign(foundProperty, property);
                }
            });
        }
    };
}
