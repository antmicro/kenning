/*
 * Copyright (c) 2022-2024 Antmicro <www.antmicro.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Module handles the autolayout calculations.
 * Defines intermediate graph representation, containing:
 * - nodes - list of nodes with properties:
 *   - id - unique id
 *   - width, height - node dimensions
 *   - position - position set after the calculations with the layout engine is done.
 * - connections - list of connections in a graph, each defines property:
 *   - id - unique id
 *   - from - id of the starting node
 *   - to - id of node at the end point
 *
 * Layout Manager contains list of registered layout engines. Layout engines are
 * requested by user in a specification. If no layout engine is set, the default option
 * (setting position to (0, 0)) is used
 */

import CytoscapeLayoutEngine from './layoutEngines/cytoscapeEngine.js';
import NoLayoutAlgorithm from './layoutEngines/noLayoutEngine.js';
import D3Engine from './layoutEngines/d3Engine.js';

/* eslint-disable no-param-reassign */
function dataflowToGraph(dataflow) {
    const interfaceToNodeId = new Map();
    dataflow.nodes.forEach((node) => {
        node.interfaces.forEach((intf) => interfaceToNodeId.set(intf.id, node.id));
    });

    const nodes = dataflow.nodes
        .filter((node) => node.position === undefined)
        .map((node) => ({
            id: node.id,
        }));
    const connections = dataflow.connections
        .filter(
            (connection) =>
                nodes.filter((nodeState) => nodeState.id === interfaceToNodeId.get(connection.from))
                    .length > 0 &&
                nodes.filter((nodeState) => nodeState.id === interfaceToNodeId.get(connection.to))
                    .length > 0,
        )
        .map((connection) => ({
            id: connection.id,
            from: interfaceToNodeId.get(connection.from),
            to: interfaceToNodeId.get(connection.to),
        }));
    return { nodes, connections };
}

function graphToDataflow(graph, dataflow) {
    const idToPosition = new Map();
    graph.nodes.forEach((node) => idToPosition.set(node.id, node.position));
    dataflow.nodes = dataflow.nodes.map((node) => ({
        ...node,
        position: idToPosition.has(node.id) ? idToPosition.get(node.id) : node.position,
    }));
    return dataflow;
}

export default class LayoutManager {
    layoutEngine = undefined;

    usedAlgorithm = undefined;

    graph = undefined;

    // Default option when no layout algorithm is specified
    // Currently it is possible to register it, when more layout algorithms
    // are added it should be 1) automatically registered 2) not possible to
    // choose in available algorithms
    availableEngines = {
        NoLayout: new NoLayoutAlgorithm(),
        CytoscapeEngine: new CytoscapeLayoutEngine(),
        D3Engine: new D3Engine(),
    };

    constructor() {
        this.useAlgorithm('NoLayout');
    }

    useAlgorithm(algorithm) {
        const [engineName, algorithmName] = algorithm.split(' - ');
        const layoutEngine = this.availableEngines[engineName];
        if (layoutEngine === undefined) {
            throw new Error(`Could not parse the ${algorithm} autolayout algorithm`);
        }
        this.layoutEngine = layoutEngine;
        if (algorithmName !== undefined) {
            this.layoutEngine.chooseAlgorithm(algorithmName);
        }
        this.usedAlgorithm = algorithm;
    }

    getAvailableAlgorithms() {
        return Object.entries(this.availableEngines)
            .map(([engineName, engine]) => {
                if (engineName !== 'NoLayout') {
                    return engine.availableAlgorithms.map(
                        (algorithm) => `${engineName} - ${algorithm}`,
                    );
                }
                return engineName;
            })
            .flat();
    }

    registerGraph(dataflow) {
        this.graph = dataflowToGraph(dataflow);
    }

    async computeLayout(dataflow) {
        this.updateDimensions();
        const layout = await this.runEngine(this.graph);
        return graphToDataflow(layout, dataflow);
    }

    async runEngine(graph) {
        return this.layoutEngine.calculate(graph);
    }

    updateDimensions() {
        this.graph.nodes = this.graph.nodes.map((node) => {
            const HTMLelement = document.getElementById(node.id);
            return {
                ...node,
                width: HTMLelement.offsetWidth,
                height: HTMLelement.offsetHeight,
            };
        });
    }
}
