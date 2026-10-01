/*
 * Copyright (c) 2022-2023 Antmicro <www.antmicro.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 */

import cytoscape from 'cytoscape';
import cola from 'cytoscape-cola';
import dagre from 'cytoscape-dagre';

import BaseLayoutEngine from './baseEngine.js';

cytoscape.use(dagre);
cytoscape.use(cola);

export default class CytoscapeLayoutEngine extends BaseLayoutEngine {
    // The only cytoscape algorithm not defined here are 'null' (every node to
    // (0, 0)) and 'preset' (every node to user defined position)
    availableAlgorithms = [
        'cola',
        'dagre-network-simplex',
        'dagre-tight-tree',
        'dagre-longest-path',
        'random',
        'grid',
        'circle',
        'concentric',
        'breadthfirst',
        'cose',
    ];

    async calculate(graph) {
        const cytoscapeGraph = cytoscape({
            elements: {
                nodes: graph.nodes.map((node) => ({
                    data: { id: node.id, width: node.width, height: node.height },
                })),
                edges: graph.connections.map((connection) => ({
                    data: { id: connection.id, source: connection.from, target: connection.to },
                })),
            },
            style: [
                {
                    selector: 'node',
                    style: {
                        shape: 'rectangle',
                        width: 'data(width)',
                        height: 'data(height)',
                    },
                },
            ],
            styleEnabled: true,
        });

        const options = { name: this.activeAlgorithm };
        /* eslint-disable no-unused-vars */
        switch (this.activeAlgorithm) {
            case 'random':
                options.boundingBox = {
                    x1: 0,
                    y1: 0,
                    w: 2000,
                    h: 2000,
                };
                break;
            case 'grid':
                options.avoidOverlapPadding = 150;
                break;
            case 'cose':
                options.nodeOverlap = 1000;
                options.idealEdgeLength = (edge) => 300;
                break;
            case 'cola':
                options.nodeSpacing = (node) => 150;
                break;
            case 'dagre-network-simplex':
            case 'dagre-tight-tree':
            case 'dagre-longest-path': {
                const [name, ...ranker] = this.activeAlgorithm.split('-');
                options.nodeSep = 50;
                options.rankSep = 100;
                options.ranker = ranker.join('-');
                options.name = name;
                break;
            }
            default:
                break;
        }
        /* eslint-enable no-unused-vars */

        const layout = cytoscapeGraph.layout(options);
        layout.run();
        if (['cose', 'cola'].includes(this.activeAlgorithm)) {
            // wait until asynchronous algorithm finish calculations
            await layout.promiseOn('layoutstop');
        }
        return {
            ...graph,
            nodes: cytoscapeGraph.nodes().map((node) => ({
                id: node.id(),
                // node.position defines the center of node but graph representation
                // required coordinates of top left corner
                position: {
                    x: node.position().x - node.width() / 2,
                    y: node.position().y - node.height() / 2,
                },
            })),
        };
    }
}
