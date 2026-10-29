/*
 * Copyright (c) 2022-2026 Antmicro <www.antmicro.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

import * as d3 from 'd3';
import BaseLayoutEngine from './baseEngine.js';

function* cyclicPairs(arr) {
    for (let i = 0; i < arr.length; i += 1) {
        yield [arr[i], arr[(i + 1) % arr.length]];
    }
}

function computeCentroid(polygon) {
    const signedArea = 0.5 * cyclicPairs(polygon)
        .reduce(
            (acc, pair) => acc + (
                pair[0][0] * pair[1][1] - pair[0][1] * pair[1][0]
            ),
            0);
    const cx = cyclicPairs(polygon)
        .reduce(
            (acc, pair) => acc + (pair[0][0] + pair[1][0]) * (
                pair[0][0] * pair[1][1] - pair[0][1] * pair[1][0]
            ), 0) / (6 * signedArea);
    const cy = cyclicPairs(polygon)
        .reduce(
            (acc, pair) => acc + (pair[0][1] + pair[1][1]) * (
                pair[0][0] * pair[1][1] - pair[0][1] * pair[1][0]
            ), 0) / (6 * signedArea);

    return [cx, cy];
}

function collide(node1, node2, pos1, pos2) {
    const cond = (firstCoord, nodeDim, secondCoord) => firstCoord + nodeDim + 50 < secondCoord;

    const overlapX = !(
        cond(pos1[0], node1.width, pos2[0]) ||
        cond(pos2[0], node2.width, pos1[0])
    );

    const overlapY = !(
        cond(pos1[1], node1.height, pos2[1]) ||
        cond(pos2[1], node2.height, pos1[1])
    );

    return overlapX && overlapY;
}

const average = (arr) => arr.reduce((acc, el) => acc + el, 0) / arr.length;

export default class D3Engine extends BaseLayoutEngine {
    /* eslint-disable class-methods-use-this */
    /* eslint-disable no-loop-func */
    availableAlgorithms = [
        'lloyd',
    ];

    async calculate(graph) {
        if (!graph.nodes.length) return graph;
        // compute boundaries
        const meanWidth = average(graph.nodes.map((node) => node.width));
        const meanHeight = average(graph.nodes.map((node) => node.height));

        let collisions = true;
        const attempts = Math.max(30 - 10 * Math.log10(graph.nodes.length + 1), 5);

        const step = 10;
        let attemptId = 0;
        let positions = [];

        while (collisions && attemptId < attempts) {
            const [xl, yt, xr, yb] = [
                0,
                0,
                2.5 * (meanWidth + attemptId * step) * (graph.nodes.length) ** 0.5,
                2.5 * (meanHeight + attemptId * step) * (graph.nodes.length) ** 0.5,
            ];

            collisions = false;
            // assign random positions
            positions = graph.nodes.map((_node) => [
                Math.random() * (xr - xl) + xl,
                Math.random() * (yb - yt) + yt,
            ]);
            for (let i = 0; i < 10; i += 1) {
                const delaunay = d3.Delaunay.from(positions);
                const voronoi = delaunay.voronoi([xl, yt, xr, yb]);
                positions = [...voronoi.cellPolygons()].map(
                    (polygon) => computeCentroid(polygon),
                );
            }
            collisions = graph.nodes.some(
                (node1, i1) => graph.nodes.slice(i1 + 1).some(
                    (node2, ir) => collide(
                        node1,
                        node2,
                        positions[i1],
                        positions[i1 + 1 + ir],
                    ),
                ),
            );
            attemptId += 1;
        }

        const nodes = graph.nodes.map(
            (node, i) => ({
                ...node,
                position: { x: positions[i][0], y: positions[i][1] },
            }),
        );
        return { ...graph, nodes };
    }
}
