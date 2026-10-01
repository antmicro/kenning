/*
 * Copyright (c) 2022-2023 Antmicro <www.antmicro.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

import { BaseLayoutAlgorithm } from './baseEngine.js';

/**
 * Default layout algorithm that puts all nodes into (0, 0) position
 */
export default class NoLayoutAlgorithm extends BaseLayoutAlgorithm {
    /* eslint-disable class-methods-use-this */
    availableAlgorithms = [
        'NoLayout',
    ];

    activeAlgorithm = 'NoLayout';

    calculate(graph) {
        const nodes = graph.nodes.map((node) => ({
            ...node,
            position: { x: 0, y: 0 },
        }));
        return { ...graph, nodes };
    }
}
