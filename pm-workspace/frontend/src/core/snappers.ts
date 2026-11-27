/*
 * Copyright (c) 2022-2023 Antmicro <www.antmicro.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

import { useGraph } from '@baklavajs/renderer-vue';
import { Ref } from 'vue';

/* eslint-disable @typescript-eslint/no-explicit-any */
/**
 * Function for calculating node position based on given movementStep.
 *
 * @param movementStep Vue's reference to value containing snap offset
 */
export function gridSnapper(movementStep: Ref<number>) {
    const calculateSnappedPosition = (coord: number) =>
        Math.round(coord / movementStep.value) * movementStep.value;

    return calculateSnappedPosition;
}

/**
 * Creates function that aligns the value of node position along specified axis if it is close
 * enough to other node position
 *
 * @param kind Either 'x' or 'y', defines along which axis the coordinate is aligned
 */
export function nodeSnapper(kind: 'x' | 'y') {
    const { graph } = useGraph();
    const snapDistance = 100;
    const calculateSnappedPosition = (
        coord: number,
        nodeId: string,
        selectedNodesIds: Array<string>) => {
        const nearestCoords = graph.value.nodes
            // any definition is an ad-hoc solution as we don't have our node definition
            .filter((node: any) => node.id !== nodeId && !selectedNodesIds.includes(node.id))
            .map((node: any) => node.position[kind])
            .filter((otherCoords) => Math.abs(coord - otherCoords) < snapDistance);
        return nearestCoords.length !== 0 ? Math.min(...nearestCoords) : undefined;
    };
    return calculateSnappedPosition;
}
