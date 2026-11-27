/*
 * Copyright (c) 2022-2023 Antmicro <www.antmicro.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 */

import { Ref } from 'vue';
import { useGraph } from '@baklavajs/renderer-vue';
import useDragMove from './useDragMove';

interface coordinates {
    x: number;
    y: number;
}

/* eslint-disable no-param-reassign, @typescript-eslint/no-explicit-any */
export default function useGroupDragMove(
    dragRootNodePosition: Ref<coordinates>,
    dragRootNodeId = undefined,
    gridSnapperInstance = undefined,
) {
    const { graph } = useGraph() as { graph: any };
    const totalOffset:coordinates = {
        x: 0,
        y: 0,
    };

    const groupDragMove = useDragMove(
        dragRootNodePosition,
        gridSnapperInstance,
        dragRootNodeId,
    );

    const groupPointerMove = groupDragMove.onPointerMove;
    const groupPointerUp = groupDragMove.onPointerUp;

    const onPointerMove = (ev: PointerEvent) => {
        const groupPositionCoords = {
            x: dragRootNodePosition.value.x,
            y: dragRootNodePosition.value.y,
        };

        groupPointerMove(ev);

        const dx = dragRootNodePosition.value.x - groupPositionCoords.x;
        const dy = dragRootNodePosition.value.y - groupPositionCoords.y;

        totalOffset.x += dx;
        totalOffset.y += dy;

        graph.value.selectedNodes.forEach((node: any) => {
            if (node.id !== dragRootNodeId) {
                node.position.x += dx;
                node.position.y += dy;
            }
        });
    };

    const onPointerUp = () => {
        if (Math.abs(totalOffset.x) + Math.abs(totalOffset.y) !== 0) {
            // Call history event for multiple node dragging
            graph.value.selectedNodes.forEach((node: any) => {
                node.position.x -= totalOffset.x;
                node.position.y -= totalOffset.y;
            });
            graph.value.dragNodes(graph.value.selectedNodes);
            graph.value.selectedNodes.forEach((node: any) => {
                node.position.x += totalOffset.x;
                node.position.y += totalOffset.y;
            });
        }
        totalOffset.x = 0;
        totalOffset.y = 0;

        groupPointerUp();
    };

    groupDragMove.onPointerMove = onPointerMove;
    groupDragMove.onPointerUp = onPointerUp;

    return groupDragMove;
}
