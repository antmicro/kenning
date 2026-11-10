/*
 * Copyright (c) 2022-2024 Antmicro <www.antmicro.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 */

import { computed, ref, Ref } from 'vue';
import { useGraph, useViewModel } from '@baklavajs/renderer-vue';
import { gridSnapper, nodeSnapper } from '../core/snappers';

interface coordinates {
    x: number;
    y: number;
}

/* eslint-disable no-param-reassign, @typescript-eslint/no-explicit-any */
export default function useDragMove(
    positionRef: Ref<coordinates>,
    gridSnapperInstance = undefined,
    nodeId = undefined,
) {
    // any definition is an ad-hoc solution as we don't have our graph definition
    const { graph } = useGraph() as { graph: any };
    const { viewModel } = useViewModel();

    const calculateSnappedPosition = gridSnapperInstance ?? gridSnapper(ref(1));

    const gridSize = computed(() => viewModel.value.settings.background.gridSize);
    const backgroundGridSnapper = gridSnapper(gridSize);

    const nodeSnappers = {
        x: nodeSnapper('x'),
        y: nodeSnapper('y'),
    };

    const draggingStartPoint = ref<coordinates | null>(null);
    const draggingStartPosition = ref<coordinates | null>(null);
    const movementStepEnabled = ref<boolean>(true);

    const dragging = computed(() => {
        // If we didn't move, then we are not dragging
        if (!!draggingStartPoint.value && draggingStartPosition.value) {
            return (
                draggingStartPosition.value.x !== positionRef.value.x
                || draggingStartPosition.value.y !== positionRef.value.y
            );
        }
        return false;
    });

    /* eslint-disable arrow-body-style */
    const calculatePosition = (pos: number, kind: 'x' | 'y', align = false, gridSnap = false) => {
        if (align && nodeId !== undefined) {
            const alignedCoord = nodeSnappers[kind](
                pos,
                nodeId,
                graph.value.selectedNodes.map((node: { id: string }) => node.id));
            if (alignedCoord !== undefined) {
                return alignedCoord;
            }
        }
        if (gridSnap) {
            return backgroundGridSnapper(pos);
        }
        // allow either snap-to-grid or free movement
        return movementStepEnabled.value ? calculateSnappedPosition(pos) : pos;
    };
    /* eslint-enable arrow-body-style */

    if (positionRef.value) {
        positionRef.value.x = calculatePosition(positionRef.value.x, 'x');
        positionRef.value.y = calculatePosition(positionRef.value.y, 'y');
    }

    const onPointerDown = (ev: PointerEvent) => {
        movementStepEnabled.value = !ev.shiftKey;
        draggingStartPoint.value = {
            x: ev.pageX,
            y: ev.pageY,
        };
        draggingStartPosition.value = {
            x: positionRef.value.x,
            y: positionRef.value.y,
        };
    };

    const onPointerMove = (ev: PointerEvent) => {
        movementStepEnabled.value = !ev.shiftKey;
        const align = ev.ctrlKey;
        const gridSnap = ev.shiftKey;
        if (draggingStartPoint.value && draggingStartPosition.value) {
            const dx = ev.pageX - draggingStartPoint.value.x;
            const dy = ev.pageY - draggingStartPoint.value.y;
            positionRef.value.x = calculatePosition(
                draggingStartPosition.value.x + dx / graph.value.scaling,
                'x',
                align,
                gridSnap,
            );
            positionRef.value.y = calculatePosition(
                draggingStartPosition.value.y + dy / graph.value.scaling,
                'y',
                align,
                gridSnap,
            );
        }
    };

    const onPointerUp = () => {
        draggingStartPoint.value = null;
        draggingStartPosition.value = null;
        movementStepEnabled.value = true;
    };

    return {
        dragging,
        onPointerDown,
        onPointerMove,
        onPointerUp,
        draggingStartPosition,
    };
}
