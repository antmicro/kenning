/*
 * Copyright (c) 2022-2024 Antmicro <www.antmicro.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Implements additional functions for CustomNode
 */

import { useViewModel, useGraph } from '@baklavajs/renderer-vue';
import { nextTick } from 'vue';

import notifyEvents from './notifyEvents.js';

/**
 * Updates a side and optionally a sidePosition of an interface
 *
 * @param node in which the interface is updated
 * @param intf interface to update
 * @param newSide new side of the interface
 * @param newSidePosition new position of the interface. If it is occupied
 * @param swap if true then if an interface is found in 'newSidePosition' then they
 * are swapped
 * then an old interface is moved.
 */
/* eslint-disable no-param-reassign */
export function updateInterfacePosition(
    node,
    intf,
    newSide,
    newSidePosition = undefined,
    swap = false,
    editNode = true,
) {
    if (editNode) {
        const { graph } = useGraph();
        graph.value.editNode(node);
    }

    const oldSidePosition = intf.sidePosition;
    const oldSide = intf.side;

    intf.side = newSide;
    if (intf.bus?.stubs !== undefined) {
        intf.bus.stubs.forEach((stub) => { stub.side = newSide; });
    }
    if (newSidePosition !== undefined) {
        intf.sidePosition = newSidePosition;
    }

    const found = [
        ...Object.values(node.inputs),
        ...Object.values(node.outputs),
    ].find(
        (io) => io.id !== intf.id &&
            io.sidePosition === intf.sidePosition &&
            io.side === intf.side,
    );

    if (found !== undefined) {
        if (newSidePosition !== undefined && swap) {
            found.sidePosition = oldSidePosition;
            found.side = oldSide;
        } else {
            const intfToMove = newSidePosition === undefined ? intf : found;

            // Finding the first non occupied side position on that side
            const sameSide = [
                ...Object.values(node.inputs),
                ...Object.values(node.outputs),
            ].filter((io) => io.side === intfToMove.side && !io.hidden);
            const occupiedPositions = sameSide.map((io) => io.sidePosition);

            let proposedPosition = 0;
            while (occupiedPositions.includes(proposedPosition)) {
                proposedPosition += 1;
            }

            intfToMove.sidePosition = proposedPosition;
        }
    }
}

/**
 * Wrapper for nodes removal
 *
 * @param node node to remove. Can be either a graph node or a regular node
 * @param unwrapGraph determines whether to unwrap the graph node contents
 * into the current graph. Can be used only when removing a graph node
 */
export function removeNode(node, unwrapGraph = false) {
    const { viewModel } = useViewModel();

    // get graph in which node is present
    const graphWithNode = Array.from(viewModel.value.editor.graphs)
        .find((graph) => graph.nodes.some((n) => n.id === node.id));

    const isGraphNode = viewModel.value.editor.isGraphNode(node.type);
    if (isGraphNode && unwrapGraph) {
        viewModel.value.editor.unwrapSubgraph(node);
    } else {
        const { graph } = useGraph();
        notifyEvents.removedNode.emit(node);
        graph.value.ungroupNode(node);
        graph.value.removeNode(node);
    }

    const { removeEmptySubgraphs } = viewModel.value.settings;

    if (graphWithNode?.nodes.length === 0 && viewModel.value.editor.isInSubgraph()
        && removeEmptySubgraphs) {
        viewModel.value.editor.unregisterGraph(graphWithNode);
        const { graphNode } = graphWithNode;
        viewModel.value.editor.backFromSubgraph();

        nextTick().then(() => {
            notifyEvents.subgraphDestroyed.emit({
                node: graphNode,
                subgraph: graphWithNode,
            });
        });
    }
}
