/*
 * Copyright (c) 2022-2023 Antmicro <www.antmicro.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Function that check intersection between two bounding boxes
 * @returns True when bboxes intersect, False otherwise
 */
function boundingRectIntersection(bboxA, bboxB) {
    return !(bboxB.xBegin > bboxA.xEnd
        || bboxB.xEnd < bboxA.xBegin
        || bboxB.yBegin > bboxA.yEnd
        || bboxB.yEnd < bboxA.yBegin
    );
}

/**
 * Checks if node is present inside selection rectangle
 * @returns
 */
export default function nodeInsideSelection(graph, node, position, boundingRect) {
    const nodeHTMLelement = document.getElementById(node.id);
    const selectionBoundingRect = boundingRect;

    const panningX = graph.panning.x;
    const panningY = graph.panning.y;
    const { scaling } = graph;
    const navBarHeight = 60;

    const nodeX = scaling * (panningX + position.x);
    const nodeY = scaling * (panningY + position.y) + navBarHeight;
    const nodeWidth = nodeHTMLelement.offsetWidth;
    const nodeHeight = nodeHTMLelement.offsetHeight;

    const nodeBoundingRect = {
        xBegin: nodeX,
        yBegin: nodeY,
        xEnd: nodeX + nodeWidth * scaling,
        yEnd: nodeY + nodeHeight * scaling,
    };

    return boundingRectIntersection(selectionBoundingRect, nodeBoundingRect);
}
