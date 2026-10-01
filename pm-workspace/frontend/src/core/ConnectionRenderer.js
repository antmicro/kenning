/*
 * Copyright (c) 2022-2026 Antmicro <www.antmicro.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Handles calculation of a SVG path a connection should have based on it's characteristic
 * (for example loopback) and style. It is handled by `ConnectionRenderer`, which
 * deals with saving the style user has chosen and delegating away the calculation to
 * suitable renderer
 */

/* eslint-disable max-classes-per-file */
import getDomElements from '../custom/connection/domResolver.js';
import getPortCoordinates from '../custom/connection/connectionTools.js';

import PriorityQueue from './PriorityQueue.ts';

/**
 * Retrieves top point of a node based on it's DOM element If the element does not yet exists,
 * returns 0
 *
 * @param interfaceRef interface reference
 * @returns Y coordinate of a bottom of a node, adjusted for canvas transformation
 */
/* eslint-disable no-unused-vars */
function nodeTopPoint(interfaceRef) {
    const nodeHtml = document.getElementById(interfaceRef.nodeId);
    const nodeTop = nodeHtml ? nodeHtml.offsetTop : 0;
    return nodeTop;
}

/**
 * Used for loopback connections, calculates the y coordinate of a bottom point of a node
 * based on it's DOM element. If the element does not yet exists, returns 0
 *
 * @param interfaceRef interface reference
 * @returns Y coordinate of a bottom of a node, adjusted for canvas transformation
 */
function nodeBottomPoint(interfaceRef) {
    const nodeHtml = document.getElementById(interfaceRef.nodeId);
    const nodeBottom = nodeHtml ? nodeHtml.offsetTop + nodeHtml.offsetHeight : 0;
    return nodeBottom;
}

/**
 * Utility function that calculates the x and y radius of an ellipse given center point and
 * a slope at a specified point
 *
 * @param x X coordinate of a point on an ellipse
 * @param y Y coordinate of a point on an ellipse
 * @param cx X coordinate of a center point
 * @param cy Y coordinate of a center point
 * @param slope dy/dx value on a (x, y) point
 * @returns Array of two elements: radius parallel to x axis and y axis respectively
 */
function calculateEllipseR(x, y, cx, cy, slope) {
    const rx = Math.sqrt(Math.abs((x - cx) * (x - cx) + ((x - cx) * (y - cy)) / slope));
    const ry = Math.sqrt(Math.abs((y - cy) * (y - cy) + (y - cy) * (x - cx) * slope));
    return [rx, ry];
}

/**
 * Function that returns first non occupied position index based on input array.
 *
 * @param arr - Array with occupied positions.
 * @returns Free position index.
 */
function getIndex(arr) {
    if (arr?.length) {
        const arrSorted = arr.sort((sp1, sp2) => sp1 - sp2);
        const out = arrSorted.findIndex((val, index, _) => val !== index);
        if (out === -1) {
            return arrSorted.length;
        }
        return out;
    }

    return 0;
}

class NormalizedConnection {
    /**
     * Class that makes sure that the connection is in correct order, which means that from and to
     * sockets and their coordinates are properly set.
     */
    constructor(x1, y1, x2, y2, connection) {
        this.x1 = x1;
        this.y1 = y1;
        this.x2 = x2;
        this.y2 = y2;
        this.from = connection.from;
        if (connection.to) {
            this.to = connection.to;
            if (
                (this.from.direction === 'input' && this.to.direction === 'output') ||
                (this.from.direction === 'input' && this.to.direction === 'inout') ||
                (this.from.direction === 'inout' && this.to.direction === 'output')
            ) {
                [this.x1, this.x2, this.y1, this.y2] = [this.x2, this.x1, this.y2, this.y1];
                [this.from, this.to] = [this.to, this.from];
            }
        }
    }
}

/**
 * Defines the type of a point in the graph.
 */
const PointType = Object.freeze({
    /** Point representing the `from` interface of the connection. */
    FROM_INT: 'fromInt',
    /** Auxiliary point used between the `from` interface and regular grid. */
    FROM_HELPER: 'fromHelper',
    /** Regular point - an element of the grid. */
    REG: 'reg',
    /** Auxiliary point used between regular grid and `to` interface. */
    TO_HELPER: 'toHelper',
    /** Point representing `to` interface of the connection. */
    TO_INT: 'toInt',
});

/**
 * Parameters used in A* pathfinding algorithm during rendering.
 */
const aStarConfig = {
    /** Time in ms after which A* cache entries are considered expired. */
    cacheInvalidationPeriod: 50,
    /** Function used to compute the distance metric in the A* algorithm. */
    distanceType: (x1, y1, x2, y2) => Math.abs(x1 - x2) + Math.abs(y1 - y2),
    /** Function used to compute the grid step size for each path. */
    gridStepFunc: (x1, y1, x2, y2) => Math.max(
        50,
        Math.abs(x1 - x2) / 10,
        Math.abs(y1 - y2) / 10,
    ),
    /** Zone size used in spatial hashing. */
    zoneStep: 500,
    /** Function used when accessing zoneInfo map. */
    zoneInfoKey: (idx, idy) => `${idx}:${idy}`,
    /** Factor by which segment cost is multiplied when it intersects a node. */
    intersectionFactor: 10,
};

export default class ConnectionRenderer {
    style = 'curved';

    viewModel = null;

    randomizedOffset = false;

    switchableInterfaces = false;

    shiftDistance = 15;

    aStarCache = new Map();

    /**
     * Defines the shift the connection should have compared to the default position based on the
     * index of the `from` interface and `to` interface bound to the connection in the nodes.
     * It allows to make a visual distinction between connections going to different interfaces
     * in the same node. The value of the shift is constant distance times the index of an
     * interface, adjusted for any canvas transformation
     * This funnction is symmetrical.
     *
     * @param ncFrom from node reference
     * @param ncTo to node reference
     * @param graph the graph definition
     * @returns Value the connection should shift from it's default position
     */
    getShift(ncFrom, ncTo, graph) {
        const fromPosition = ncFrom.sidePosition;
        const toPosition = ncTo.sidePosition;

        const shiftIndex = (fromPosition + toPosition) / 2;

        if (this.randomizedOffset) {
            const fromInterfaceNeighbours = this.getInterfaceNeighbors(ncFrom, graph);
            const toInterfaceNeighbours = this.getInterfaceNeighbors(ncTo, graph);

            // the string is a sum of utf16 representation of each character
            let toRandomIndex =
                [...ncTo.id].reduce((accumulator, char) => accumulator + char.charCodeAt(0), 0) ??
                0;
            let fromRandomIndex =
                [...ncFrom.id].reduce((accumulator, char) => accumulator + char.charCodeAt(0), 0) ??
                0;

            const randomIndex = (toRandomIndex ^ fromRandomIndex); // eslint-disable-line no-bitwise
            fromRandomIndex = randomIndex % fromInterfaceNeighbours.length;
            toRandomIndex = randomIndex % toInterfaceNeighbours.length;
            const randomShiftIndex = (fromRandomIndex + toRandomIndex) / 2;
            return this.shiftDistance * (randomShiftIndex / 2 + shiftIndex / 2);
        }

        return this.shiftDistance * shiftIndex;
    }

    /**
     * Checks maximum number of interfaces and computes maximum shift.
     *
     * @param ncFrom node from reference
     * @param ncTo to node reference
     * @param graph the graph definition
     * @returns Maximum value of the shift that the connection can obtain
     */
    getMaxShift(ncFrom, ncTo, graph) {
        const maxFrom = this.getInterfaceNeighbors(ncFrom, graph).length - 1;
        const maxTo = this.getInterfaceNeighbors(ncTo, graph).length - 1;
        return this.shiftDistance * ((maxFrom + maxTo) / 2);
    }

    /**
     * Checks node and returns interfaces on the same side.
     *
     * @param interfaceRef node reference
     * @param graph the graph definition
     * @returns Array of neighboring intefraces on the same side
     */
    // eslint-disable-next-line class-methods-use-this
    getInterfaceNeighbors(interfaceRef, graph) {
        const node = graph.findNodeById(interfaceRef.nodeId);
        return [
            ...Object.values(node.inputs),
            ...Object.values(node.outputs),
        ].filter((c) => c.side === interfaceRef.side && c.port);
    }

    /**
     * Checks whether one of the nodes is above or below the other one.
     *
     * @param ncFrom from interface reference
     * @param ncTo to interface reference
     * @param graph the graph definition
     * @returns True if one node is higher than the other one
     */
    // eslint-disable-next-line class-methods-use-this
    someAboveOrBelow(ncFrom, ncTo, graph) {
        const nodeFromTop = nodeTopPoint(ncFrom);
        const nodeToTop = nodeTopPoint(ncTo);
        const nodeFromBottom = nodeBottomPoint(ncFrom);
        const nodeToBottom = nodeBottomPoint(ncTo);
        return nodeFromBottom < nodeToTop || nodeFromTop > nodeToBottom;
    }

    /* eslint-disable class-methods-use-this */
    curvedRender(x1, y1, x2, y2, connection) {
        const graph = this.viewModel.displayedGraph;
        const nc = new NormalizedConnection(x1, y1, x2, y2, connection);
        const dx = 0.3 * Math.abs(nc.x1 - nc.x2);
        const dy = 0.3 * Math.abs(nc.y1 - nc.y2);

        if (nc.to) {
            if (nc.from.side === 'left' && nc.to.side === 'top') {
                return `M ${nc.x1} ${nc.y1} C ${nc.x1 - dx} ${nc.y1}, ${nc.x2} ${nc.y2 - dy}, ${
                    nc.x2
                } ${nc.y2}`;
            }
            if (nc.from.side === 'left' && nc.to.side === 'bottom') {
                return `M ${nc.x1} ${nc.y1} C ${nc.x1 - dx} ${nc.y1}, ${nc.x2} ${nc.y2 + dy}, ${
                    nc.x2
                } ${nc.y2}`;
            }
            if (nc.from.side === 'left' && nc.to.side === 'right') {
                return `M ${nc.x1} ${nc.y1} C ${nc.x1 - dx} ${nc.y1}, ${nc.x2 + dx} ${nc.y2}, ${
                    nc.x2
                } ${nc.y2}`;
            }

            if (nc.from.side === 'right' && nc.to.side === 'top') {
                return `M ${nc.x1} ${nc.y1} C ${nc.x1 + dx} ${nc.y1}, ${nc.x2} ${nc.y2 - dy}, ${
                    nc.x2
                } ${nc.y2}`;
            }
            if (nc.from.side === 'right' && nc.to.side === 'bottom') {
                return `M ${nc.x1} ${nc.y1} C ${nc.x1 + dx} ${nc.y1}, ${nc.x2} ${nc.y2 + dy}, ${
                    nc.x2
                } ${nc.y2}`;
            }
            if (nc.from.side === 'right' && nc.to.side === 'left') {
                return `M ${nc.x1} ${nc.y1} C ${nc.x1 + dx} ${nc.y1}, ${nc.x2 - dx} ${nc.y2}, ${
                    nc.x2
                } ${nc.y2}`;
            }

            if (nc.from.side === 'bottom' && nc.to.side === 'left') {
                return `M ${nc.x1} ${nc.y1} C ${nc.x1} ${nc.y1 + dy}, ${nc.x2 - dx} ${nc.y2}, ${
                    nc.x2
                } ${nc.y2}`;
            }
            if (nc.from.side === 'bottom' && nc.to.side === 'right') {
                return `M ${nc.x1} ${nc.y1} C ${nc.x1} ${nc.y1 + dy}, ${nc.x2 + dx} ${nc.y2}, ${
                    nc.x2
                } ${nc.y2}`;
            }
            if (nc.from.side === 'bottom' && nc.to.side === 'top') {
                return `M ${nc.x1} ${nc.y1} C ${nc.x1} ${nc.y1 + dy}, ${nc.x2} ${nc.y2 - dy}, ${
                    nc.x2
                } ${nc.y2}`;
            }

            if (nc.from.side === 'top' && nc.to.side === 'left') {
                return `M ${nc.x1} ${nc.y1} C ${nc.x1} ${nc.y1 - dy}, ${nc.x2 - dx} ${nc.y2}, ${
                    nc.x2
                } ${nc.y2}`;
            }
            if (nc.from.side === 'top' && nc.to.side === 'right') {
                return `M ${nc.x1} ${nc.y1} C ${nc.x1} ${nc.y1 - dy}, ${nc.x2 + dx} ${nc.y2}, ${
                    nc.x2
                } ${nc.y2}`;
            }
            if (nc.from.side === 'top' && nc.to.side === 'bottom') {
                return `M ${nc.x1} ${nc.y1} C ${nc.x1} ${nc.y1 - dy}, ${nc.x2} ${nc.y2 + dy}, ${
                    nc.x2
                } ${nc.y2}`;
            }

            if (nc.from.side === 'right' && nc.to.side === 'right') {
                const rightmost = Math.max(nc.x1 + dx, nc.x2 + dx) + 50;
                return `M ${nc.x1} ${nc.y1} C ${rightmost} ${nc.y1}, ${rightmost} ${nc.y2}, ${nc.x2} ${nc.y2}`;
            }
            if (nc.from.side === 'left' && nc.to.side === 'left') {
                const leftmost = Math.min(nc.x1 - dx, nc.x2 - dx) - 50;
                return `M ${nc.x1} ${nc.y1} C ${leftmost} ${nc.y1}, ${leftmost} ${nc.y2}, ${nc.x2} ${nc.y2}`;
            }
            if (nc.from.side === 'top' && nc.to.side === 'top') {
                const topmost = Math.min(nc.y1 - dy, nc.y2 - dy);
                return `M ${nc.x1} ${nc.y1} C ${nc.x1} ${topmost}, ${nc.x2} ${topmost}, ${nc.x2} ${nc.y2}`;
            }
            if (nc.from.side === 'bottom' && nc.to.side === 'bottom') {
                const bottommost = Math.max(nc.y1 + dy, nc.y2 + dy);
                return `M ${nc.x1} ${nc.y1} C ${nc.x1} ${bottommost}, ${nc.x2} ${bottommost}, ${nc.x2} ${nc.y2}`;
            }
        }

        if (nc.from.side === 'right') {
            return `M ${nc.x1} ${nc.y1} C ${nc.x1 + dx} ${nc.y1}, ${nc.x2 - dx} ${nc.y2}, ${
                nc.x2
            } ${nc.y2}`;
        }
        if (nc.from.side === 'left') {
            return `M ${nc.x1} ${nc.y1} C ${nc.x1 - dx} ${nc.y1}, ${nc.x2 + dx} ${nc.y2}, ${
                nc.x2
            } ${nc.y2}`;
        }
        if (nc.from.side === 'top') {
            return `M ${nc.x1} ${nc.y1} C ${nc.x1} ${nc.y1 - dy}, ${nc.x2} ${nc.y2 - dy}, ${
                nc.x2
            } ${nc.y2}`;
        }
        if (nc.from.side === 'bottom') {
            return `M ${nc.x1} ${nc.y1} C ${nc.x1} ${nc.y1 + dy}, ${nc.x2} ${nc.y2 + dy}, ${
                nc.x2
            } ${nc.y2}`;
        }

        // unreachable, added to make eslint happy
        return undefined;
    }

    curvedRenderLoopback(x1, y1, x2, y2, connection) {
        const graph = this.viewModel.displayedGraph;
        const nc = new NormalizedConnection(x1, y1, x2, y2, connection);
        const sideMargin = 10;

        if (nc.from.id === nc.to.id) {
            // The same interface
            const shift = this.getShift(nc.from, nc.to, graph) + 30;
            const x = nc.from.side === 'right' ? nc.x1 + shift : nc.x1 - shift;
            return `M ${nc.x1} ${nc.y1}
            A ${sideMargin} ${sideMargin / 2} 0 0 0 ${x} ${nc.y1}
            A ${sideMargin} ${sideMargin / 2} 0 0 0 ${nc.x1} ${nc.y1}`;
        }
        if (nc.from.side === 'left' && nc.to.side === 'left') {
            const leftRx = sideMargin;
            const leftRy = Math.abs(nc.y1 - nc.y2) / 2;
            const renderingSide = nc.y1 > nc.y2 ? 1 : 0;

            return `M ${nc.x1} ${nc.y1}
            A ${leftRx} ${leftRy} 0 0 ${renderingSide} ${nc.x2} ${nc.y2}`;
        }
        if (nc.from.side === 'right' && nc.to.side === 'right') {
            const leftRx = sideMargin;
            const leftRy = Math.abs(nc.y1 - nc.y2) / 2;
            const renderingSide = nc.y1 > nc.y2 ? 0 : 1;

            return `M ${nc.x1} ${nc.y1}
            A ${leftRx} ${leftRy} 0 0 ${renderingSide} ${nc.x2} ${nc.y2}`;
        }

        const shift = this.getShift(nc.from, nc.to, graph) + 30;

        const leftx = nc.from.side === 'left' ? nc.x1 : nc.x2;
        const rightx = nc.to.side === 'right' ? nc.x2 : nc.x1;

        const lefty = nc.from.side === 'left' ? nc.y1 : nc.y2;
        const righty = nc.to.side === 'right' ? nc.y2 : nc.y1;
        const bottomY = nodeBottomPoint(nc.from);

        const y = bottomY + shift;

        const rightCx = rightx - shift;
        const rightCy = (y + righty) / 2;
        const [rightRx, rightRy] = calculateEllipseR(rightx, y, rightCx, rightCy, 1);

        const bottomCx = (rightx + leftx) / 2;
        const bottomCy = bottomY;
        const [bottomRx, bottomRy] = calculateEllipseR(rightx, y, bottomCx, bottomCy, 1);

        const leftCx = leftx + shift;
        const leftCy = (y + lefty) / 2;
        const [leftRx, leftRy] = calculateEllipseR(leftx, y, leftCx, leftCy, -1);

        return `M ${rightx} ${righty}
        A ${rightRx} ${rightRy} 0 0 1 ${rightx} ${y}
        A ${bottomRx} ${bottomRy} 0 0 1 ${leftx} ${y}
        A ${leftRx} ${leftRy} 0 0 1 ${leftx} ${lefty}`;
    }

    correctAnchor(anchor, anchors, nc) {
        const path = this.orthogonalAnchorsPath(anchors, nc);
        let minDist = Infinity;
        let minIdx = -1;
        const tx = anchor.x;
        const ty = anchor.y;
        const distance = (a, b) => Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2);
        const between = (x, a, b) => x >= Math.min(a, b) && x <= Math.max(a, b);
        for (let i = 0; i < path.length - 1; i += 1) {
            const vertical = (path[i].x === path[i + 1].x);
            let dist = vertical ? Math.abs(path[i].x - tx) : Math.abs(path[i].y - ty);

            const betwix = vertical ? between(ty, path[i].y, path[i + 1].y)
                : between(tx, path[i].x, path[i + 1].x);
            if (!betwix) {
                dist = Math.min(distance(anchor, path[i]), distance(anchor, path[i + 1]));
            }

            if (dist < minDist) {
                minDist = dist;
                minIdx = i;
            }
        }
        // eslint-disable-next-line no-param-reassign
        anchor.index = minIdx;
    }

    orthogonalAnchorsPath(anchors, nc) {
        anchors.filter((a) => a.index === -1)
            .forEach((a, i) => this.correctAnchor(a, anchors.slice(0, i), nc));

        const ogPath = this.orthogonalRender(nc.x1, nc.y1, nc.x2, nc.y2, nc);
        const anchored = Array(ogPath.length).fill(undefined);

        const firstMiddle = 1;
        anchored[firstMiddle] = { ...ogPath[firstMiddle] };
        const lastMiddle = ogPath.length - 2;
        anchored[lastMiddle] = { ...ogPath[lastMiddle] };

        const anchorToSeg = [];
        const clamp = (x, a, b, padding = 0) => Math.min(Math.max(x, Math.min(a, b) + padding),
            Math.max(a, b) - padding);
        anchors.forEach((anchor) => {
            const anchorTrans = { ...anchor };
            // eslint-disable-next-line no-param-reassign
            anchor.index = clamp(anchor.index, firstMiddle, ogPath.length - 2 - 1);
            let idx = anchor.index;
            const vertical = (ogPath[idx].x === ogPath[idx + 1].x);
            if (anchored[idx]) {
                ogPath.splice(idx + 1, 0, { x: anchored[idx].x, y: anchored[idx].y });
                anchored.splice(idx + 1, 0, undefined);
                idx += 1;
            }
            if (anchored[idx + 1]) {
                ogPath.splice(idx + 1, 0, { x: anchored[idx + 1].x, y: anchored[idx + 1].y });
                anchored.splice(idx + 1, 0, undefined);
            }
            ogPath.splice(idx + 1, 0, anchorTrans);
            anchored.splice(idx + 1, 0, anchorTrans);
            if (vertical) {
                // changing x
                ogPath[idx].x = anchorTrans.x;
                ogPath[idx + 1].x = anchorTrans.x;
                ogPath[idx + 2].x = anchorTrans.x;
            } else {
                ogPath[idx].y = anchorTrans.y;
                ogPath[idx + 1].y = anchorTrans.y;
                ogPath[idx + 2].y = anchorTrans.y;
            }
            anchorToSeg.push({
                id: anchor.id,
                type: vertical,
                a: ogPath[idx],
                b: ogPath[idx + 2],
                mid: ogPath[idx + 1],
            });
        });
        anchorToSeg.reverse().forEach((info) => {
            const anch = anchors.find((a) => a.id === info.id);
            const isVert = info.type;
            const { a, b, mid } = info;
            if (isVert) {
                anch.y = clamp(anch.y, a.y, b.y, 0.01);
                mid.y = anch.y;
            } else {
                anch.x = clamp(anch.x, a.x, b.x, 0.01);
                mid.x = anch.x;
            }
        });
        return ogPath;
    }

    alternativeOrthogonalAnchorsPath(anchors, nc) {
        const path = [{ x: nc.x1, y: nc.y1 }];
        let direction = nc.from.side;
        let offset = 0;
        if (direction === 'left') offset = -20;
        else offset = 20;
        path.push({ x: nc.x1 + offset, y: nc.y1 });
        direction = 'horizontal';
        anchors.forEach((anchor) => {
            if (direction === 'vertical') {
                path.push({
                    x: path[path.length - 1].x + (anchor.x - path[path.length - 1].x) / 2,
                    y: path[path.length - 1].y,
                });
                path.push({
                    x: path[path.length - 2].x + (anchor.x - path[path.length - 2].x) / 2,
                    y: anchor.y,
                });
                path.push({
                    x: anchor.x,
                    y: anchor.y,
                });
                direction = 'horizontal';
            } else {
                path.push({
                    x: path[path.length - 1].x,
                    y: path[path.length - 1].y + (anchor.y - path[path.length - 1].y) / 2,
                });
                path.push({
                    x: anchor.x,
                    y: path[path.length - 2].y + (anchor.y - path[path.length - 2].y) / 2,
                });
                path.push({
                    x: anchor.x,
                    y: anchor.y,
                });
                direction = 'vertical';
            }
        });
        if (direction === 'vertical') {
            path.push({
                x: path[path.length - 1].x + (nc.x2 - path[path.length - 1].x) / 2,
                y: path[path.length - 1].y,
            });
            path.push({
                x: path[path.length - 2].x + (nc.x2 - path[path.length - 2].x) / 2,
                y: nc.y2,
            });
            path.push({
                x: nc.x2,
                y: nc.y2,
            });
        } else {
            path.push({
                x: path[path.length - 1].x,
                y: path[path.length - 1].y + (nc.y2 - path[path.length - 1].y) / 2,
            });
            if (nc.to.side === 'left') offset = 10;
            else offset = -10;
            path.push({
                x: nc.x2 - offset,
                y: path[path.length - 2].y + (nc.y2 - path[path.length - 2].y) / 2,
            });
            path.push({
                x: nc.x2 - offset,
                y: nc.y2,
            });
            path.push({
                x: nc.x2,
                y: nc.y2,
            });
        }
        return path;
    }

    orthogonalRenderLoopback(x1, y1, x2, y2, connection) {
        const graph = this.viewModel.displayedGraph;
        const nc = new NormalizedConnection(x1, y1, x2, y2, connection);
        const shift = this.getShift(nc.from, nc.to, graph) + 30;

        if (connection.anchors !== undefined && connection.anchors.length) {
            const anchors = connection.anchors.map((a) => {
                const tx = a.x;
                const ty = a.y;
                return { x: tx, y: ty };
            });

            const path = [{ x: nc.x1, y: nc.y1 }];
            const direction = nc.from.side;
            const xShift = direction === 'right' ? shift : -shift;
            path.push({ x: nc.x1 + xShift, y: nc.y1 });
            anchors.forEach((anchor) => {
                path.push({
                    x: path[path.length - 1].x,
                    y: anchor.y,
                });
                path.push({
                    x: anchor.x,
                    y: anchor.y,
                });
                // NOTE: Duplicate path is for proper anchor position calculation
                path.push({
                    x: anchor.x,
                    y: anchor.y,
                });
            });
            path.push({
                x: path[path.length - 1].x,
                y: nc.y2,
            });
            path.push({
                x: nc.x2 + xShift,
                y: nc.y2,
            });
            path.push({
                x: nc.x2,
                y: nc.y2,
            });
            return path;
        }

        const bottomY = nodeBottomPoint(nc.from);
        const y = bottomY + shift;

        if (nc.from.side === 'right' && nc.to.side === 'left') {
            return `M ${nc.x1} ${nc.y1}
            h ${shift}
            V ${y} H ${nc.x2 - shift} V ${nc.y2} H ${nc.x2}`;
        }
        if (nc.from.side === 'left' && nc.to.side === 'right') {
            return `M ${nc.x2} ${nc.y2}
            h ${shift}
            V ${y} H ${nc.x1 - shift} V ${nc.y1} H ${nc.x1}`;
        }
        if (nc.from.side === nc.to.side) {
            return this.curvedRenderLoopback(nc.x1, nc.y1, nc.x2, nc.y2, connection);
        }
        // unreachable, added to make eslint happy
        return undefined;
    }

    aStarRender(x1, y1, x2, y2, connection) {
        const cached = this.aStarCache.get(connection.id);

        if (cached && Date.now() - cached.lastRun < aStarConfig.cacheInvalidationPeriod) {
            // Check whether path starts from the same point
            const firstPoint = cached.path.at(0);
            const lastPoint = cached.path.at(-1);

            if (firstPoint.x === x1 && firstPoint.y === y1 &&
                lastPoint.x === x2 && lastPoint.y === y2
            ) {
                return cached.path;
            }
        }

        const nc = new NormalizedConnection(x1, y1, x2, y2, connection);

        const graph = this.viewModel.displayedGraph;

        const nodesInfo = graph.nodes.map((node) => {
            const HTMLelement = document.getElementById(node.id);
            return {
                id: node.id,
                position: node.position,
                width: HTMLelement.offsetWidth,
                height: HTMLelement.offsetHeight,
            };
        });

        const minMargin = 30;

        const regGridStep = aStarConfig.gridStepFunc(nc.x1, nc.y1, nc.x2, nc.y2);

        // Map addressed with formatted strings,
        // with lists of nodeInfo instances as values
        const zoneInfo = new Map();

        // Initialization of spatial hashing data and zones
        // In the end, each zone entry contains a list of node object,
        // each represented by its id, position, width and height
        nodesInfo.forEach((nInfo) => {
            const initIdx = Math.floor((nInfo.position.x - minMargin) / aStarConfig.zoneStep);
            const initIdy = Math.floor((nInfo.position.y - minMargin) / aStarConfig.zoneStep);

            const finalIdx = Math.floor(
                (nInfo.position.x + nInfo.width + minMargin) / aStarConfig.zoneStep,
            );
            const finalIdy = Math.floor(
                (nInfo.position.y + nInfo.height + minMargin) / aStarConfig.zoneStep,
            );

            for (
                let i = Math.min(initIdx, finalIdx);
                i <= Math.max(initIdx, finalIdx);
                i += 1
            ) {
                for (
                    let j = Math.min(initIdy, finalIdy);
                    j <= Math.max(initIdy, finalIdy);
                    j += 1
                ) {
                    const zoneKey = aStarConfig.zoneInfoKey(i, j);
                    const zList = zoneInfo.get(zoneKey);
                    const newList = [];
                    if (zList === undefined) {
                        zoneInfo.set(zoneKey, newList);
                    }
                    zoneInfo.get(zoneKey).push(nInfo);
                }
            }
        });

        if (connection.to) {
            const shift = this.getShift(nc.from, nc.to, graph);

            const fromNode = graph.nodes.filter((node) => node.id === nc.from.nodeId)[0];
            const fromNodeWidth = document.getElementById(fromNode.id).offsetWidth;
            const toNode = graph.nodes.filter((node) => node.id === nc.to.nodeId)[0];
            const toNodeWidth = document.getElementById(toNode.id).offsetWidth;

            const fromPoints = [
                {
                    x: nc.from.side === 'left' ?
                        fromNode.position.x - minMargin :
                        fromNode.position.x + fromNodeWidth + minMargin,
                    y: nc.y1,
                    type: PointType.FROM_HELPER,
                },
                {
                    x: nc.x1, y: nc.y1, type: PointType.FROM_INT, side: nc.from.side,
                },
            ];

            const toPoints = [
                {
                    x: nc.to.side === 'left' ?
                        toNode.position.x - minMargin :
                        toNode.position.x + toNodeWidth + minMargin,
                    y: nc.y2,
                    type: PointType.TO_HELPER,
                },
                {
                    x: nc.x2, y: nc.y2, type: PointType.TO_INT, side: nc.to.side,
                },
            ];

            const result = this.astar(
                fromPoints,
                toPoints,
                regGridStep,
                shift,
                minMargin,
                nodesInfo,
                zoneInfo,
            ).map((point) => ({
                x: point.x,
                y: point.y,
            }));

            this.aStarCache.set(connection.id, {
                path: result,
                lastRun: Date.now(),
            });

            return result;
        }
        const middlePoint = (nc.x1 + nc.x2) / 2;

        const result = [{ x: nc.x1, y: nc.y1 },
            { x: middlePoint, y: nc.y1 },
            { x: middlePoint, y: nc.y2 },
            { x: nc.x2, y: nc.y2 },
        ];

        this.aStarCache.set(connection.id, {
            path: result,
            lastRun: Date.now(),
        });

        return result;
    }

    aStarRenderLoopback(x1, y1, x2, y2, connection) {
        return this.aStarRender(x1, y1, x2, y2, connection);
    }

    /**
     * Computes a path between two points using A* algorithm.
     *
     * @param fromPoints Array of points related to `from` interface
     * @param toPoints Array of points related to `to` interface
     * @param regGridStep Grid step size used by A* to find paths
     * @param shift Offset by which grid is translated
     * @param minMargin Minimum margin around nodes within which
     *                  segments are considered to intersect the node
     * @param nodesInfo Array containing exact positions and sizes of nodes
     * @param zoneInfo Map used in spatial hashing
     * @returns Array of objects with x and y coordinates,
     *          representing consecutive points along the path
     */
    astar(
        fromPoints,
        toPoints,
        regGridStep,
        shift,
        minMargin,
        nodesInfo,
        zoneInfo,
    ) {
        // helper function returning key to index `pointsToIndex` map
        const key = (point) => `${point.x}:${point.y}:${point.type}`;

        /**
         * Reconstructs a path by following predecessors from the current point.
         *
         * @param pointsToIndex Mapping from point keys to their indices
         * @param predecessors Mapping from point index to its direct predecessor
         * @param cur Current point
         * @returns Array of consecutive points from the start to the `cur` point
         */
        function reconstructPath(pointsToIndex, predecessors, cur) {
            let current = cur;
            const totalPath = [current];
            while (true) {
                const currentIndex = pointsToIndex.get(key(current));
                if (!predecessors.has(currentIndex)) break;
                const newCurrent = predecessors.get(currentIndex);

                if (newCurrent.x !== current.x && newCurrent.y !== current.y) {
                    if (totalPath.find((point) => point?.type === PointType.REG)) {
                        // handle `from` connection
                        totalPath.unshift({
                            x: current.x,
                            y: newCurrent.y,
                        });
                    } else {
                        // handle `to` connection
                        totalPath.unshift({
                            x: newCurrent.x,
                            y: current.y,
                        });
                    }
                }
                current = newCurrent;
                totalPath.unshift(current);
            }
            return totalPath;
        }

        /**
         * Computes the neighbours of the current point, taking into account its type
         * and points near the `from` and `to` interfaces.
         *
         * @param current Point whose neighbours are to be found
         * @param fromP Array of points related to `from` interface
         * @param toP Array of points related to `to` interface
         * @returns Array of point objects containing x and y coordinates and their type
         */
        function getNeighbours(current, fromP, toP) {
            switch (current.type) {
                case PointType.FROM_INT:
                    return fromP.slice(0, -1);
                case PointType.FROM_HELPER: {
                    const leftSide = fromP.at(-1).side === 'left';
                    const modShift = shift % regGridStep;
                    const baseX = leftSide ?
                        Math.floor(
                            (current.x + modShift) / regGridStep,
                        ) * regGridStep - modShift :
                        Math.floor(
                            (current.x - modShift) / regGridStep,
                        ) * regGridStep + modShift + regGridStep;
                    const baseY = Math.floor(
                        (current.y - modShift) / regGridStep,
                    ) * regGridStep + modShift;

                    const additionalRegs = [
                        {
                            x: baseX,
                            y: baseY,
                            type: PointType.REG,
                        },
                        {
                            x: baseX,
                            y: baseY + regGridStep,
                            type: PointType.REG,
                        },
                    ];
                    return fromP.slice(0, -1).concat(additionalRegs);
                }
                case PointType.REG: {
                    const toHelperPredicate = (toHelper) => {
                        if (toP.at(-1).side === 'left') {
                            return toHelper.x <= current.x + regGridStep &&
                                toHelper.x >= current.x &&
                                toHelper.y <= current.y + regGridStep &&
                                toHelper.y >= current.y - regGridStep;
                        }
                        return toHelper.x <= current.x &&
                                toHelper.x >= current.x - regGridStep &&
                                toHelper.y <= current.y + regGridStep &&
                                toHelper.y >= current.y - regGridStep;
                    };

                    return [
                        {
                            x: current.x + regGridStep,
                            y: current.y,
                            type: PointType.REG,
                        },
                        {
                            x: current.x - regGridStep,
                            y: current.y,
                            type: PointType.REG,
                        },
                        {
                            x: current.x,
                            y: current.y - regGridStep,
                            type: PointType.REG,
                        },
                        {
                            x: current.x,
                            y: current.y + regGridStep,
                            type: PointType.REG,
                        },
                    ].concat(toP.slice(0, -1).filter(toHelperPredicate));
                }
                case PointType.TO_HELPER:
                    return toP;
                default:
                    return [];
            }
        }

        /**
         * Checks whether segment whose endpoints are `current` and `neighbour`
         * intersects node described by `nInfo`.
         *
         * @param current First endpoint of the segment
         * @param neighbour Second endpoint of the segment
         * @param nInfo Node info object, containing position and size of a node
         * @returns Whether the segment intersects the node
         */
        function checkIntersection(current, neighbour, nInfo) {
            if (current.x === neighbour.x) {
                if (
                    !(
                        current.x <= nInfo.position.x - minMargin ||
                        current.x >= nInfo.position.x + nInfo.width + minMargin ||
                        Math.min(
                            current.y,
                            neighbour.y,
                        ) >= nInfo.position.y + nInfo.height + minMargin ||
                        Math.max(current.y, neighbour.y) <= nInfo.position.y - minMargin
                    )
                ) {
                    return true;
                }
            } else if (
                !(
                    current.y <= nInfo.position.y - minMargin ||
                    current.y >= nInfo.position.y + nInfo.height + minMargin ||
                    Math.min(
                        current.x,
                        neighbour.x,
                    ) >= nInfo.position.x + nInfo.width + minMargin ||
                    Math.max(current.x, neighbour.x) <= nInfo.position.x - minMargin
                )
            ) {
                return true;
            }
            return false;
        }

        /**
         * Computes the weight used by A* as the cost of the segment
         * whose endpoints are `current` and `neighbour`.
         *
         * @param current First endpoint of the segment
         * @param neighbour Second endpoint of the segment
         * @returns Weight of the segment
         */
        function computeWeight(current, neighbour) {
            const manhattanDist = aStarConfig.distanceType(
                current.x,
                current.y,
                neighbour.x,
                neighbour.y,
            );

            // Iterate only over nodes from the same zones
            // as processed segment

            const initIdx = Math.floor(current.x / aStarConfig.zoneStep);
            const initIdy = Math.floor(current.y / aStarConfig.zoneStep);

            const finalIdx = Math.floor(neighbour.x / aStarConfig.zoneStep);
            const finalIdy = Math.floor(neighbour.y / aStarConfig.zoneStep);

            for (let i = Math.min(initIdx, finalIdx); i <= Math.max(initIdx, finalIdx); i += 1) {
                for (
                    let j = Math.min(initIdy, finalIdy);
                    j <= Math.max(initIdy, finalIdy);
                    j += 1
                ) {
                    const zoneKey = aStarConfig.zoneInfoKey(i, j);
                    const zList = zoneInfo.get(zoneKey) ?? [];

                    if (zList.some(
                        (nInfo) => checkIntersection(
                            current, neighbour, nInfo,
                        ),
                    )) {
                        return aStarConfig.intersectionFactor * manhattanDist;
                    }
                }
            }
            return manhattanDist;
        }

        const fromPoint = fromPoints.at(-1);
        const toPoint = toPoints.at(-1);

        const pointsEqual = (v1) => (v2) => v1.x === v2.x && v1.y === v2.y;

        const openSet = new PriorityQueue();
        openSet.enqueue(
            fromPoint,
            aStarConfig.distanceType(
                fromPoint.x,
                fromPoint.y,
                toPoint.x,
                toPoint.y,
            ),
        );

        const visited = [fromPoint];
        const pointsToIndex = new Map();
        pointsToIndex.set(key(fromPoint), 0);

        // Map from index of node in `visited` array to its predecessor
        const predecessors = new Map();

        // Map containing the costs of the best paths found so far for each point
        const gScores = new Map();
        gScores.set(0, 0); // fromPoint has index 0 in `visited` and a gScore of 0

        while (!openSet.isEmpty()) {
            const peeked = openSet.peek();
            if (pointsEqual(peeked)(toPoint)) {
                return reconstructPath(pointsToIndex, predecessors, peeked);
            }
            const current = openSet.dequeue();

            const neighbours = getNeighbours(current, fromPoints, toPoints);

            neighbours.forEach((neighbour) => {
                // need to check, whether neighbour is not already at the visited list
                const neighbourKey = key(neighbour);
                const neighbourIndex = pointsToIndex.get(neighbourKey) ?? pointsToIndex.size;

                if (neighbourIndex >= pointsToIndex.size) {
                    visited.push(neighbour);
                    pointsToIndex.set(neighbourKey, neighbourIndex);
                }

                const newGScore = (
                    gScores.get(
                        pointsToIndex.get(key(current)),
                    ) ?? Infinity
                ) + computeWeight(current, neighbour);

                if (newGScore < (gScores.get(neighbourIndex) ?? Infinity)) {
                    predecessors.set(neighbourIndex, current);
                    gScores.set(neighbourIndex, newGScore);

                    // Lower bound of the remaining path length
                    const hScore = aStarConfig.distanceType(
                        neighbour.x,
                        neighbour.y,
                        toPoint.x,
                        toPoint.y,
                    );

                    const newFScore = newGScore + hScore;
                    if (openSet.contains(visited[neighbourIndex])) {
                        openSet.updatePrio(visited[neighbourIndex], newFScore);
                    } else {
                        openSet.enqueue(visited[neighbourIndex], newFScore);
                    }
                }
            });
        }

        // fallback, if path not found for some reason
        const middlePoint = (fromPoint.x + toPoint.x) / 2;
        return [{ x: fromPoint.x, y: fromPoint.y },
            { x: middlePoint, y: fromPoint.y },
            { x: middlePoint, y: toPoint.y },
            { x: toPoint.x, y: toPoint.y },
        ];
    }

    orthogonalRender(x1, y1, x2, y2, connection) {
        const graph = this.viewModel.displayedGraph;
        const nc = new NormalizedConnection(x1, y1, x2, y2, connection);

        if (connection.anchors !== undefined && connection.anchors.length) {
            if (connection.anchors.some((a) => a.legacy)) {
                return this.alternativeOrthogonalAnchorsPath(connection.anchors, nc);
            }
            return this.orthogonalAnchorsPath(connection.anchors, nc);
        }

        const minMargin = 30;
        const middlePointX = (nc.x1 + nc.x2) / 2;
        const middlePointY = (nc.y1 + nc.y2) / 2;
        const intfPad = 20;

        if (connection.to) {
            const shift = this.getShift(nc.from, nc.to, graph);

            if (nc.from.side === 'right' && nc.to.side === 'left') {
                const mid = Math.max(nc.x1, middlePointX) + shift + minMargin;

                const firstTurn = mid < nc.x2 - shift - minMargin ? nc.x1 + shift + minMargin : mid;
                const lastTurn = nc.x2 - shift - minMargin;

                // S connection
                if (
                    mid >= nc.x2 - shift - minMargin &&
                    (firstTurn > nc.x2 - minMargin || lastTurn < nc.x1 - minMargin)
                ) {
                    return [{ x: nc.x1, y: nc.y1 },
                        { x: nc.x1 + intfPad, y: nc.y1 },
                        { x: firstTurn, y: nc.y1 },
                        { x: firstTurn, y: (nc.y1 + nc.y2) / 2 },
                        { x: lastTurn, y: (nc.y1 + nc.y2) / 2 },
                        { x: lastTurn, y: nc.y2 },
                        { x: nc.x2 - intfPad, y: nc.y2 },
                        { x: nc.x2, y: nc.y2 },
                    ];
                }

                // Z connection
                return [{ x: nc.x1, y: nc.y1 },
                    { x: nc.x1 + intfPad, y: nc.y1 },
                    { x: mid, y: nc.y1 },
                    { x: mid, y: nc.y2 },
                    { x: nc.x2 - intfPad, y: nc.y2 },
                    { x: nc.x2, y: nc.y2 },
                ];
            }
            if (nc.from.side === 'left' && nc.to.side === 'right') {
                const mid = Math.max(nc.x2, middlePointX) + shift + minMargin;

                const firstTurn = mid < nc.x1 - shift - minMargin ? nc.x2 + shift + minMargin : mid;
                const lastTurn = nc.x1 - shift - minMargin;

                // S connection
                if (
                    mid >= nc.x1 - shift - minMargin &&
                    (firstTurn > nc.x1 - minMargin || lastTurn < nc.x2 - minMargin)
                ) {
                    return [{ x: nc.x2, y: nc.y2 },
                        { x: nc.x2 + intfPad, y: nc.y2 },
                        { x: firstTurn, y: nc.y2 },
                        { x: firstTurn, y: (nc.y1 + nc.y2) / 2 },
                        { x: lastTurn, y: (nc.y1 + nc.y2) / 2 },
                        { x: lastTurn, y: nc.y1 },
                        { x: nc.x1 - intfPad, y: nc.y1 },
                        { x: nc.x1, y: nc.y1 },
                    ];
                }

                // Z connection
                return [{ x: nc.x2, y: nc.y2 },
                    { x: nc.x2 + intfPad, y: nc.y2 },
                    { x: mid, y: nc.y2 },
                    { x: mid, y: nc.y1 },
                    { x: nc.x1 - intfPad, y: nc.y1 },
                    { x: nc.x1, y: nc.y1 },
                ];
            }
            if (nc.from.side === 'bottom' && nc.to.side === 'right') {
                if (nc.x2 - intfPad > nc.x1) {
                // S connection
                    return [{ x: nc.x1, y: nc.y1 },
                        { x: nc.x1, y: nc.y1 + intfPad },
                        { x: middlePointX, y: nc.y1 + intfPad },
                        { x: nc.x2 + intfPad, y: nc.y1 + intfPad },
                        { x: nc.x2 + intfPad, y: nc.y2 },
                        { x: nc.x2, y: nc.y2 },
                    ];
                }

                const midX = nc.y2 > middlePointY ? nc.x1 : middlePointX + minMargin;

                return [{ x: nc.x1, y: nc.y1 },
                    { x: nc.x1, y: nc.y1 + intfPad },
                    { x: midX, y: nc.y1 + intfPad },
                    { x: midX, y: nc.y2 },
                    { x: nc.x2 + intfPad, y: nc.y2 },
                    { x: nc.x2, y: nc.y2 },
                ];
            }
            if (nc.from.side === 'right' && nc.to.side === 'bottom') {
                if (nc.x1 - intfPad > nc.x2) {
                // S connection
                    return [{ x: nc.x1, y: nc.y1 },
                        { x: nc.x1 + intfPad, y: nc.y1 },
                        { x: nc.x1 + intfPad, y: nc.y2 + intfPad },
                        { x: middlePointX, y: nc.y2 + intfPad },
                        { x: nc.x2, y: nc.y2 + intfPad },
                        { x: nc.x2, y: nc.y2 },
                    ];
                }

                const midX = nc.y1 > middlePointY ? nc.x2 : middlePointX + minMargin;

                return [{ x: nc.x1, y: nc.y1 },
                    { x: nc.x1 + intfPad, y: nc.y1 },
                    { x: midX, y: nc.y1 },
                    { x: midX, y: nc.y2 + intfPad },
                    { x: nc.x2, y: nc.y2 + intfPad },
                    { x: nc.x2, y: nc.y2 },
                ];
            }
            if (nc.from.side === 'left' && nc.to.side === 'bottom') {
                if (nc.x1 - intfPad < nc.x2) {
                // S connection
                    return [{ x: nc.x1, y: nc.y1 },
                        { x: nc.x1 - intfPad, y: nc.y1 },
                        { x: nc.x1 - intfPad, y: nc.y2 + intfPad },
                        { x: middlePointX, y: nc.y2 + intfPad },
                        { x: nc.x2, y: nc.y2 + intfPad },
                        { x: nc.x2, y: nc.y2 },
                    ];
                }

                const midX = nc.y1 > middlePointY ? nc.x2 : middlePointX + minMargin;

                return [{ x: nc.x1, y: nc.y1 },
                    { x: nc.x1 - intfPad, y: nc.y1 },
                    { x: midX, y: nc.y1 },
                    { x: midX, y: nc.y2 + intfPad },
                    { x: nc.x2, y: nc.y2 + intfPad },
                    { x: nc.x2, y: nc.y2 },
                ];
            }
            if (nc.from.side === 'bottom' && nc.to.side === 'left') {
                if (nc.x2 - intfPad < nc.x1) {
                // S connection
                    return [{ x: nc.x1, y: nc.y1 },
                        { x: nc.x1, y: nc.y1 + intfPad },
                        { x: middlePointX, y: nc.y1 + intfPad },
                        { x: nc.x2 - intfPad, y: nc.y1 + intfPad },
                        { x: nc.x2 - intfPad, y: nc.y2 },
                        { x: nc.x2, y: nc.y2 },
                    ];
                }

                const midX = nc.y2 > middlePointY ? nc.x1 : middlePointX + minMargin;

                return [{ x: nc.x1, y: nc.y1 },
                    { x: nc.x1, y: nc.y1 + intfPad },
                    { x: midX, y: nc.y1 + intfPad },
                    { x: midX, y: nc.y2 },
                    { x: nc.x2 - intfPad, y: nc.y2 },
                    { x: nc.x2, y: nc.y2 },
                ];
            }
            if (nc.from.side === 'top' && nc.to.side === 'right') {
                if (nc.x2 - intfPad > nc.x1) {
                    // S connection
                    return [{ x: nc.x1, y: nc.y1 },
                        { x: nc.x1, y: nc.y1 - intfPad },
                        { x: middlePointX, y: nc.y1 - intfPad },
                        { x: nc.x2 + intfPad, y: nc.y1 - intfPad },
                        { x: nc.x2 + intfPad, y: nc.y2 },
                        { x: nc.x2, y: nc.y2 },
                    ];
                }

                const midX = middlePointX + minMargin;

                return [{ x: nc.x1, y: nc.y1 },
                    { x: nc.x1, y: nc.y1 - intfPad },
                    { x: midX, y: nc.y1 - intfPad },
                    { x: midX, y: nc.y2 },
                    { x: nc.x2 + intfPad, y: nc.y2 },
                    { x: nc.x2, y: nc.y2 },
                ];
            }
            if (nc.from.side === 'right' && nc.to.side === 'top') {
                if (nc.x2 - intfPad < nc.x1) {
                    // S connection
                    return [{ x: nc.x1, y: nc.y1 },
                        { x: nc.x1 + intfPad, y: nc.y1 },
                        { x: nc.x1 + intfPad, y: nc.y2 - intfPad },
                        { x: middlePointX, y: nc.y2 - intfPad },
                        { x: nc.x2, y: nc.y2 - intfPad },
                        { x: nc.x2, y: nc.y2 },
                    ];
                }

                const midX = middlePointX + minMargin;

                return [{ x: nc.x1, y: nc.y1 },
                    { x: nc.x1 + intfPad, y: nc.y1 },
                    { x: midX, y: nc.y1 },
                    { x: midX, y: nc.y2 - intfPad },
                    { x: nc.x2, y: nc.y2 - intfPad },
                    { x: nc.x2, y: nc.y2 },
                ];
            }
            if (nc.from.side === 'left' && nc.to.side === 'top') {
                if (nc.x2 - intfPad > nc.x1) {
                    // S connection
                    return [{ x: nc.x1, y: nc.y1 },
                        { x: nc.x1 - intfPad, y: nc.y1 },
                        { x: nc.x1 - intfPad, y: nc.y2 - intfPad },
                        { x: middlePointX, y: nc.y2 - intfPad },
                        { x: nc.x2, y: nc.y2 - intfPad },
                        { x: nc.x2, y: nc.y2 },
                    ];
                }

                const midX = middlePointX + minMargin;

                return [{ x: nc.x1, y: nc.y1 },
                    { x: nc.x1 - intfPad, y: nc.y1 },
                    { x: midX, y: nc.y1 },
                    { x: midX, y: nc.y2 - intfPad },
                    { x: nc.x2, y: nc.y2 - intfPad },
                    { x: nc.x2, y: nc.y2 },
                ];
            }
            if (nc.from.side === 'top' && nc.to.side === 'left') {
                if (nc.x2 - intfPad < nc.x1) {
                    // S connection
                    return [{ x: nc.x1, y: nc.y1 },
                        { x: nc.x1, y: nc.y1 - intfPad },
                        { x: middlePointX, y: nc.y1 - intfPad },
                        { x: nc.x2 - intfPad, y: nc.y1 - intfPad },
                        { x: nc.x2 - intfPad, y: nc.y2 },
                        { x: nc.x2, y: nc.y2 },
                    ];
                }

                const midX = middlePointX + minMargin;

                return [{ x: nc.x1, y: nc.y1 },
                    { x: nc.x1, y: nc.y1 - intfPad },
                    { x: midX, y: nc.y1 - intfPad },
                    { x: midX, y: nc.y2 },
                    { x: nc.x2 - intfPad, y: nc.y2 },
                    { x: nc.x2, y: nc.y2 },
                ];
            }
            if (nc.from.side === 'right' && nc.to.side === 'right') {
                const mid = Math.max(nc.x1, nc.x2, middlePointX) + shift + minMargin;
                return [{ x: nc.x1, y: nc.y1 },
                    { x: nc.x1 + intfPad, y: nc.y1 },
                    { x: mid, y: nc.y1 },
                    { x: mid, y: nc.y2 },
                    { x: nc.x2 + intfPad, y: nc.y2 },
                    { x: nc.x2, y: nc.y2 },
                ];
            }
            if (nc.from.side === 'left' && nc.to.side === 'left') {
                const mid = Math.min(nc.x1, nc.x2, middlePointX) - shift - minMargin;
                return [{ x: nc.x1, y: nc.y1 },
                    { x: nc.x1 - intfPad, y: nc.y1 },
                    { x: mid, y: nc.y1 },
                    { x: mid, y: nc.y2 },
                    { x: nc.x2 - intfPad, y: nc.y2 },
                    { x: nc.x2, y: nc.y2 },
                ];
            }
            if (nc.from.side === 'top' && nc.to.side === 'top') {
                const mid = Math.min(nc.y1, nc.y2, middlePointY) - minMargin;

                return [{ x: nc.x1, y: nc.y1 },
                    { x: nc.x1, y: nc.y1 - intfPad },
                    { x: nc.x1, y: mid },
                    { x: nc.x2, y: mid },
                    { x: nc.x2, y: nc.y2 - intfPad },
                    { x: nc.x2, y: nc.y2 },
                ];
            }
            if (nc.from.side === 'bottom' && nc.to.side === 'bottom') {
                const mid = Math.max(nc.y1, nc.y2, middlePointY) + minMargin;

                return [{ x: nc.x1, y: nc.y1 },
                    { x: nc.x1, y: nc.y1 + intfPad },
                    { x: nc.x1, y: mid },
                    { x: nc.x2, y: mid },
                    { x: nc.x2, y: nc.y2 + intfPad },
                    { x: nc.x2, y: nc.y2 },
                ];
            }
            if (nc.from.side === 'bottom' && nc.to.side === 'top') {
                const mid = Math.min(nc.y2, middlePointY);

                return [{ x: nc.x1, y: nc.y1 },
                    { x: nc.x1, y: nc.y1 + intfPad },
                    { x: middlePointX, y: nc.y1 + intfPad },
                    { x: middlePointX, y: mid - intfPad },
                    { x: nc.x2, y: mid - intfPad },
                    { x: nc.x2, y: nc.y2 - intfPad },
                    { x: nc.x2, y: nc.y2 },
                ];
            }
            if (nc.from.side === 'top' && nc.to.side === 'bottom') {
                const mid = Math.min(nc.y1, middlePointY);

                return [{ x: nc.x1, y: nc.y1 },
                    { x: nc.x1, y: nc.y1 - intfPad },
                    { x: nc.x1, y: mid - intfPad },
                    { x: middlePointX, y: mid - intfPad },
                    { x: middlePointX, y: nc.y2 + intfPad },
                    { x: nc.x2, y: nc.y2 + intfPad },
                    { x: nc.x2, y: nc.y2 },
                ];
            }
        }
        return [{ x: nc.x1, y: nc.y1 },
            { x: middlePointX, y: nc.y1 },
            { x: middlePointX, y: nc.y2 },
            { x: nc.x2, y: nc.y2 },
        ];
    }

    alternativeOrthogonalRender(x1, y1, x2, y2, connection) {
        const graph = this.viewModel.displayedGraph;
        const nc = new NormalizedConnection(x1, y1, x2, y2, connection);

        if (connection.anchors !== undefined && connection.anchors.length) {
            // legacy anchors were not identified by segment index so cannot be used intechangibly.
            if (connection.anchors.some((a) => a.legacy)) {
                return this.alternativeOrthogonalAnchorsPath(connection.anchors, nc);
            }
            return this.orthogonalAnchorsPath(connection.anchors, nc);
        }

        const minMargin = 30;
        const middlePoint = (nc.x1 + nc.x2) / 2;

        if (connection.to) {
            const shift = this.getShift(nc.from, nc.to, graph);
            const maxShift = this.getMaxShift(nc.from, nc.to, graph) || 1;

            const turnDistance = Math.abs(nc.x1 - nc.x2) - 2 * minMargin;
            const minTurnDistance = minMargin * (2 + maxShift / 30);
            const minTurnDistanceViolated = turnDistance < minTurnDistance;
            const maxTurnDistance = minTurnDistance * 2;

            let interpolatedShift;
            if (turnDistance < maxTurnDistance) {
                interpolatedShift = (shift / maxShift) * turnDistance;
            } else {
                interpolatedShift = (shift / maxShift) * maxTurnDistance
                    + (turnDistance - maxTurnDistance) / 2;
            }

            const differentLevels = this.someAboveOrBelow(nc.from, nc.to, graph);

            if (nc.from.side === 'right' && nc.to.side === 'left') {
                // S connection
                const nodeOverlap = nc.x1 > nc.x2;
                if ((differentLevels && minTurnDistanceViolated) || nodeOverlap) {
                    const firstTurn = nc.x1 + minMargin + (shift / maxShift) * minTurnDistance;
                    const lastTurn = nc.x2 - minMargin + (shift / maxShift - 1) * minTurnDistance;
                    return `M ${nc.x1} ${nc.y1}
                    H ${firstTurn}
                    V ${(nc.y1 + nc.y2) / 2}
                    H ${lastTurn}
                    V ${nc.y2}
                    H ${nc.x2}`;
                }

                // Z connection
                const mid = nc.x1 + minMargin + interpolatedShift;
                return `M ${nc.x1} ${nc.y1} H ${mid} V ${nc.y2} H ${nc.x2}`;
            }
            if (nc.from.side === 'right' && nc.to.side === 'bottom') {
                const midPointY = (nc.y1 + nc.y2) / 2;

                const firstTurn = nc.x1 - nc.x2 > 0 ? nc.x1 + minMargin : nc.x2;
                const lastTurn = nc.y2 + minMargin;

                return `M ${nc.x1} ${nc.y1}
                    H ${firstTurn}
                    V ${midPointY}
                    H ${nc.y1 - nc.y2 < 0 ? middlePoint : nc.x2}
                    V ${lastTurn}
                    H ${nc.x2}
                    V ${nc.y2}`;
            }
            if (nc.from.side === 'left' && nc.to.side === 'bottom') {
                const midPointY = (nc.y1 + nc.y2) / 2;

                const firstTurn = nc.x1 - nc.x2 < 0 ? nc.x1 + minMargin : nc.x2;
                const lastTurn = nc.y2 + minMargin;

                return `M ${nc.x1} ${nc.y1}
                    H ${firstTurn}
                    V ${midPointY}
                    H ${nc.y1 - nc.y2 < 0 ? middlePoint : nc.x2}
                    V ${lastTurn}
                    H ${nc.x2}
                    V ${nc.y2}`;
            }
            if (nc.from.side === 'right' && nc.to.side === 'top') {
                const midPointY = (nc.y1 + nc.y2) / 2;

                const firstTurn = nc.x1 - nc.x2 > 0 ? nc.x1 + minMargin : nc.x2;
                const lastTurn = nc.y2 - minMargin;

                return `M ${nc.x1} ${nc.y1}
                    H ${firstTurn}
                    V ${midPointY}
                    H ${nc.y1 - nc.y2 > 0 ? middlePoint : nc.x2}
                    V ${lastTurn}
                    H ${nc.x2}
                    V ${nc.y2}`;
            }
            if (nc.from.side === 'left' && nc.to.side === 'top') {
                const midPointY = (nc.y1 + nc.y2) / 2;

                const firstTurn = nc.x1 - nc.x2 < 0 ? nc.x1 - minMargin : nc.x2;
                const lastTurn = nc.y2 - minMargin;

                return `M ${nc.x1} ${nc.y1}
                    H ${firstTurn}
                    V ${midPointY}
                    H ${nc.y1 - nc.y2 > 0 ? middlePoint : nc.x2}
                    V ${lastTurn}
                    H ${nc.x2}
                    V ${nc.y2}`;
            }
            if (nc.from.side === 'left' && nc.to.side === 'right') {
                // S connection
                const nodeOverlap = nc.x2 > nc.x1;
                if ((differentLevels && minTurnDistanceViolated) || nodeOverlap) {
                    const firstTurn = nc.x2 + minMargin + (shift / maxShift) * minTurnDistance;
                    const lastTurn = nc.x1 - minMargin + (shift / maxShift - 1) * minTurnDistance;
                    return `M ${nc.x2} ${nc.y2}
                    H ${firstTurn}
                    V ${(nc.y1 + nc.y2) / 2}
                    H ${lastTurn}
                    V ${nc.y1}
                    H ${nc.x1}`;
                }

                // Z connection
                const mid = nc.x2 + minMargin + interpolatedShift;
                return `M ${nc.x2} ${nc.y2} H ${mid} V ${nc.y1} H ${nc.x1}`;
            }
            if (nc.from.side === 'right' && nc.to.side === 'right') {
                return `M ${nc.x1} ${nc.y1} H ${
                    Math.max(nc.x1, nc.x2, middlePoint) + shift + minMargin
                } V ${nc.y2} H ${nc.x2}`;
            }
            if (nc.from.side === 'left' && nc.to.side === 'left') {
                return `M ${nc.x1} ${nc.y1} H ${
                    Math.min(nc.x1, nc.x2, middlePoint) - shift - minMargin
                } V ${nc.y2} H ${nc.x2}`;
            }
            if (nc.from.side === 'top' && nc.to.side === 'top') {
                const midPointY = (nc.y1 + nc.y2) / 2;

                return `M ${nc.x1} ${nc.y1} V ${
                    Math.min(nc.y1, nc.y2, midPointY) - minMargin
                } H ${nc.x2} V ${nc.y2}`;
            }
            if (nc.from.side === 'bottom' && nc.to.side === 'bottom') {
                const midPointY = (nc.y1 + nc.y2) / 2;

                return `M ${nc.x1} ${nc.y1} V ${
                    Math.max(nc.y1, nc.y2, midPointY) + minMargin
                } H ${nc.x2} V ${nc.y2}`;
            }
        }
        return `M ${nc.x1} ${nc.y1} H ${middlePoint} V ${nc.y2} H ${nc.x2}`;
    }

    alternativeOrthogonalRenderLoopback(x1, y1, x2, y2, connection) {
        return this.orthogonalRenderLoopback(x1, y1, x2, y2, connection);
    }

    supportSwitchableInterfaces() {
        return ['orthogonal', 'alternativeOrthogonal', 'straight', 'aStar'].includes(this.style);
    }

    connectionRefresh(connection) {
        if (this.switchableInterfaces && this.supportSwitchableInterfaces()) {
            const from = getDomElements(connection.from);
            const to = getDomElements(connection.to);

            const [x1, y1] = getPortCoordinates(from);
            const [x2, y2] = getPortCoordinates(to);

            this.switchableConnectionRefresh(x1, y1, x2, y2, connection);
        }
    }

    getClosestAnchor(anchors, x, y) {
        if (!anchors?.length) {
            return undefined;
        }

        const closestAnchor = anchors.map((anch) => [
            anch, Math.abs(x - anch.x) + Math.abs(y - anch.y),
        ]).sort((a, b) => a[1] - b[1]).at(0)[0];

        return closestAnchor;
    }

    switchableConnectionRefreshAnchors(normalizedConnection, anchors, graph) {
        const nc = normalizedConnection;

        const fromNode = graph.nodes.filter((node) => nc.from.nodeId === node.id)[0];
        const fromNodeInputs = Object.values(fromNode.inputs)
            .filter((ni) => !ni.hidden);
        const fromNodeOutputs = Object.values(fromNode.outputs)
            .filter((no) => !no.hidden);
        const toNode = graph.nodes.filter((node) => nc.to.nodeId === node.id)[0];
        const toNodeInputs = Object.values(toNode.inputs).filter((ni) => !ni.hidden && ni.port);
        const toNodeOutputs = Object.values(toNode.outputs).filter((no) => !no.hidden && no.port);

        const fromNodeLefts = [...fromNodeInputs, ...fromNodeOutputs].filter((intf) => intf.side === 'left');
        const fromNodeRights = [...fromNodeInputs, ...fromNodeOutputs].filter((intf) => intf.side === 'right');

        const toNodeLefts = [...toNodeInputs, ...toNodeOutputs].filter((intf) => intf.side === 'left');
        const toNodeRights = [...toNodeInputs, ...toNodeOutputs].filter((intf) => intf.side === 'right');

        const nextFromNodeLeftIndex = getIndex(fromNodeLefts.map((intf) => intf.sidePosition));
        const nextFromNodeRightIndex = getIndex(fromNodeRights.map((intf) => intf.sidePosition));

        const nextToNodeLeftIndex = getIndex(toNodeLefts.map((intf) => intf.sidePosition));
        const nextToNodeRightIndex = getIndex(toNodeRights.map((intf) => intf.sidePosition));

        const toLastSide = nc.to.side;
        const fromLastSide = nc.from.side;

        if (nc.to) {
            const x = nc.x2;
            const y = nc.y2;
            // get anchor closest to interface
            const anchorIndexes = anchors.map((anch) => anch.index);

            const maxIndex = Math.max(...anchorIndexes);
            const anchorsToCheck = anchors.filter((anch) => anch.index === maxIndex);
            const closestAnchor = this.getClosestAnchor(anchorsToCheck, x, y);

            if (x < closestAnchor.x) {
                nc.to.side = 'right';

                if (toLastSide !== 'right') {
                    nc.to.sidePosition = nextToNodeRightIndex;
                }
            } else {
                nc.to.side = 'left';

                if (toLastSide !== 'left') {
                    nc.to.sidePosition = nextToNodeLeftIndex;
                }
            }
        }

        if (nc.from) {
            const x = nc.x1;
            const y = nc.y1;
            // get anchor closest to interface
            const anchorIndexes = anchors.map((anch) => anch.index);

            const minIndex = Math.min(...anchorIndexes);
            const anchorsToCheck = anchors.filter((anch) => anch.index === minIndex);
            const closestAnchor = this.getClosestAnchor(anchorsToCheck, x, y);

            if (x < closestAnchor.x) {
                nc.from.side = 'right';

                if (fromLastSide !== 'right') {
                    nc.from.sidePosition = nextFromNodeRightIndex;
                }
            } else {
                nc.from.side = 'left';
                if (fromLastSide !== 'left') {
                    nc.from.sidePosition = nextFromNodeLeftIndex;
                }
            }
        }
    }

    switchableConnectionRefresh(x1, y1, x2, y2, connection) {
        const graph = this.viewModel.displayedGraph;
        const nc = new NormalizedConnection(x1, y1, x2, y2, connection);

        if (connection.anchors !== undefined && connection.anchors.length) {
            // We should take anchors into account.
            this.switchableConnectionRefreshAnchors(nc, connection.anchors, graph);
            return;
        }

        // ignore interfaces with top and bottom sides
        if (nc.from.side === 'top' || nc.from.side === 'bottom' ||
            nc.to.side === 'top' || nc.to.side === 'bottom'
        ) {
            return;
        }

        const minMargin = 30 * graph.scaling;

        const fromNode = graph.nodes.filter((node) => nc.from.nodeId === node.id)[0];
        const fromNodePosInterfaces = graph.editor.getNodeInterfacePositions(fromNode.type);
        const fromNodeWidth = fromNode.width;
        const fromNodeInputs = Object.values(fromNode.inputs)
            .filter((ni) => !ni.hidden);
        const fromNodeOutputs = Object.values(fromNode.outputs)
            .filter((no) => !no.hidden);
        const toNode = graph.nodes.filter((node) => nc.to.nodeId === node.id)[0];
        const toNodePosInterfaces = graph.editor.getNodeInterfacePositions(toNode.type);
        const toNodeWidth = toNode.width;
        const toNodeInputs = Object.values(toNode.inputs).filter((ni) => !ni.hidden && ni.port);
        const toNodeOutputs = Object.values(toNode.outputs).filter((no) => !no.hidden && no.port);

        const fromNodeLefts = [...fromNodeInputs, ...fromNodeOutputs].filter((intf) => intf.side === 'left');
        const fromNodeRights = [...fromNodeInputs, ...fromNodeOutputs].filter((intf) => intf.side === 'right');

        const toNodeLefts = [...toNodeInputs, ...toNodeOutputs].filter((intf) => intf.side === 'left');
        const toNodeRights = [...toNodeInputs, ...toNodeOutputs].filter((intf) => intf.side === 'right');

        const nextFromNodeLeftIndex = getIndex(fromNodeLefts.map((intf) => intf.sidePosition));
        const nextFromNodeRightIndex = getIndex(fromNodeRights.map((intf) => intf.sidePosition));

        const nextToNodeLeftIndex = getIndex(toNodeLefts.map((intf) => intf.sidePosition));
        const nextToNodeRightIndex = getIndex(toNodeRights.map((intf) => intf.sidePosition));

        // ignore positioned interfaces
        if (Object.hasOwn(fromNodePosInterfaces, nc.from.name)
            || Object.hasOwn(toNodePosInterfaces, nc.to.name)) {
            return;
        }

        const lastFromSide = nc.from.side;
        const lastToSide = nc.to.side;

        if (connection.to) {
            if (nc.from.side === 'right' && nc.to.side === 'left') {
                const dx = nc.x1 - nc.x2;

                if (dx > minMargin * 2 + fromNodeWidth) {
                    nc.from.side = 'left';
                    nc.from.sidePosition = nextFromNodeLeftIndex;
                    nc.to.side = 'left';
                    nc.to.sidePosition = nextToNodeLeftIndex;
                }
            } else if (nc.from.side === 'left' && nc.to.side === 'right') {
                const dx = nc.x1 - nc.x2;

                if (dx < -minMargin * 2 - fromNodeWidth) {
                    nc.to.side = 'right';
                    nc.to.sidePosition = nextToNodeRightIndex;
                    nc.from.side = 'right';
                    nc.from.sidePosition = nextFromNodeRightIndex;
                }
            }

            if (nc.from.side === 'left' && nc.to.side === 'left') {
                const dx = nc.x1 - nc.x2;
                if (dx < -minMargin * 2 - fromNodeWidth) {
                    nc.from.side = 'right';
                    nc.from.sidePosition = nextFromNodeRightIndex;
                    nc.to.side = 'left';
                    nc.to.sidePosition = nextToNodeLeftIndex;
                } else if (dx > minMargin * 2 + fromNodeWidth + toNodeWidth) {
                    nc.from.side = 'left';
                    nc.from.sidePosition = nextFromNodeLeftIndex;
                    nc.to.side = 'right';
                    nc.to.sidePosition = nextToNodeRightIndex;
                }
            } else if (nc.from.side === 'right' && nc.to.side === 'right') {
                const dx = nc.x1 - nc.x2;
                if (dx > minMargin * 2 + fromNodeWidth) {
                    nc.from.side = 'right';
                    nc.from.sidePosition = nextFromNodeRightIndex;
                    nc.to.side = 'left';
                    nc.to.sidePosition = nextToNodeLeftIndex;
                } else if (dx < -minMargin * 2 - fromNodeWidth + toNodeWidth) {
                    nc.from.side = 'right';
                    nc.from.sidePosition = nextFromNodeRightIndex;
                    nc.to.side = 'left';
                    nc.to.sidePosition = nextToNodeLeftIndex;
                }
            }

            if (lastFromSide !== nc.from.side || lastToSide !== nc.to.side) {
                this.aStarCache.delete(connection.id);
            }
        }
    }

    straightRender(x1, y1, x2, y2, connection) {
        if (connection.anchors?.some((a) => a.legacy)) {
            return [{ x: x1, y: y1 }]
                .concat(connection.anchors)
                .concat([{ x: x2, y: y2 }]);
        }
        const path = [{ x: x1, y: y1 }];
        (connection.anchors ?? []).forEach((anchor) => {
            path.splice(anchor.index + 1, 0, { x: anchor.x, y: anchor.y });
        });
        return path.concat([{ x: x2, y: y2 }]);
    }

    straightRenderLoopback(x1, y1, x2, y2, connection) {
        return this.curvedRenderLoopback(x1, y1, x2, y2, connection);
    }

    constructor(viewModel, style, randomizedOffset = false) {
        this.viewModel = viewModel;
        this.style = style;
        this.randomizedOffset = randomizedOffset;
    }

    /**
     * Chooses the render method based on active style and connection characteristic
     *
     * @param x1 X coordinate of from interface
     * @param y1 Y coordinate of from interface
     * @param x2 X coordinate of to interface
     * @param y2 Y coordinate of to interface
     * @param connection BaklavaJS-defined connection to render
     * @returns String defining connection path in SVG format
     */
    render(x1, y1, x2, y2, connection, draggedNode) {
        // Invalid connection
        if ((x1 === x2 && y1 === y2) &&
            !this.isLoopback(connection)) {
            return undefined;
        }
        const loopback = this.isLoopback(connection) ? 'Loopback' : '';
        if (
            this.style === 'aStar' &&
            draggedNode &&
            (
                connection.from.nodeId === draggedNode.id ||
                connection.to.nodeId === draggedNode.id
            )
        ) {
            return this[`orthogonalRender${loopback}`](x1, y1, x2, y2, connection);
        }
        return this[`${this.style}Render${loopback}`](x1, y1, x2, y2, connection);
    }

    /**
     * Tests whether the connection is loopback (connects the node with itself)
     *
     * @param connection BaklavaJS-defined connection to test
     * @returns True if connection is loopback.
     */
    isLoopback(connection) {
        // Temporary connections that are not connected to any output (.to is undefined)
        // are not loopback
        return !!connection.to && connection.from.nodeId === connection.to.nodeId;
    }

    /**
     * Checks whether the current style supports anchors.
     *
     * @returns True if style supports anchors.
     */
    supportsAnchors() {
        return ['orthogonal', 'alternativeOrthogonal', 'straight'].includes(this.style);
    }
}
