/*
 * Copyright (c) 2022-2026 Antmicro <www.antmicro.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

import { computed } from 'vue';
import { useGraph } from '@baklavajs/renderer-vue';
import useDragMove from './useDragMove';

export default function usePanZoom() {
    const { graph } = useGraph();

    // State needed for pinch-zoom
    let pointerCache = [];
    let prevDiff = -1;
    let midpoint = { x: 0, y: 0 };

    const isGraphSmall = () => graph.value.size().graphWidth <= 500 ||
            graph.value.size().graphHeight <= 500;

    // The limit for zooming that does not allow for zooming if
    // `zoomLimit` number of graphs would fit into the editor
    // vertically or horizontally.
    let zoomLimit = 4;

    const panningRef = computed(() => graph.value.panning);
    const dragMove = useDragMove(panningRef);

    const applyZoom = (centerX, centerY, newScale) => {
        if (newScale <= 0) {
            return;
        }

        if (isGraphSmall()) {
            zoomLimit = 8;
        }

        const currentPoint = [
            centerX / graph.value.scaling - graph.value.panning.x,
            centerY / graph.value.scaling - graph.value.panning.y,
        ];
        const newPoint = [
            centerX / newScale - graph.value.panning.x,
            centerY / newScale - graph.value.panning.y,
        ];
        const diff = [newPoint[0] - currentPoint[0], newPoint[1] - currentPoint[1]];

        const editorHeight = window.innerHeight;
        const editorWidth = window.innerWidth;

        const allowZoomOut =
            zoomLimit * graph.value.size().graphWidth > editorWidth / newScale ||
            zoomLimit * graph.value.size().graphHeight > editorHeight / newScale ||
            (
                newScale > graph.value.scaling &&
                graph.value.size().graphHeight !== -Infinity
            );

        if (allowZoomOut) {
            graph.value.scaling = newScale;
            graph.value.panning.x += diff[0];
            graph.value.panning.y += diff[1];
        }
    };

    const calculateScale = (scrollDelta) => {
        // For smaller graphs, allow the larger scale than for larger graphs.
        const upperLimitOfScale = isGraphSmall() ? 2 : 1.5;

        const newScale = graph.value.scaling * (1 - scrollDelta / 3000);
        return Math.min(newScale, upperLimitOfScale);
    };

    const onMouseWheel = (() => {
        let timeout = null; // throttle timer
        let accumulatedDelta = 0;

        return (ev) => {
            if (ev.target.type === 'textarea' && ev.target.className === 'baklava-input') return;

            ev.preventDefault();

            let scrollAmount = ev.deltaY;
            if (ev.deltaMode === 1) scrollAmount *= 32; // Firefox fix, multiplier is trial & error

            accumulatedDelta += scrollAmount;

            if (!timeout) {
                timeout = setTimeout(() => {
                    // Limit the zooming
                    const newScale = calculateScale(accumulatedDelta);
                    applyZoom(ev.clientX, ev.clientY, newScale);

                    accumulatedDelta = 0;
                    timeout = null;
                }, 10); // to batch scroll events
            }
        };
    })();

    const getCoordsFromCache = () => ({
        ax: pointerCache[0].clientX,
        ay: pointerCache[0].clientY,
        bx: pointerCache[1].clientX,
        by: pointerCache[1].clientY,
    });

    const onPointerDown = (ev) => {
        dragMove.onPointerDown(ev);

        if (pointerCache.length <= 1) {
            pointerCache.push(ev);
        }

        if (pointerCache.length === 2) {
            const { ax, ay, bx, by } = getCoordsFromCache(); // eslint-disable-line object-curly-newline,max-len
            const dx = ax - bx;
            const dy = ay - by;
            prevDiff = Math.sqrt(dx * dx + dy * dy);

            midpoint = {
                x: ax + (bx - ax) / 2,
                y: ay + (by - ay) / 2,
            };
        }
    };

    const onPointerMove = (() => {
        let timeout = null;
        let accumulatedDX = 0;
        let accumulatedDY = 0;
        let lastEvent = null;
        return (ev) => {
            if (pointerCache.length === 2) {
                for (let i = 0; i < pointerCache.length; i += 1) {
                    if (ev.pointerId === pointerCache[i].pointerId) {
                        pointerCache[i] = ev;
                        break;
                    }
                }

                const {
                    ax, ay, bx, by,
                } = getCoordsFromCache();
                const dx = ax - bx;
                const dy = ay - by;
                const curDiff = Math.sqrt(dx * dx + dy * dy);

                if (prevDiff > 0) {
                    if (!timeout) {
                        timeout = setTimeout(() => {
                            const newScale = graph.value.scaling * (1 + (curDiff - prevDiff) / 500);
                            applyZoom(midpoint.x, midpoint.y, newScale);

                            prevDiff = curDiff;
                            timeout = null;
                        }, 10);
                    }
                }
            } else if (pointerCache.length === 1) {
                if (lastEvent) {
                    accumulatedDX += ev.pageX - lastEvent.pageX;
                    accumulatedDY += ev.pageY - lastEvent.pageY;
                }

                lastEvent = ev;

                if (!timeout) {
                    timeout = setTimeout(() => {
                        if (lastEvent) {
                            const syntheticEvent = {
                                ...lastEvent,
                                pageX: lastEvent.pageX - accumulatedDX,
                                pageY: lastEvent.pageY - accumulatedDY,
                            };

                            dragMove.onPointerMove(syntheticEvent);

                            accumulatedDX = 0;
                            accumulatedDY = 0;
                            lastEvent = null;
                        }

                        timeout = null;
                    }, 16);
                }
            } else {
                dragMove.onPointerMove(ev);
            }
        };
    })();

    const onPointerUp = (ev) => {
        dragMove.onPointerUp();

        pointerCache = pointerCache.filter((p) => p.pointerId !== ev.pointerId);
        prevDiff = -1;
    };

    const onZoomIn = () => {
        // Limit the zooming to 1.5x of the original size
        const newScale = Math.min(graph.value.scaling * 1.2, 1.5);
        applyZoom(window.innerWidth / 2, window.innerHeight / 2, newScale);
    };

    const onZoomOut = () => {
        applyZoom(window.innerWidth / 2, window.innerHeight / 2, graph.value.scaling * 0.8);
    };

    return {
        ...dragMove,
        onPointerDown,
        onPointerMove,
        onPointerUp,
        onMouseWheel,
        onZoomIn,
        onZoomOut,
    };
}
