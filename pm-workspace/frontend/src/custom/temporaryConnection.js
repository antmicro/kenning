/*
 * Copyright (c) 2022-2023 Antmicro <www.antmicro.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 */

/**
 * Module used to handle user interaction with connections and interfaces.
 * It is heavily baklava implementation based.
 */

import { provide, inject, ref } from 'vue';
import { useGraph } from '@baklavajs/renderer-vue';

/**
 * This enum class is based on baklavajs enum `TemporaryConnectionState`
 * and is compatible with it as it is unimportable in baklavajs sources
 */
export const TemporaryConnectionState = {
    NONE: 0,
    ALLOWED: 1,
    FORBIDDEN: 2,
};

export function provideTemporaryConnection() {
    const { graph } = useGraph();

    const temporaryConnection = ref(null);
    const hoveringOver = ref(null);
    const hoveredOut = ref(false);
    let hoveringOverElement = null;

    const onMouseMove = (ev) => {
        if (temporaryConnection.value) {
            const element = document.elementFromPoint(ev.clientX, ev.clientY);
            // Touch does not support hover, check if event pointing on interface
            if (ev.pointerType === 'touch') {
                // Hover out, trigger pointerout
                if (hoveringOverElement && hoveringOverElement !== element) {
                    hoveredOut.value = true;
                    hoveringOverElement.dispatchEvent(new PointerEvent('pointerout'));
                    hoveringOverElement = null;
                }
                // Hover over port, trigger pointerover
                if (element && (element.classList.contains('__port') || element.classList.contains('__port-bus'))) {
                    element.dispatchEvent(new PointerEvent('pointerover'));
                    hoveringOverElement = element;
                }
                temporaryConnection.value.mx =
                    ev.clientX / graph.value.scaling - graph.value.panning.x;
                temporaryConnection.value.my =
                    ev.clientY / graph.value.scaling - graph.value.panning.y;
            } else {
                // eslint-disable-next-line no-bitwise
                hoveredOut.value |= hoveringOverElement && hoveringOverElement !== element;
                temporaryConnection.value.mx =
                    ev.offsetX / graph.value.scaling - graph.value.panning.x;
                temporaryConnection.value.my =
                    ev.offsetY / graph.value.scaling - graph.value.panning.y;
            }
        }
    };

    const onMouseDown = (ev) => {
        hoveringOverElement = ev.target;
        if (hoveringOver.value) {
            temporaryConnection.value = {
                status: TemporaryConnectionState.NONE,
                from: hoveringOver.value,
                to: hoveringOver.value,
            };

            temporaryConnection.value.mx = undefined;
            temporaryConnection.value.my = undefined;
            hoveredOut.value = false;
        }
    };

    const onMouseUp = (ev) => {
        if (temporaryConnection.value && hoveringOver.value && hoveredOut.value) {
            graph.value.addConnection(
                temporaryConnection.value.from, temporaryConnection.value.to, ev.offsetY);
        }
        temporaryConnection.value = null;
    };

    const hoveredOver = (ni) => {
        hoveringOver.value = ni ?? null;
        if (ni && temporaryConnection.value && hoveredOut.value) {
            temporaryConnection.value.to = ni;
            const checkConnectionResult = graph.value.checkConnection(
                temporaryConnection.value.from,
                temporaryConnection.value.to,
            );
            temporaryConnection.value.status = checkConnectionResult.connectionAllowed
                ? TemporaryConnectionState.ALLOWED
                : TemporaryConnectionState.FORBIDDEN;

            if (checkConnectionResult.connectionAllowed) {
                const ids = checkConnectionResult.connectionsInDanger.map((c) => c.id);
                graph.value.connections.forEach((c) => {
                    if (ids.includes(c.id)) {
                        c.isInDanger = true; // eslint-disable-line no-param-reassign
                    }
                });
            }
        } else if (!ni && temporaryConnection.value) {
            temporaryConnection.value.to = undefined;
            temporaryConnection.value.status = TemporaryConnectionState.NONE;
            graph.value.connections.forEach((c) => {
                c.isInDanger = false; // eslint-disable-line no-param-reassign
            });
        }
    };

    provide('temporaryConnection', {
        temporaryConnection,
        hoveredOver,
    });

    return {
        temporaryConnection,
        render: hoveredOut,
        hoveredOver,
        onMouseMove,
        onMouseDown,
        onMouseUp,
    };
}
export function useTemporaryConnection() {
    const temporaryConnection = inject('temporaryConnection');
    if (!temporaryConnection) {
        throw new Error('useTemporaryConnection must be used within a BaklavaEditor');
    }
    return temporaryConnection;
}
