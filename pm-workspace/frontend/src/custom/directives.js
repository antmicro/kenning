/*
 * Copyright (c) 2022-2023 Antmicro <www.antmicro.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

const elListeners = {};

const longPressFactory = (pointerDownAction, pointerUpAction) => ({
    beforeMount: (el, binding) => {
        const data = {
            timeout: null,
            newEvent: null,
            longPress: false,
            moveEvents: 0,
        };
        elListeners[el] = {
            pointerdown: (ev) => {
                if (ev.pointerType === 'touch' && data.timeout === null) {
                    data.moveEvents = 0;
                    data.longPress = false;
                    pointerDownAction(el, binding, ev, data);
                    data.timeout = setTimeout(() => { data.longPress = true; }, binding.arg ?? 500);
                }
            },
            pointermove: () => { data.moveEvents += 1; },
            pointerup: (ev) => {
                if (ev.pointerType !== 'touch') return;
                // If long press is valid send right click to element
                if (data.longPress && data.moveEvents < 10) {
                    pointerUpAction(el, binding, ev, data);
                }
                if (data.timeout) {
                    clearTimeout(data.timeout);
                    data.timeout = null;
                }
            },
        };
        el.addEventListener('pointerdown', elListeners[el].pointerdown);
        el.addEventListener('pointermove', elListeners[el].pointermove);
        el.addEventListener('pointerup', elListeners[el].pointerup);
    },
    unmounted: (el) => {
        el.removeEventListener('pointerdown', elListeners[el].pointerdown);
        el.removeEventListener('pointermove', elListeners[el].pointermove);
        el.removeEventListener('pointerup', elListeners[el].pointerup);
    },
});

export const longPress = longPressFactory(
    // eslint-disable-next-line no-param-reassign
    (_el, _binding, ev, data) => { data.newEvent = ev; },
    (_el, binding, _ev, data) => { setTimeout(() => binding.value(data.newEvent), 50); },
);

export const longPressToRight = longPressFactory(
    (_el, _binding, ev, data) => {
        // eslint-disable-next-line no-param-reassign
        data.newEvent = new PointerEvent('pointerdown', {
            button: 2, // Right click
            clientX: ev.clientX,
            clientY: ev.clientY,
            screenX: ev.screenX,
            screenY: ev.screenY,
            pointerType: 'mouse',
            relatedTarget: ev.relatedTarget,
        });
    },
    (el, _binding, _ev, data) => { setTimeout(() => { el.dispatchEvent(data.newEvent); }, 50); },
);
