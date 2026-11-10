<!--
Copyright (c) 2022-2023 Antmicro <www.antmicro.com>

SPDX-License-Identifier: Apache-2.0
-->

<template>
    <div class="rectangle-selection-border" :style="styles">
        <div class="rectangle-selection">

        </div>
    </div>
</template>

<script>
import { computed, defineComponent, ref } from 'vue'; // eslint-disable-line object-curly-newline

export default defineComponent({
    setup() {
        const selecting = ref(false);
        const selectionBegin = ref({ x: 0, y: 0 });
        const selectionEnd = ref({ x: 0, y: 0 });
        const boundingRect = computed(() => ({
            xBegin: selectionBegin?.value.x < selectionEnd?.value.x ?
                selectionBegin?.value.x : selectionEnd?.value.x,

            yBegin: selectionBegin?.value.y < selectionEnd?.value.y ?
                selectionBegin?.value.y : selectionEnd?.value.y,

            xEnd: selectionBegin?.value.x >= selectionEnd?.value.x ?
                selectionBegin?.value.x : selectionEnd?.value.x,

            yEnd: selectionBegin?.value.y >= selectionEnd?.value.y ?
                selectionBegin?.value.y : selectionEnd?.value.y,
        }));

        const styles = computed(() => ({
            position: 'absolute',
            visibility: `${selecting.value ? 'visible' : 'hidden'}`,
            top: `${boundingRect.value.yBegin}px`,
            left: `${boundingRect.value.xBegin}px`,
            width: `${Math.abs(boundingRect.value.xEnd - boundingRect.value.xBegin) ?? 0}px`,
            height: `${Math.abs(boundingRect.value.yEnd - boundingRect.value.yBegin) ?? 0}px`,
        }));

        const onPointerDown = (ev) => {
            selecting.value = true;
            selectionBegin.value = { x: ev.pageX, y: ev.pageY };
            selectionEnd.value = { x: ev.pageX, y: ev.pageY };
        };

        const onPointerMove = (ev) => {
            if (selecting.value) {
                selectionEnd.value = { x: ev.pageX, y: ev.pageY };
            }
        };

        const onPointerUp = () => {
            selecting.value = false;

            selectionBegin.value = { x: 0, y: 0 };
            selectionEnd.value = { x: 0, y: 0 };
        };

        return {
            styles,
            onPointerDown,
            onPointerMove,
            onPointerUp,
            selecting,
            boundingRect,
        };
    },
});

</script>
