<!--
Copyright (c) 2022-2023 Antmicro <www.antmicro.com>

SPDX-License-Identifier: Apache-2.0
-->

<template>
    <g
        @pointerdown.left="startDrag"
        @pointerdown.right.exact="rightclickCallback"
        class="anchor"
    >
        <circle
            :style="styles"
        />
    </g>
</template>

<script lang="ts">
import { computed, ref, defineComponent } from 'vue';
import { useGraph, useViewModel } from '@baklavajs/renderer-vue';
import useDragMove from '../custom/useDragMove';

export default defineComponent({
    props: {
        position: {
            type: Object as () => { x: number, y: number },
            required: true,
        },
        rightclickCallback: {
            required: true,
            type: Function,
        },
        connection: { required: true },
        index: {
            required: true,
            type: Number,
        },
    },
    setup(props) {
        // any definition is an ad-hoc solution as we don't have our graph definition
        const { graph } = useGraph() as { graph: any };
        const { viewModel } = useViewModel() as { viewModel: any };
        const radius = 7.5;
        const styles = computed(() => ({
            cx: `${props.position.x}px`,
            cy: `${props.position.y}px`,
            r: `${radius}px`,
        }));

        const dragMove = useDragMove(ref(props.position));

        const stopDrag = () => {
            document.removeEventListener('pointermove', dragMove.onPointerMove);
            document.removeEventListener('pointerup', stopDrag);
            // eslint-disable-next-line max-len
            graph.value.editAnchor(props.connection, props.index, props.position, dragMove.draggingStartPosition.value);
            dragMove.onPointerUp();
            graph.value.refreshConnections();
        };

        const startDrag = (ev: PointerEvent) => {
            if (viewModel.value.editor.readonly) return;
            dragMove.onPointerDown(ev);
            document.addEventListener('pointermove', dragMove.onPointerMove);
            document.addEventListener('pointerup', stopDrag);
        };

        return {
            styles,
            startDrag,
        };
    },
});
</script>

<style lang="scss">
.anchor {
    pointer-events: all;

    & > circle {
        fill: $green;
    }

    & > circle:hover {
        fill: $red;
    }
}
</style>
