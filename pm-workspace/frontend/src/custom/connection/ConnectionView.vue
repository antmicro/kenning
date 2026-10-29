<!--
Copyright (c) 2022-2026 Antmicro <www.antmicro.com>

SPDX-License-Identifier: Apache-2.0
-->

<!--
Component defining connection between two different nodes
Inherits from baklavajs/renderer-vue/src/connection/ConnectionView.vue
-->

<template>
    <g v-if="Array.isArray(parsedNewD)">
        <path :d="parsedNewD" class="connection-wrapper baklava-connection"></path>
        <!-- The connection wrapper is rendered twice, once for
            easy highlight detection, and once for creating anchor points -->

        <!-- eslint-disable vue/valid-v-for -->
        <g
            v-for="(d, index) in parsedNewD"
            @pointerdown.left.exact="onMouseDown"
            @pointerdown.right.exact="(ev) => onHoldStart(ev, index)"
            @pointerdown="(ev) => { if(ev.pointerType === 'touch') onMouseDown(ev) }"
            @pointerdown.left.ctrl.exact="(ev) => onMouseCtrlDown(ev, index)"
        >
        <path :d="d" class="connection-wrapper baklava-connection"></path>
        <path :d="d" class="baklava-connection" :class="cssClasses" :style="style"></path>
        </g>
        <template v-if="hasAnchors">
            <template v-if="hover || !editorManager.baklavaView.settings.hideAnchors">
                <Anchor
                    v-for="(anchor, index) in connection.anchors"
                    :key="anchor.id"
                    :position="anchor"
                    :connection="connection"
                    :index="index"
                    :rightclickCallback="() => removeAnchor(index)"
                />
            </template>
        </template>
    </g>
    <g
        v-else
        @pointerdown.left.exact="onMouseDown"
        @pointerdown="(ev) => { if(ev.pointerType === 'touch') onMouseDown(ev) }"
        @pointerdown.left.ctrl.exact="(ev) => onMouseCtrlDown(ev, 0)"
    >
        <path :d="parsedNewD" class="connection-wrapper baklava-connection"></path>
        <path :d="parsedNewD" class="baklava-connection" :class="cssClasses" :style="style"></path>
    </g>
</template>

<script>
import {
    defineComponent, computed, toRef, ref,
} from 'vue';
import { Components, useGraph, useViewModel } from '@baklavajs/renderer-vue';
import doubleClick from '../../core/doubleClick';
import Anchor from '../../components/Anchor.vue';
import EditorManager from '../../core/EditorManager';

/* eslint-disable vue/no-mutating-props,no-param-reassign */
export default defineComponent({
    extends: Components.Connection,
    props: {
        isHighlighted: { default: false },
        connection: { required: true },
        draggedNode: { required: true },
        hover: { default: false },
    },
    components: { Anchor },
    setup(props) {
        const { classes } = Components.Connection.setup(props);
        const { graph } = useGraph();
        const { viewModel } = useViewModel();
        const { interfaceTypes } = viewModel.value;
        const editorManager = EditorManager.getEditorManagerInstance();
        const hover = toRef(props, 'hover');

        const connectionStyle = interfaceTypes.getConnectionStyle(
            props.connection.from,
            props.connection.to,
        );

        const cssClasses = computed(() => ({
            ...classes.value,
            '--hover': props.isHighlighted || hover.value,
            '--dashed': connectionStyle.interfaceConnectionPattern === 'dashed',
            '--dotted': connectionStyle.interfaceConnectionPattern === 'dotted',
        }));

        const style = computed(() => ({
            '--color': connectionStyle.interfaceConnectionColor,
        }));

        const draggingOffset = ref({});
        const draggingStartPoint = ref({});
        const curAnchor = ref({});
        const isHolding = ref(false);

        const holdMove = (ev) => {
            // eslint-disable-next-line no-bitwise
            if (isHolding.value && (ev.buttons & 2)) {
                const dx = ev.pageX - draggingStartPoint.value.x;
                const dy = ev.pageY - draggingStartPoint.value.y;
                curAnchor.value.x = draggingStartPoint.value.x + draggingOffset.value.x
                    + dx / graph.value.scaling;
                curAnchor.value.y = draggingStartPoint.value.y + draggingOffset.value.y
                    + dy / graph.value.scaling;
            }
        };
        const onHoldStop = () => {
            if (isHolding.value) {
                document.removeEventListener('mousemove', holdMove);
                document.removeEventListener('mouseup', onHoldStop);
                isHolding.value = false;
            }
        };
        const onHoldStart = (ev, index) => {
            if (
                viewModel.value.editor.readonly ||
                !viewModel.value.connectionRenderer.supportsAnchors()
            ) return;

            draggingOffset.value = { x: ev.offsetX - ev.pageX, y: ev.offsetY - ev.pageY };
            draggingStartPoint.value = { x: ev.pageX, y: ev.pageY };
            const newAnchor = {
                x: ev.offsetX,
                y: ev.offsetY,
                id: Date.now(),
            };
            curAnchor.value =
                graph.value.addAnchor(newAnchor, props.connection, index);
            isHolding.value = true;
            document.addEventListener('mousemove', holdMove);
            document.addEventListener('mouseup', onHoldStop);
        };
        const onMouseDown = doubleClick(700, (ev) => {
            if (!viewModel.value.editor.readonly) {
                ev.preventDefault();
                graph.value.removeConnection(props.connection);
            }
        });

        const removeAnchor = (idx) => {
            if (viewModel.value.editor.readonly) return;
            graph.value.removeAnchor(props.connection, idx);
        };

        const onMouseCtrlDown = (ev, index) => {
            if (
                viewModel.value.editor.readonly ||
                !viewModel.value.connectionRenderer.supportsAnchors()
            ) return;
            ev.preventDefault();

            const newAnchor = {
                x: ev.offsetX,
                y: ev.offsetY,
                id: Date.now(),
            };
            graph.value.addAnchor(newAnchor, props.connection, index);
        };

        const newD = computed(() =>
            viewModel.value.connectionRenderer.render(
                props.x1,
                props.y1,
                props.x2,
                props.y2,
                props.connection,
                props.draggedNode,
            ),

        );

        const parsedNewD = computed(() => {
            const d = newD.value;

            if (Array.isArray(d) && d.length) {
                const parsedArray = [];
                for (let i = 0; i < d.length - 1; i += 1) {
                    parsedArray.push(`M ${d[i].x} ${d[i].y} L ${d[i + 1].x} ${d[i + 1].y}`);
                }
                return parsedArray;
            }
            return d;
        });

        const hasAnchors = computed(() =>
            props.connection.anchors !== undefined &&
            props.connection.anchors.length &&
            viewModel.value.connectionRenderer.supportsAnchors(),
        );

        return {
            cssClasses,
            parsedNewD,
            onMouseDown,
            onHoldStart,
            onHoldStop,
            onMouseCtrlDown,
            style,
            hasAnchors,
            removeAnchor,
            editorManager,
        };
    },
});
</script>
