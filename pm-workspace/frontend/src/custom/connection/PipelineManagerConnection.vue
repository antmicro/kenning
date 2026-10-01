<!--
Copyright (c) 2022-2026 Antmicro <www.antmicro.com>

SPDX-License-Identifier: Apache-2.0
-->

<!--
Wraps the connection, defines it's type and parameters.
Inherits from baklavajs/renderer-vue/src/connection/ConnectionView.vue
-->

<template>
    <ConnectionView
        ref="conn"
        :x1="d.x1"
        :y1="d.y1"
        :x2="d.x2"
        :y2="d.y2"
        :state="state"
        :connection="connection"
        :isHighlighted="isHighlighted"
        :draggedNode="props.draggedNode"
        :hover="hover"
        @mouseover="hover = true"
        @mouseleave="hover = false"
    ></ConnectionView>
</template>

<script>
import { defineComponent, ref, computed, watch, nextTick, onMounted, onBeforeUnmount } from 'vue'; // eslint-disable-line object-curly-newline
import { Components, useGraph } from '@baklavajs/renderer-vue';
import ConnectionView from './ConnectionView.vue';
import getDomElements from './domResolver';
import getPortCoordinates from './portCoordinates';
import { TemporaryConnectionState } from '../temporaryConnection.js';

export default defineComponent({
    extends: Components.ConnectionWrapper,
    props: {
        connection: { required: true },
        isHighlighted: { default: false },
        draggedNode: { required: true },
    },
    components: { ConnectionView },
    setup(props) {
        const conn = ref(null);
        const { graph } = useGraph();

        let resizeObserver;
        const d = ref({
            x1: 0,
            y1: 0,
            x2: 0,
            y2: 0,
        });

        // eslint-disable-next-line no-confusing-arrow
        const state = computed(() =>
            props.connection.isInDanger
                ? TemporaryConnectionState.FORBIDDEN
                : TemporaryConnectionState.NONE,
        );

        const hover = ref(false);

        /**
         * Check whether the connection path contains the x, y point
         *
         * @param elements result of `document.elementsFromPoint()` for a given x, y point.
         */
        const containsPoint = (elements) =>
            elements.includes(conn.value.$el.firstChild);

        const fromStubChange = computed(
            () => props.connection.from.offset,
        );
        const toStubChange = computed(
            () => props.connection.to.offset,
        );
        const getNodePositions = (nodeId) => {
            const node = graph.value?.findNodeById(nodeId);
            if (!node) return [];

            return [
                node.position,
                ...(node.views?.filter((v) => v?.position).map((v) => v.position) ?? []),
            ];
        };

        const fromNodePositions = computed(() => getNodePositions(props.connection.from.nodeId));
        const toNodePositions = computed(() => getNodePositions(props.connection.to.nodeId));

        const fromNode = computed(() => graph.value.findNodeById(props.connection.from.nodeId));
        const toNode = computed(() => graph.value.findNodeById(props.connection.to.nodeId));

        const getBuses = (intfs) => intfs.flatMap((i) => i.bus?.stubs ?? []);
        const getNodeSides = (node) => [
            ...Object.values(node.value?.inputs ?? {}),
            ...Object.values(node.value?.outputs ?? {}),
            ...getBuses(Object.values(node.value?.inputs ?? {})),
            ...getBuses(Object.values(node.value?.outputs ?? {})),
        ].map((io) => [io.side, io.sidePosition]);

        const fromNodeInterfacesSide = computed(() => getNodeSides(fromNode));
        const toNodeInterfacesSide = computed(() => getNodeSides(toNode));

        const updateCoords = () => {
            const from = getDomElements(props.connection.from);
            const to = getDomElements(props.connection.to);
            if (from.node && to.node) {
                if (!resizeObserver) {
                    resizeObserver = new ResizeObserver(() => {
                        updateCoords();
                    });
                    resizeObserver.observe(from.node);
                    resizeObserver.observe(to.node);
                }
            }

            const [x1, y1] = getPortCoordinates(from);
            const [x2, y2] = getPortCoordinates(to);
            d.value = {
                x1,
                y1,
                x2,
                y2,
            };
        };

        const currentViewName = computed(() => graph.value.editor.currentView);
        watch(currentViewName, async () => {
            await nextTick();
            updateCoords();
        });
        // If any side of any interface in from or to node changes we may need to
        // Rerender connections
        watch([fromNodeInterfacesSide, toNodeInterfacesSide], async () => {
            await nextTick();
            updateCoords();
        });

        watch(
            [fromNodePositions, toNodePositions],
            async () => {
                await nextTick();
                updateCoords();
            },
            { deep: true },
        );
        watch(
            [fromStubChange, toStubChange],
            async () => {
                await nextTick();
                updateCoords();
            },
            { deep: true },
        );

        onMounted(async () => {
            await nextTick();
            updateCoords();
        });

        onBeforeUnmount(() => {
            if (resizeObserver) {
                resizeObserver.disconnect();
            }
        });

        return {
            d,
            state,
            containsPoint,
            conn,
            hover,
            props,
        };
    },
});
</script>
