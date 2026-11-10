<!--
Copyright (c) 2026 Antmicro <www.antmicro.com>

SPDX-License-Identifier: Apache-2.0
-->

<!--
Go back section of the navigation top app bar. Lets the user return to upper level graph or
exit from a subgraph.
-->

<script>
import Arrow from '../../icons/Arrow.vue';
import EditorManager from '../../core/EditorManager';

export default {
    components: {
        Arrow,
    },
    props: {
        navbarItems: {
            required: true,
            type: Array,
        },
        mobileClasses: {
            required: true,
            type: Object,
        },
        isHovered: {
            required: true,
            type: Function,
        },
        resetHoverInfo: {
            required: true,
            type: Function,
        },
        updateHoverInfo: {
            required: true,
            type: Function,
        },
        toggleGraphDetails: {
            required: true,
            type: Function,
        },
    },
    methods: {
        returnFromSubgraph() {
            this.editorManager.returnFromSubgraph();
            this.resetHoverInfo('subgraphReturn');
            this.toggleGraphDetails(true);
        },
        parentGraph() {
            return this.editorManager.getParentGraph();
        },
        goToParentGraph(graph) {
            this.editorManager.switchToGraph(graph);
            this.resetHoverInfo('parentGraph');
        },
    },
    data() {
        const editorManager = EditorManager.getEditorManagerInstance();
        return {
            editorManager,
        };
    },
};
</script>
<template>
    <div
        v-if="editorManager.editor.isInSubgraph() ||
            (parentGraph() && !editorManager.editor.isInSubgraph())"
        class="navigation">
        <div
            v-if="editorManager.editor.isInSubgraph()"
            :class="['hoverbox', mobileClasses]"
            role="button"
            @click="returnFromSubgraph"
            @pointerover="() => updateHoverInfo('subgraphReturn')"
            @pointerleave="() => resetHoverInfo('subgraphReturn')"
        >
            <Arrow
                rotate="down"
                :hover="isHovered('subgraphReturn')"
                color="white"
                class="small_svg"
            />
            <div :class="['tooltip', mobileClasses]">
                <span>Return from subgraph editor</span>
            </div>
        </div>
        <div
            v-if="parentGraph() && !editorManager.editor.isInSubgraph()"
            :class="['hoverbox', mobileClasses]"
            role="button"
            @click="goToParentGraph(parentGraph())"
            @pointerover="() => updateHoverInfo('parentGraph')"
            @pointerleave="() => resetHoverInfo('parentGraph')"
        >
            <Arrow
                rotate="up"
                :hover="isHovered('parentGraph')"
                color="white"
                class="small_svg"
            />
            <div :class="['tooltip', mobileClasses]">
                <span>Go to parent graph</span>
            </div>
        </div>
    </div>
</template>

<style lang="scss" scoped>
@import './simple_toggle_style.scss'
</style>
