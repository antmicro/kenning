<!--
Copyright (c) 2026 Antmicro <www.antmicro.com>

SPDX-License-Identifier: Apache-2.0
-->

<!--
Defines the component responsible for opening/closing node details when within a node.
-->

<script>
import Sidebar from '../../icons/Sidebar.vue';

export default {
    emits: ['hoverStart', 'hoverStop', 'onClicked'],
    props: {
        mobileClasses: {
            required: true,
            type: Object,
        },
        hover: {
            required: true,
            type: Boolean,
        },
        editorManager: {
            required: true,
            type: Object,
        },
        openPanel: {
            required: true,
            type: Boolean,
        },
    },
    components: {
        Sidebar,
    },
    methods: {
        getRef() {
            return this.$refs.container;
        },
    },
};
</script>
<template>
    <div
        ref="container"
        v-if="editorManager.editor.getExposedProperties().length > 0
            || editorManager.editor.isInSubgraph()"
        :class="['hoverbox', mobileClasses]"
        role="button"
        @click="() => this.$emit('onClicked')"
        @pointerover="() => this.$emit('hoverStart')"
        @pointerleave="() => this.$emit('hoverStop')"
    >
        <Sidebar :hover="hover" class="small_svg"/>
        <div :class="['tooltip', mobileClasses]">
            <span v-if="!openPanel">Show graph details</span>
            <span v-else>Hide graph details</span>
        </div>
    </div>
</template>
<style lang="scss" scoped>
@import './simple_toggle_style.scss'
</style>
