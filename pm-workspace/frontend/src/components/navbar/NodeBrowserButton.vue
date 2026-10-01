<!--
Copyright (c) 2026 Antmicro <www.antmicro.com>

SPDX-License-Identifier: Apache-2.0
-->

<!--
Defines the component responsible for opening/closing node browser.
-->

<script>
import Cube from '../../icons/Cube.vue';

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
        hideHud: {
            required: true,
            type: Boolean,
        },
        readonly: {
            required: true,
            type: Boolean,
        },
        openPanel: {
            required: true,
            type: Boolean,
        },
    },
    components: {
        Cube,
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
        v-if="!hideHud && !readonly"
        :class="['hoverbox', mobileClasses]"
        role="button"
        @click="() => this.$emit('onClicked')"
        @pointerover="() => this.$emit('hoverStart')"
        @pointerleave="() => this.$emit('hoverStop')"
    >
        <Cube :hover="hover" class="small_svg"/>
        <div :class="['tooltip-left', mobileClasses, 'tooltip']">
            <span v-if="openPanel">Hide node browser</span>
            <span v-else>Show node browser</span>
        </div>
    </div>
</template>
<style lang="scss" scoped>
@import './simple_toggle_style.scss'
</style>
