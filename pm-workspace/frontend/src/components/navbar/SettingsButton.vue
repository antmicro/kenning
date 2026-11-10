<!--
Copyright (c) 2026 Antmicro <www.antmicro.com>

SPDX-License-Identifier: Apache-2.0
-->

<!--
Settings part of navigation top app bar. Allows the user to change general editor settings.
-->

<script>
import Cogwheel from '../../icons/Cogwheel.vue';

export default {
    model: {
        prop: 'openPanel',
        event: 'update:openPanel',
    },
    emits: ['hoverStart', 'hoverStop', 'onClicked', 'update:openPanel'],
    props: {
        mobileClasses: {
            required: true,
            type: Object,
        },
        hover: {
            required: true,
            type: Boolean,
        },
        openPanel: {
            type: Boolean,
        },
    },
    components: {
        Cogwheel,
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
        :class="['hoverbox', mobileClasses]"
        role="button"
        @click="() => this.$emit('onClicked')"
        @pointerover="() => this.$emit('hoverStart')"
        @pointerleave="() => this.$emit('hoverStop')"
        v-click-outside="() => this.$emit('update:openPanel', false)"
    >
        <Cogwheel :hover="hover" class="small_svg" />
        <div :class="['tooltip', mobileClasses]">
            <span v-if="!openPanel">Show settings</span>
            <span v-else>Hide settings</span>
        </div>
    </div>
</template>
<style lang="scss" scoped>
@import './simple_toggle_style.scss'
</style>
