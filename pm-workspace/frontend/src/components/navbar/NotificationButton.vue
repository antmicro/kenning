<!--
Copyright (c) 2026 Antmicro <www.antmicro.com>

SPDX-License-Identifier: Apache-2.0
-->

<!--
Defines the component responsible for opening/closing notifications panel.
-->
<script>
import Bell from '../../icons/Bell.vue';

export default {
    props: {
        mobileClasses: {
            required: true,
            type: Object,
        },
        hideHud: {
            required: true,
            type: Boolean,
        },
        notificationCount: {
            required: true,
            type: Number,
        },
        hover: {
            required: true,
            type: Boolean,
        },
        openPanel: {
            required: true,
            type: Boolean,
        },
    },
    components: {
        Bell,
    },
    methods: {
        getRef() {
            return this.$refs.container;
        },
    },
    emits: ['onClicked', 'hover', 'hoverStop'],
};
</script>
<template>
    <div
        ref="container"
        v-if="!hideHud"
        :class="['hoverbox', mobileClasses]"
        role="button"
        @click="() => this.$emit('onClicked')"
        @pointerover="() => this.$emit('hover')"
        @pointerleave="() => this.$emit('hoverStop')"
    >
        <Bell
            id="navbar-bell"
            :color="
                (notificationCount > 0) ?
                'green' : 'gray'
            "
            :hover="hover"
            class="small_svg"
        />
        <div
            v-if="openPanel"
            :class="['tooltip', mobileClasses]"
        >
            <span>Hide notifications</span>
        </div>
        <div
            v-else :class="['tooltip', mobileClasses]"
        >
            <span>Show notifications</span>
        </div>
    </div>
</template>
<style lang="scss" scoped>
@import './simple_toggle_style.scss'
</style>
