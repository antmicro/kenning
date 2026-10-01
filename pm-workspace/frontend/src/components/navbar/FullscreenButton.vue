<!--
Copyright (c) 2026 Antmicro <www.antmicro.com>

SPDX-License-Identifier: Apache-2.0
-->

<!--
Fullscreen button in the top app bar.
-->

<script>
import { api as fullscreen } from 'vue-fullscreen';
import Expand from '../../icons/Expand.vue';
import Collapse from '../../icons/Collapse.vue';
import NotificationHandler from '../../core/notifications';

export default {
    emits: ['hoverStart', 'hoverStop'],
    props: {
        mobileClasses: {
            required: true,
            type: Object,
        },
        hover: {
            required: true,
            type: Boolean,
        },
    },
    components: {
        Expand,
        Collapse,
    },
    methods: {
        /**
         * Toggle fullscreen mode.
         *
         * For iframe, the allow="fullscreen" attribute must be set.
         */
        toggleFullscreen() {
            if (!fullscreen.isEnabled) {
                NotificationHandler.showToast('error', 'Fullscreen is not supported');
                return;
            }
            fullscreen.toggle();
        },
        getPanel() {
            return this.fullscreenPanel;
        },
    },
    data() {
        return {
            fullscreenPanel: {
                isOpen: false,
            },
        };
    },
    async mounted() {
        // Listen for fullscreen change
        document.addEventListener('fullscreenchange', () => {
            this.fullscreenPanel.isOpen = !fullscreen.isFullscreen;
        });
    },
};
</script>
<template>
    <div
        :class="['hoverbox', mobileClasses]"
        role="button"
        @click="toggleFullscreen"
        @pointerover="() => this.$emit('hoverStart', this.fullscreenPanel)"
        @pointerleave="() => this.$emit('hoverStop', this.fullscreenPanel)"
    >
        <Expand
            :hover="hover"
            class="small_svg"
            v-if="!fullscreenPanel.isOpen"
        />
        <Collapse :hover="hover" class="small_svg" v-else />
        <div :class="['tooltip', mobileClasses]">
            <span v-if="!fullscreenPanel.isOpen">Enable fullscreen</span>
            <span v-else>Disable fullscreen</span>
        </div>
    </div>
</template>
<style lang="scss" scoped>
@import './simple_toggle_style.scss'
</style>
