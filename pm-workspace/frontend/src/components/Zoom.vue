<!--
Copyright (c) 2025 Antmicro <www.antmicro.com>

SPDX-License-Identifier: Apache-2.0
-->

<!--
This component is used to control the zoom level of the graph.
It emits 'center', 'zoomIn' and 'zoomOut' events when the respective buttons are clicked.
The component can be displayed in in a normal mode or in a floating mode (higher).
-->

<template>
    <div class="zoom-container export-hidden" :class="floatingStatus">
        <div
            role="button"
            class="zoom-center"
            @pointerover="() => updateHoverInfo('center')"
            @pointerleave="() => resetHoverInfo('center')"
            @click="$emit('center')"
        >
            <Crosshair :hover="isHovered('center')" />
        </div>
        <div class="zoom-control-wrapper">
            <div
                role="button"
                class="zoom-control"
                @pointerover="() => updateHoverInfo('plus')"
                @pointerleave="() => resetHoverInfo('plus')"
                @click="$emit('zoomIn')"
            >
                <Plus :hover="isHovered('plus')" />
            </div>
            <div
                role="button"
                class="zoom-control"
                @pointerover="() => updateHoverInfo('minus')"
                @pointerleave="() => resetHoverInfo('minus')"
                @click="$emit('zoomOut')"
            >
                <Minus :hover="isHovered('minus')" />
            </div>
        </div>
    </div>
</template>

<script>
import { computed } from 'vue';
import Minus from '../icons/Minus.vue';
import Plus from '../icons/Plus.vue';
import Crosshair from '../icons/Crosshair.vue';

export default {
    components: {
        Minus,
        Plus,
        Crosshair,
    },

    props: {
        floating: {
            type: Boolean,
            default: false,
        },
    },

    setup(props) {
        const floatingStatus = computed(() => ({
            floating: props.floating,
            normal: !props.floating,
        }));

        return { floatingStatus };
    },

    emits: ['zoomIn', 'zoomOut', 'center'],

    data() {
        return {
            hoverInfo: {
                isHovered: false,
                hoveredPanel: undefined,
            },
        };
    },

    methods: {
        isHovered(name) {
            return this.hoverInfo.hoveredPanel === name && this.hoverInfo.isHovered;
        },

        updateHoverInfo(name) {
            this.hoverInfo.hoveredPanel = name;
            this.hoverInfo.isHovered = true;
        },

        resetHoverInfo(name) {
            if (this.hoverInfo.hoveredPanel === name) {
                this.hoverInfo.hoveredPanel = undefined;
                this.hoverInfo.isHovered = false;
            }
        },
    },
};
</script>

<style lang="scss" scoped>
.zoom-container {
    display: flex;
    flex-direction: column;
    position: fixed;
    z-index: 3;
    align-items: center;
    gap: 22px;

    .zoom-center {
        width: 44px;
        height: 44px;
        display: flex;
        justify-content: center;
        align-items: center;
        cursor: pointer;
        background-color: $gray-600;
        border: 1px solid $gray-500;
        border-radius: 100%;
    }
    .zoom-control-wrapper {
        display: flex;
        flex-direction: column;
        background-color: $gray-600;
        border: 1px solid $gray-500;
        border-radius: 3px;

        .zoom-control {
            width: 44px;
            height: 44px;
            display: flex;
            justify-content: center;
            align-items: center;
            cursor: pointer;
        }

        .zoom-control:first-of-type {
            border-bottom: 1px solid $gray-500;
        }
    }
}
.zoom-container.floating {
    bottom: 70px;
    right: 30px;
}
.zoom-container.normal {
    bottom: 30px;
    right: 30px;
}
</style>
